# WikiLense design (phases 1 to 3, 2026-09-17)

Retrieval over Wikipedia inside MariaDB 11.8: chunk pages, embed chunks, store them in a
`VECTOR` column with a cosine `VECTOR INDEX`, query with `VEC_DISTANCE_COSINE`, and combine the
semantic search with ordinary SQL predicates and joins in the same statement. FEVEROUS claims are
the queries; their gold evidence is the ground truth for recall.

This file is the contract between modules. Values marked *provisional* were working choices of
phases 1 and 2, made so that coding did not stall. The experiments of phase 3
(`scripts/run_experiments.py`, numbers in `results/SUMMARY.md`) fixed them; the section "Phase 3
outcomes and chosen defaults" at the end says what was chosen and why, and the README repeats it.

## Repository layout

```
wikilense/                 Python package
  config.py                Settings from environment / .env (DB connection, model, chunk and search parameters)
  corpus.py                Read data/corpus/*.jsonl; element-id parsing; corpus selection logic
  wikitext.py              FEVEROUS page parsing: page order, section tree, text units, links, markup cleaning
  chunking.py              Sentence-window chunker -> Chunk objects that remember their sentence keys
  embedding.py             Embedder (sentence-transformers) -> float32 numpy; vector <-> text helpers
  db.py                    PyMySQL connection, schema apply/reset, vector parameter helpers, bulk inserts
  ingest.py                Pipeline: corpus -> page/section/sentence/link/claim tables -> chunks -> embeddings
  search.py                search() (strategies inline/overfetch/none/rrf), filters, join back to sentences
  evaluate.py              Recall and latency harness
  cli.py                   `wikilense` command: init-db, ingest, query, eval, serve
  web.py                   Minimal FastAPI query page
scripts/build_corpus.py    Select pages and claims from a shard -> data/corpus/
scripts/fetch_wiki_shard.py  Download one Wikipedia shard (existing)
sql/schema.sql             DDL, the single source of truth; db.apply_schema() executes it
data/corpus/               pages.jsonl, claims.jsonl, MANIFEST.md (committed, a few MB)
data/feverous/             raw downloads (not committed)
tests/                     pytest; markers `db` (needs MariaDB) and `slow` (needs the model)
results/                   harness outputs (committed)
docs/DESIGN.md             this file
docs/ARCHITECTURE.md       beginner's guide: pipeline, every file, one query end to end
```

Rules for all code: Python 3.10+, standard library where it is enough, type hints, docstrings that
say what a function returns, no SQL built by string formatting (parameters only; identifiers only
from fixed literals in the code), no credentials in code, no debug flags left on. Tests live in
`tests/` and are real tests.

## Data contracts

**Page record** (`data/corpus/pages.jsonl`): the unmodified FEVEROUS page object, one per line.
Keys: `title`; `order` (element keys in page order); `sentence_N` (string); `section_N`
(`{"value": heading, "level": int}`); `table_N` (`{"table": [[cell]], "type": ..., "caption"?}`,
cell = `{"id", "value", "is_header", "row_span", "column_span"}`); `list_N`
(`{"list": [{"id", "value", "level", "type"?}], "type"}`).

**Claim record** (`data/corpus/claims.jsonl`): the FEVEROUS record (`id`, `claim`, `label`,
`evidence`, `annotator_operations`, `challenge`) plus `"split": "train" | "dev"`.
`evidence` is a list of alternative evidence sets; each has `content` (list of element ids) and
`context`. Only `content` counts as evidence.

**Element id**: `<page title>_<type>_<numbers>` with type in `sentence`, `cell`, `header_cell`,
`table_caption`, `item`. Parse with a regex anchored at the end,
`^(?P<page>.+?)_(?P<type>sentence|cell|header_cell|table_caption|item)_(?P<nums>\d+(?:_\d+)*)$`
(non-greedy page group: with a greedy one every `header_cell` id parses as page `X_header`, type
`cell`; measured on all 383,137 train+dev ids), so that a title containing an underscore still parses. `corpus.parse_element_id(el)` returns
`(page_title_nfc, type, key)` where `key` is the page-object key (`sentence_7`, `item_0_1`,
`cell_0_1_1` ...). Normalise page titles to Unicode NFC before any comparison.

**Text unit**: one sentence (`sentence_N`) or one list item (`item_N_M`), in page order. Tables
and captions are not text units in phase 1 (recorded in page statistics only; a documented
limitation).

## Parsing rules (`wikitext.py`)

- Walk `order`. Keep a section stack keyed by `level`. Section ordinal 0 is the lead (heading
  `""`, level 1, path `""`). Every later `section_N` opens a new section with `path` =
  headings joined by `" > "` from the top of the stack.
- Emit `TextUnit(element_key, ordinal, section_ordinal, text, raw)` for every `sentence_N` and
  every item of every `list_N`, `ordinal` counting text units in page order from 0.
- `clean_text(raw)`: `[[target|shown]]` -> `shown`; `[[target]]` -> `target` with `_` -> space;
  remove any remaining `[[`/`]]`; collapse whitespace; strip. Measure residual markup over the
  corpus and report it.
- `extract_links(page)` from sentences, list items and table cells: target = text before `|`,
  drop a `#fragment`, `_` -> space, NFC. Return `(to_title, source_element_key)` pairs, in page
  order, duplicates kept.
- `page_stats(page)`: `n_sentences`, `n_items`, `n_words` (words over cleaned text units),
  `n_chars`, `n_sections`, `n_tables`, `n_lists`.

## Chunking rules (`chunking.py`)

- `chunk_units(units, max_words, overlap_units=1)` is a pure function over one page's text
  units; ingest passes `Settings.chunk_max_words` and `Settings.chunk_overlap_units`. Chosen with
  the experiments: `max_words` 240 (120 in phases 1 and 2; equal recall at equal retrieved text
  with half the vectors, `results/SUMMARY.md` section 4 and Phase 3 below) and `overlap_units` 1
  (kept, not swept). A chunk never crosses a section boundary.
- Only *chunkable* units take part (`TextUnit.chunkable`): a unit that is empty after cleaning, or
  a hatnote (`wikitext.HATNOTE_RE`: "Main article: X", "See also: Y", "For other uses, see Z",
  ...), is in no chunk and in no `chunk_sentence` row, but stays a `sentence` row so evidence ids
  resolve (hatnotes since phase 5: 1,316 of the 33,637 units of the corpus).
- Fill a chunk with consecutive units until adding the next unit would exceed `max_words`; a
  single unit longer than `max_words` becomes its own chunk. The next chunk starts
  `overlap_units` units before the end of the previous chunk, but only when the previous chunk
  has more than `overlap_units` units (no infinite loops, no duplicate chunks).
- `Chunk(ordinal, section_ordinal, element_keys, text, n_words)`; `text` is the cleaned units
  joined by a single space; `ordinal` counts chunks per page from 0.
- The text that is embedded is `f"{title} > {section_path}: {text}"` when the section path is
  non-empty, else `f"{title}: {text}"`. The stored `chunk.text` stays the plain text. Ingest adds
  the prefix; it was tested against no prefix and kept (exact rankings: article recall@10 74/75
  both ways, evidence recall@20 63/66 with the prefix against 61/66 without; see results/SUMMARY.md section 5).

## Embedding (`embedding.py`)

- Model: `BAAI/bge-small-en-v1.5`, 384 dimensions, MIT licence, 33M parameters (chosen in phase
  1 and kept; no other model was measured, so the choice rests on the reasons below).
  Reason: strong on passage retrieval for its size, fast enough on CPU for a grader, and it fits
  the 8 GB GPU with room to spare. Normalised embeddings, so cosine distance equals
  `1 - dot`.
- `Embedder(model_name=..., device=None, revision=None)` loads lazily; the default model is
  pinned to the Hub commit `config.DEFAULT_EMBEDDING_REVISION` (5c38ec7c405e..., the snapshot
  every result was measured with; since 2026-09-23), so a later update of the model repository
  cannot change the vectors; `dim` property; `embed_passages(texts,
  batch_size=64) -> np.ndarray[float32, (n, dim)]`; `embed_queries(texts)` prepends the model's
  query instruction (`"Represent this sentence for searching relevant passages: "` for bge).
- Helpers, independent of the model: `vector_to_text(v) -> "[...]"` for `VEC_FromText` and
  `text_to_vector(s)` for what `VEC_ToText` returns. The bytes bound for a `VECTOR` parameter
  (float32 little-endian, the MariaDB storage format) come from `db.vec_param(v)`, and
  `db.vec_from_bytes(b)` reads a stored vector back (until 2026-09-23 `embedding.vector_to_bytes`
  and `bytes_to_vector` duplicated them, used by no code).

## Database (`sql/schema.sql`, `db.py`, `config.py`)

All tables InnoDB, `utf8mb4`. Title columns use `utf8mb4_bin` so that matching is exact.

| Table | Columns (PK first) | Notes |
|---|---|---|
| `page` | `page_id`, `title` UNIQUE, `n_sentences`, `n_items`, `n_words`, `n_chars`, `n_sections`, `n_tables`, `n_lists`, `revision_id` NULL, `last_edited` NULL | index on `n_words` for length filters; `revision_id` / `last_edited` (UTC) name the December 2020 Wikipedia revision the text comes from (since 2026-10-22, `data/corpus/page_meta.jsonl`), index on `last_edited` for the edit-date filters |
| `section` | `section_id`, `page_id` FK, `ordinal`, `heading`, `level`, `path` | UNIQUE (`page_id`, `ordinal`); index on `heading` |
| `sentence` | `sentence_id`, `page_id` FK, `section_id` FK, `element_key`, `ordinal`, `text` | UNIQUE (`page_id`, `element_key`); one row per text unit (sentence or list item) |
| `chunk` | `chunk_id`, `page_id` FK, `section_id` FK, `ordinal`, `text`, `n_words`, `embedding VECTOR(384) NOT NULL` | UNIQUE (`page_id`, `ordinal`); `VECTOR INDEX (embedding) M=16 DISTANCE=cosine` (M=6 in phases 1 and 2; 16 chosen with the M sweep, `results/SUMMARY.md` section 3); `FULLTEXT KEY ft_chunk_text (text)` for the `rrf` strategy (since phase 5) |
| `chunk_sentence` | (`chunk_id`, `sentence_id`) | the chunk-to-sentence map; index on `sentence_id` |
| `link` | `link_id`, `from_page_id` FK, `to_title`, `to_page_id` NULL FK, `source_element` | `to_page_id` resolved when the target is in the corpus; indexes on `from_page_id`, `to_page_id`, `to_title` |
| `category` | `category_id`, `name` UNIQUE (`utf8mb4_bin`) | Wikipedia category names, no "Category:" prefix (since 2026-10-22) |
| `page_category` | (`page_id`, `category_id`) both FK | the many-to-many map; index (`category_id`, `page_id`) for the category filter |
| `claim` | `claim_id` (FEVEROUS id), `split`, `text`, `label`, `challenge` | |
| `claim_evidence` | (`claim_id`, `evidence_set`, `position`), `element_id`, `page_title`, `element_type`, `page_id` NULL, `sentence_id` NULL | resolved ids when the page / text unit is in the corpus; indexes on `page_id`, `sentence_id` |
| `ingest_meta` | `key`, `value` | embedding model, dim, chunk parameters, index M, corpus sha256 (pages, claims, page_meta or `none`), ingested_at |

Vector dimension 384 is written literally in the DDL and checked against `Embedder.dim` at
ingest; a mismatch is an error with a clear message.

`config.Settings` (from environment, `.env` loaded with python-dotenv): `WIKILENSE_DB_HOST`,
`WIKILENSE_DB_PORT`, `WIKILENSE_DB_USER`, `WIKILENSE_DB_PASSWORD`, `WIKILENSE_DB_NAME`,
`WIKILENSE_EMBEDDING_MODEL` (default above), `WIKILENSE_CHUNK_MAX_WORDS` (240; 120 until
2026-09-17), `WIKILENSE_CHUNK_OVERLAP_UNITS` (1), `WIKILENSE_VECTOR_DIM` (384) and
`WIKILENSE_EF_SEARCH` (100, `Settings.ef_search`: the `mhnsw_ef_search` value the application
passes per query, 1 to 10000; 0 keeps the server's session value, whose default stays 20;
anything else is a `SettingsError` when the settings load), and `WIKILENSE_TEST_DB_NAME`
(default the database name plus `_test`; the test suite resets that database's schema, so
`tests/conftest.py` refuses a name that equals the main database or does not end in `_test`). The
index `M` (16; 6 until 2026-09-17) is written literally in `sql/schema.sql` and named
`db.VECTOR_INDEX_M` in the code; it is not a setting (`WIKILENSE_INDEX_M` existed until
2026-09-23 but only changed what `ingest_meta` recorded, never the index).

`db.connect(settings) -> pymysql.Connection` (autocommit off, `charset="utf8mb4"`).
`db.apply_schema(conn, reset=False)` runs `sql/schema.sql` statement by statement; `reset=True`
drops the tables first in dependency order. `db.vec_param(v)` returns the bytes to bind for a
`VECTOR` parameter. Binding rule to verify with a round-trip test: insert with a bytes parameter,
read back with `VEC_ToText`, and compare; if PyMySQL's `_binary` literal does not work for
`VECTOR`, use `VEC_FromText(%s)` with `vector_to_text`. Queries are parameterised (`%s`) only.

Index use rule from the MariaDB docs: the index drives a query only for
`ORDER BY VEC_DISTANCE_COSINE(embedding, <constant>) LIMIT n` sorted ascending; wrapping the
distance in an expression, or a bare `WHERE VEC_DISTANCE(...) < t`, defeats it. The system
variable `mhnsw_ef_search` (default 20, session scope) sets the minimum number of candidates the
index looks at; the application passes `Settings.ef_search` (100) per query, see Phase 3.
`mhnsw_max_cache_size` (16 MB default, global) bounds the index cache; `docker-compose.yml` starts
the server with 512 MB because the M=6 graph of the 120-word ingest (8,658 chunks) is already 16 MB
on disk (the size does not change the results, see results/SUMMARY.md section 1).

## Corpus selection (`corpus.py`, `scripts/build_corpus.py`)

Inputs: one or more shards, the train and dev claim files. Rule:

1. A claim is *in-shard* when every page cited in every `content` id of every evidence set is a
   shard title (NFC). Expected from earlier checks: 75 claims (10 dev, 65 train).
2. Evidence pages = union of those pages. Expected: 54.
3. Filler pages: link targets of the evidence pages that are shard titles and not evidence pages,
   ranked by the number of distinct evidence pages linking to them (descending), then title
   (ascending); take up to 46. If fewer exist, continue in shard order with pages of at least 20
   sentences. Reason: linked pages are realistic distractors and make the "linked from" SQL filter
   meaningful.
4. Write `pages.jsonl` (raw shard lines, shard order), `claims.jsonl` (raw records plus `split`),
   and `MANIFEST.md` (rule, counts, sizes, SHA-256 of both files, the page titles).

Deterministic, standard library only, recompute every count from the files.

## Search (`search.py`), phase 2

- The k-nearest-neighbour core, `search.KNN_SQL`: `SELECT chunk_id, VEC_DISTANCE_COSINE(embedding,
  %s) AS distance FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s` (planned as a
  `knn()` function; it became strategy `none` of `search()`, which joins it back to `page` and
  `section`).
- Hybrid variants, each returning the same row shape, each with its `EXPLAIN` recorded:
  predicate on page length (`page.n_words >= %s`), on section heading (`section.heading LIKE %s`
  or `path`), on link structure (planned as `chunk.page_id IN (SELECT to_page_id FROM link WHERE
  from_page_id = ...)`, built as a join on a `SELECT DISTINCT` derived table, see Phase 2
  outcomes), and combinations; joined back to `page`, `section` and, through
  `chunk_sentence`, to the `sentence` rows of each hit.
- Filtering strategies to compare: predicate in the same statement; over-fetch `k * f` by the
  index and filter in an outer query; `mhnsw_ef_search` raised per session.

## Evaluation (`evaluate.py`), phase 2

- Queries: the corpus claims. Ground truth per claim: gold pages (any evidence set) and, for
  claims with a sentence-only evidence set, the gold text units of that set.
- Metrics at k in {1, 3, 5, 10, 20}: article recall@k (a gold page is among the pages of the
  top-k chunks), evidence recall@k (every unit of at least one sentence-only evidence set is
  covered by the top-k chunks), unit coverage@k (share of gold units covered), and latency
  (query embedding time and SQL time reported separately; p50 / p95 over repeated runs, warm).
- Sweeps: `mhnsw_ef_search`, index `M` (rebuild), chunk `max_words` at equal retrieved text
  (`max_words * k`), embedding prefix on/off.
- Output: JSON and a Markdown table in `results/`, with the machine, versions and parameters.

## Phase 1 outcomes (measured 2026-09-17, all tests green: 95 passed)

Corrections to the rules above, forced by the data, and the numbers behind the phase-1 choices
(provisional at the time; the final ones are under Phase 3).

- **Corpus**: 75 in-shard claims (65 train, 10 dev; 7 SUPPORTS, 58 REFUTES, 10 NEI), 54 evidence
  pages, 46 filler pages all taken from links (202 candidates), 100 pages, `pages.jsonl` 8.0 MB
  (large fillers such as Latin, Amsterdam, Lebanon). 65 claims have a sentence-only evidence set;
  all 75 cite a single page. SHA-256 of both files is in `data/corpus/MANIFEST.md`; the build is
  byte-identical on re-run (about 5 s).
- **Parsing** (whole shard, 9,996 pages): 514,984 sentences, 139,131 list items, 654,115 text
  units; 2,664 units are whitespace-only (empty after cleaning) and are kept as `sentence` rows so
  evidence ids resolve, but they belong to no chunk. Residual markup after `clean_text` is 86 units
  (0.013%), all real text (`|` in scores, formulas, IPA). Links: 754,400 (sentences 445,886, cells
  141,121, items 167,393); links with an empty target (`[[#Section|x]]`, `[[]]`) and bare links
  truncated by the sentence splitter are skipped; table captions are not read (139 links, no
  element id). `page_stats["n_sections"]` counts headings, so the `section` table holds
  `n_sections + 1` rows per page (lead included). Section stack pops every level >= the new one.
- **Chunking** (max_words 120, the phase-1 setting, overlap 1, whole shard; the chosen default is
  240, see Phase 3): 165,607 chunks; per page min/median/max
  1 / 7 / 336; words per chunk min/median/p95/max 1 / 100 / 119 / 543 (mean 87.3); 5.6% of chunks
  are a single unit; 2 chunks (0.001%) exceed 512 tokens at 1.3 tokens per word (two long list
  items). Two guards beyond the rule: a chunk that would only repeat units of the previous chunk
  is not emitted (1,004 on the shard) and a chunk with zero words is not emitted (100).
- **Database** (MariaDB 11.8.9, PyMySQL 1.2.0; 1.2.3 since 2026-09-22): binding a `VECTOR` parameter as plain bytes works
  only on a connection opened with `binary_prefix=True` (`db.connect` sets it) with PyMySQL 1.2.0 to
  1.2.2; PyMySQL 1.2.3 and later escape bytes as `_binary X'...'` by themselves (found on 2026-09-22
  when CI, which installs the newest driver, failed the test that documented the rule); otherwise MariaDB
  rejects it (errors 1292 / 4079). `VEC_FromText(%s)` and `UNHEX(%s)` work on any connection. The
  knn query `ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT n` gives `EXPLAIN` type `index`, key
  `embedding`; the alias form `ORDER BY distance` is identical. Resolution references
  (`link.to_page_id`, `claim_evidence.page_id`, `claim_evidence.sentence_id`) use `ON DELETE SET
  NULL`, ownership references cascade. Explicit indexes on every FK column. `claim.split`,
  `claim.label`, `claim_evidence.element_type` are ENUMs over the complete value sets found in the
  files. `section.heading` uses the server default collation (`utf8mb4_uca1400_ai_ci`), so a
  heading filter with `LIKE` is case- and accent-insensitive on purpose. `mhnsw_ef_search` outside
  its range is clamped with a warning, not an error. Schema apply 59 ms, reset 87 ms.
- **Embedding** (bge-small-en-v1.5, revision 5c38ec7c405e, 33.4M parameters, 134 MB on disk):
  RTX 4060 1,059 passages/s, CPU 78 passages/s for 100-word texts; peak GPU memory 253 MiB; the
  outputs are bitwise deterministic on both devices; model load 3 to 6 s from a warm cache.

## Ingest (`ingest.py`), contract for phase 2

`run_ingest(settings, corpus_dir=DEFAULT_CORPUS_DIR, reset=True, embedder=None, batch_size=64,
use_prefix=True, progress=True) -> IngestReport` with the counts `n_pages, n_sections,
n_sentences, n_units_empty, n_units_hatnote, n_chunks, n_links, n_links_resolved,
n_links_skipped, n_claims, n_evidence, n_evidence_page_resolved, n_evidence_sentence_resolved,
n_pages_with_meta, n_categories, n_page_categories` (the last three since 2026-10-22) and
`seconds` per stage (`schema, parse, embed, load, resolve, analyze, total`).

Before the database is touched: refuse the test database (`settings.test_db_name`), a
`settings.vector_dim` other than the schema's literal, `chunk_max_words < 1` and
`chunk_overlap_units < 0` (`IngestError`), then load the embedding model and refuse a dimension
other than `settings.vector_dim`. A mistake in `.env` therefore never drops the ingest that is in
place (until 2026-09-23 the model and the chunk settings were checked after the reset). Then seven
steps, each committed when it completes; a failure rolls back the uncommitted rows (when the
connection is still open) and closes the connection:

1. `apply_schema(reset)`. `reset=False` is for a schema made by `init-db` and refuses a database
   that already holds pages.
2. Parse every page (`parse_page`, `chunk_page`); insert `page`, `section` (lead included),
   `sentence` (every unit, empty ones and hatnotes too), `link` (`to_title` over 255 characters
   skipped and counted). Keep chunks in memory with their `embedding_text` (prefix on by default).
   Since 2026-10-22, with `page.revision_id` / `last_edited` and the `category` /
   `page_category` rows from `page_meta.jsonl` when the corpus directory has it (NULL and empty,
   with a warning, otherwise; a malformed file is an `IngestError`).
3. Embed all chunk texts in batches (`embed_passages`), then insert `chunk` rows with
   `vec_param` bytes and `chunk_sentence` rows through `insert_rows`.
4. Resolve `link.to_page_id` with one `UPDATE link JOIN page ON page.title = link.to_title`.
5. `ANALYZE TABLE` for `chunk`, `page`, `section`, `link`, `category` and `page_category`, so the
   optimizer plans the search
   statements from fresh statistics (Phase 2 outcomes: stale statistics once lost the index).
6. Insert `claim` and `claim_evidence` (`parse_element_id`; `page_id` by title, `sentence_id` by
   (`page_id`, `element_key`); cells and captions keep `sentence_id` NULL).
7. Write `ingest_meta`: `embedding_model`, `embedding_revision`, `embedding_dim`, `embedding_prefix`, `chunk_max_words`,
   `chunk_overlap_units`, `index_m` and `index_distance` (read from `SHOW CREATE TABLE chunk`,
   so they describe the index that is in place), `hatnote_pattern`, `n_units_hatnote`,
   `analyze_tables`, `analyze_seconds`, `corpus_dir`, `corpus_pages_sha256`,
   `corpus_claims_sha256`, `corpus_page_meta_sha256` (`none` without the file),
   `mariadb_version`, `wikilense_version`, `ingested_at` (UTC ISO 8601).

## Search (`search.py`), contract for phase 2

- `Hit(chunk_id, page_id, title, section_path, chunk_ordinal, n_words, distance, text,
  score=None)`; `score` is the fused score of `rrf`, `distance` always the cosine distance.
- `Filters(min_words=None, max_words=None, heading_like=None, path_like=None, linked_from=None,
  links_to=None, titles=None, category=None, edited_since=None, edited_before=None)`; every field
  optional; `heading_like` and `path_like` are SQL LIKE patterns supplied by the caller
  (parameters, never interpolated). `category` (exact name) joins a DISTINCT derived table of
  `page_category`; `edited_since` / `edited_before` (a `date` or `datetime`, read as UTC) compare
  `page.last_edited` (since 2026-10-22, see "Page metadata and the brief's four filters").
- `search(conn, qvec, k=10, filters=None, strategy="inline", overfetch=10, ef_search=None,
  query_text=None) -> list[Hit]` with strategies (`search.STRATEGIES`): `inline` (predicates and
  joins in the one statement that carries `ORDER BY VEC_DISTANCE_COSINE(...) LIMIT k`),
  `overfetch` (an inner index-driven query with `LIMIT k * overfetch`, filtered and re-limited in
  the outer query), `none` (no filters), `rrf` (since phase 5: the vector top-N and the FULLTEXT
  top-N of `query_text`, N = `k * overfetch`, each ranked with `ROW_NUMBER()` and fused by
  reciprocal rank fusion, `1 / (60 + rank)` summed over both lists; the filters apply to the
  fused list). Ties are broken by `chunk_id`: in Python for the vector strategies (a second
  `ORDER BY` key loses the index), in SQL for `rrf`. `overfetch` is at most `MAX_OVERFETCH`
  (1000). `ef_search` (1 to 10000, the server's range; None leaves the session alone) sets the
  session variable for that call and restores it afterwards, also when the statement fails;
  `resolve_ef_search(requested, default)` gives the CLI, the web page and the harness one rule:
  a missing value takes `Settings.ef_search`, and 0 means the server's session value.
- `explain_search(conn, ...)` returns the `EXPLAIN` rows for the same statement.
- `hit_sentences(conn, chunk_ids) -> dict[chunk_id, list[(element_key, text)]]` through
  `chunk_sentence`, and `page_summary(conn, page_id)`.
- Every SQL statement is a named module constant (or built from named fragments) so the README
  can quote them verbatim.

## Evaluation (`evaluate.py`), contract for phase 2

- `load_ground_truth(conn) -> list[ClaimTruth(claim_id, split, label, text, gold_pages: set[int],
  sentence_only_sets: list[set[int]])]` from `claim` and `claim_evidence` (resolved ids only; a
  set counts as sentence-only when every element is a `sentence` or `item` with a resolved
  `sentence_id`).
- `evaluate(conn, embedder, ks=(1, 3, 5, 10, 20), repeats=5, strategy="none", filters=None,
  ef_search=None, overfetch=10, claim_filters=None, settings=None) -> EvalResult`; `ef_search`
  is an int (1 to 10000), None (`settings.ef_search`, where 0 keeps the server's value) or
  `SERVER_DEFAULT_EF_SEARCH` ("server"); `claim_filters` gives per-claim Filters (the oracle
  `oracle_title_filters`). Per k: article recall (a gold page among the pages of the
  top-k chunks), evidence recall (every unit of at least one sentence-only set covered by the
  top-k chunks, over the 66 eligible claims: 65 was the phase-1 count before list items counted
  as text units, see Phase 2 outcomes), unit coverage (share of gold units covered), and
  latency: embedding time and SQL time separately, p50 / p95 / mean over `repeats` warm runs of
  every claim, plus the parameters, `ingest_meta`, machine and versions. The metrics at `k`
  come from the `LIMIT k` search of each claim, the statement whose latency is reported at `k`
  (until 2026-09-23 they were the first `k` hits of the `LIMIT max(ks)` search, which is the same
  list for `none` and `inline` but not for `overfetch` and `rrf`, whose inner candidate list is
  `k * overfetch` long; `prefix_mismatches` counts the claims where the two differ).
- `write_results(result, out_dir="results", name=...)` writes `<name>.json` and `<name>.md` (a
  table per metric).
- Sweeps are separate CLI runs (`--ef-search`, `--strategy`, `--k`), not hidden loops.

## CLI and web (`cli.py`, `web.py`), contract for phase 2

`wikilense init-db [--reset]`, `wikilense ingest [--corpus-dir] [--no-reset] [--batch-size]
[--no-prefix]`, `wikilense query "text" [--k 5] [--min-words N] [--max-words N] [--heading
PATTERN] [--path PATTERN] [--linked-from TITLE] [--links-to TITLE] [--title TITLE ...]
[--category NAME] [--edited-since YYYY-MM-DD] [--edited-before YYYY-MM-DD]
[--strategy inline|overfetch|none|rrf] [--overfetch N] [--ef-search N|0|server] [--sentences]
[--explain] [--json]`, `wikilense eval [--k 1,3,5,10,20] [--repeats 5] [--strategy ...]
[--overfetch N] [--ef-search N|0|server] [--out results] [--name NAME]`, `wikilense serve [--host
127.0.0.1] [--port 8000]`. Exit codes: 0 success, 1 an expected failure (one line on stderr),
2 bad arguments, 130 interrupted. The web page is one HTML form
(query, k, filters) that calls `GET /api/search` and shows hits with title, section path,
distance, the chunk text and the SQL that ran; no JavaScript framework, no external assets. Since
2026-10-22 a link with the form's parameters (`/?q=...&category=...`) fills the form and runs
the search (the README's screenshot, `docs/img/web_page.png`).

## Phase 2 outcomes (measured 2026-09-18, 199 tests green, commit b517a67)

- **Ingest** of the 100-page corpus: 3,173 sections, 33,637 sentence rows (372 empty), 8,868
  chunks (per page min/median/max 2 / 85 / 232), 39,155 chunk-sentence rows, 40,542 links of
  which only 373 resolve inside the corpus (78 distinct target pages), 75 claims, 114 evidence
  rows (114 with a page, 87 with a sentence: the 27 cell ids stay unresolved). 30 s in total, 26 s
  of it model load and embedding on the GPU. `chunk` is 23.6 MB of data and 9.3 MB of index.
- **Evidence-recall denominator is 66, not 65**: claim 65842's set is two sentences and two list
  items, all resolved; list items count as text units.
- **Search statements**: the inline statement uses `STRAIGHT_JOIN` with `chunk` first, because
  with a plain `JOIN` the optimizer started from `page` when table statistics were stale (right
  after a bulk insert) and lost the vector index. The link filters are joins on a materialised
  `SELECT DISTINCT to_page_id ...` derived table, because the `IN (subquery)` form became a
  semi-join that started from `link` and lost the index. `ORDER BY distance, chunk_id` also loses
  the index, so the chunk-id tie-break is done in Python. A wrong-dimension query vector returns
  NULL distances, not an error.
- **How MariaDB 11.8.9 uses the vector index** (the central finding; cost model corrected on
  2026-09-17 with the phase-3 runs, `results/SUMMARY.md` sections 2 and 6):
  - A bare `ORDER BY VEC_DISTANCE_COSINE(...) LIMIT k` (strategies `none` and `overfetch`) is the
    HNSW search proper: sub-millisecond on 8,868 chunks (SQL p50 at k=1 0.62 ms at ef_search 20,
    0.76 ms at 100, 0.91 ms at 400), it reads exactly `LIMIT` rows, and its accuracy is set by
    `mhnsw_ef_search` (article recall@10 61/75 at ef 20, 73/75 at ef 100, 74/75 at ef 200 on the
    M=6 index).
  - As soon as another table is joined in the same statement (strategy `inline`), `EXPLAIN` still
    reports `type index, key embedding`, but the server ranks exactly over the index: about 4 ms
    on 8,868 chunks (p50 4.26 ms at k=1, 4.47 ms at k=20), independent of `k` and of `ef_search`
    (hit lists identical at ef 20 and ef 100 for 75/75 claims, equal to the exact ranking), and
    growing with the table. On top of that ranking the walk continues until `k` rows have passed
    the joins and predicates: `ANALYZE` reports `r_rows` = `k` for `chunk` when nothing is
    filtered (tests/test_search.py), the walk gets longer the more the filter rejects (the oracle
    `titles` filter below: p95 15 ms), and it degenerates to a walk over the whole index when the
    filter matches nothing. `overfetch` stays bounded (p50@10 1.20 ms with overfetch 10 under the
    `min_words` filter) but returns fewer than `k` rows when the filter is selective: under
    `heading LIKE '%History%'` (268 of 8,868 chunks pass) 66 of 75 queries got fewer than 10 rows
    with overfetch 10 and 23 with overfetch 50, none with `inline`.
  - The `IN (subquery)` form of the link filter (`chunk.page_id IN (SELECT to_page_id FROM link
    ...)`) becomes a semi-join that starts from `link` and loses the vector index; it is faster
    only for a very selective filter, where sorting the few matching chunks by distance costs less
    than the index walk. `search.py` therefore joins a materialised `SELECT DISTINCT to_page_id
    ...` derived table, which keeps the index for every selectivity.
  - Baseline (`none`, ef_search 20, k=1/3/5/10/20): article recall 56/62/64/68/68 of 75, evidence
    recall 22/31/33/39/48 of 66, SQL p50 0.6 to 0.8 ms, query embedding 4.6 ms on the GPU. An
    exact ranking of the same vectors finds 66/71/73/74/74 pages, so the gap is HNSW
    approximation, not the embedding. With `mhnsw_ef_search = 100`: 65/70/72/74/74 pages and
    31/40/43/49/59 evidence sets at the same latency (p50 0.65 ms at k=1).
  - Oracle with the gold page as a `titles` filter: the evidence is in the first chunk for 35 of
    66 claims, within 5 for 50, within 20 for 65; the inline filtered statement costs 4.2 ms p50,
    15 ms p95, because the walk continues until `k` chunks of the one allowed page have passed.
  - Stability (measured 2026-09-17, `results/stability_comparison.md`): the index results are
    deterministic within one server process (three repeated runs at ef 20 returned the same 20
    chunks in the same order for 75/75 claims, at a 512 MB cache and again at 16 MB), the cache
    size changes nothing (`SET GLOBAL mhnsw_max_cache_size` 512 MB -> 16 MB: 75/75 identical hit
    lists), and a server restart changes them: the run after the restart matched the runs before
    it for 46/75 claims, 13 claims lost their gold page from the top 20 and none gained one, and
    article recall@10 at ef 20 went from 73/75 to 61/75. The committed baseline is a third state
    of the same on-disk M=6 index (58/75 identical hit lists with the pre-restart runs, 48/75 with
    the post-restart runs), and every `ALTER TABLE` rebuild of an M=6 graph is another
    approximation (13/75 identical to the pre-restart state). The phase-2 suspicion of a thrashing
    16 MB cache was wrong; 512 MB stays in `docker-compose.yml` for capacity only. A higher
    ef_search narrows the effect (61/75 identical hit lists between two index states at ef 100
    against 48/75 at ef 20), and the M=16 index reaches the exact counts at ef 20 (Phase 3).
  - Worst cases at ef_search 20: seven claims whose gold page is not in the top 20, mostly
    claims about a person or event where the gold page is a city or institution (Kabul, Acadia
    University, Los Angeles), and a taxonomy cluster (Asteraceae / Asterales) where the corpus
    holds both pages.

## Phase 3 outcomes and chosen defaults (2026-09-17)

The experiment protocol is `scripts/run_experiments.py` (groups `stability`, `ef_search`,
`index_m`, `chunk_size`, `prefix`, `filters`, `restore`, `summary`; `all` runs them in that order
and `summary` rebuilds `results/SUMMARY.md` from the JSON files). Every number below is in
`results/SUMMARY.md` or in the per-run file it names, measured on the 100-page corpus (75 claims,
66 with a sentence-only evidence set) as ingested in phase 2 (8,868 chunks of at most 120 words,
prefix on, M=6) unless the bullet says otherwise. The chosen defaults are in the code
(`sql/schema.sql`, `wikilense/config.py`, `.env.example`) and apply to the next `wikilense
ingest`; until then the main database holds the phase-2 ingest.

- **Index `M` = 16** (`sql/schema.sql`; was 6). Rebuilt with `ALTER TABLE chunk DROP INDEX ...,
  ADD VECTOR INDEX ... M=16 DISTANCE=cosine`: at ef_search 20 the M=16 graph returns the exact
  ranking's counts, article recall@10 74/75 and evidence recall@20 63/66 (hit lists identical to
  the exact ranking for 64/75 claims), at SQL p50@10 0.71 ms; the M=6 graph gives 61/75 and 42/66
  at ef 20 and needs ef 200 to 400 for 74/75 and 62 to 63/66. Build 2.56 s (M=6 0.69 s, M=32
  8.72 s); graph tablespace 17.0 MB against 16.0 MB for M=6 and 20.0 MB for M=32, which adds
  nothing (74/75 and 63/66 at 0.82 ms). Section 3, `results/index_m_rebuild.md`,
  `results/m16_ef_20.md`.
- **`mhnsw_ef_search` = 100 as the application default** (`Settings.ef_search`,
  `WIKILENSE_EF_SEARCH`; the server default of 20 is untouched). At M=16 it keeps the exact counts
  and brings the hit lists closer to the exact ranking (71/75 identical claims against 64/75 at ef
  20) for p50@10 0.87 ms instead of 0.71 ms; on the M=6 index it is where the results stop
  depending strongly on the index state (61/75 identical hit lists between two states against
  48/75 at ef 20). Section 2, `results/ef_20.md` to `results/ef_400.md`, `results/m16_ef_100.md`.
- **`chunk_max_words` = 240, overlap 1** (`config.DEFAULT_CHUNK_MAX_WORDS`; was 120). At equal
  retrieved text (chunk size x k) the 240-word chunks are never below the others: at 1,200
  nominal words article recall 73/75 / 73/75 / 73/75 and evidence recall 49/66 / 46/66 / 49/66 for
  60 / 120 / 240 words (k = 20 / 10 / 5), at about 480 to 600 words 43/66 / 40/66 / 43/66; the
  differences are within the index-state noise (two states of the M=6 index at ef 100 differ by 1
  claim in article recall@10 and 3 in evidence recall@20). What 240 changes is the cost: 4,751
  vectors against 8,868 (19,467 at 60 words), a graph of 12.0 MB against 16.0 MB (27.0 MB), SQL
  p50 0.69 ms against 0.82 ms (0.88 ms) for the 1,200-word row, ingest 20.8 s against 23.5 s
  (29.2 s). Section 4, `results/chunk60_ef_100.md`, `results/ef_100.md`,
  `results/chunk240_ef_100.md`.
- **Embedding prefix kept** (`"title > section path: "`, `use_prefix=True`). The approximate runs
  of section 5 favoured no prefix (at ef 100 article recall@1 65/75 against 61/75), but they
  compare two different M=6 graphs. The exact ranking (the inline statement, which does not depend
  on the graph) settles it: with and without the prefix article recall@10 is 74/75; evidence
  recall@10 is 54/66 with the prefix and 55/66 without, @20 63/66 with and 59/66 without
  (`results/inline_ef_20.md`, `results/noprefix_inline_ef_20.md`). No gain from removing it, so the
  prefix stays.
- **Filtered queries: strategy `inline`** (the default of `search()`), `overfetch` as the
  low-latency option. Inline always fills `k`: under `heading LIKE '%History%'` (268 of 8,868
  chunks pass) 0 of 75 queries came back short at k=10, against 66 with overfetch 10 and 23 with
  overfetch 50, at p50 5.30 ms / p95 6.93 ms (overfetch 10: 1.40 ms, overfetch 50: 2.79 ms).
  Under the non-selective `min_words >= 1000` (8,787 chunks pass) all three give article
  recall@10 64/75 and overfetch 10 costs 1.20 ms against 4.44 ms, so it is the choice when latency
  matters more than a complete result. The inline cost is an exact ranking over the index and
  grows with the table, which is the number to watch when the corpus is scaled. Section 6,
  `results/strategy_*.md`.
- **`mhnsw_max_cache_size` = 512 MB kept** (`docker-compose.yml`) for capacity only: the cache
  size did not change one hit list (75/75 identical at 512 MB and 16 MB within one server
  process) and the graph of 8,868 chunks is already 16 MB on disk, which the 16 MB default would
  not hold for a larger corpus. Section 1, `results/stability_comparison.md`.
- **Reproducibility, for the README**: at M=6 and ef 20 the counts depend on the HNSW graph the
  server holds (three states of one on-disk index were observed, and each rebuild is another
  graph), so a reader reproduces them only to within a few claims. The chosen M=16 and ef 100 are
  where the counts equal the exact ranking's (article recall@10 74/75, evidence recall@20 63/66),
  which is what makes them reproducible; `wikilense eval` after `wikilense ingest` with the
  defaults is the check.

## Phase 5 (2026-09-22): the protocol rerun on the final code

After the review fixes (hatnote units out of chunks, `ANALYZE TABLE` after ingest, the `rrf`
strategy, the chosen defaults) the whole protocol was rerun with `scripts/run_experiments.py all`
on 2026-09-22; every file in `results/` comes from that run, and `results/SUMMARY.md` is the
reference for every number in the README. Section 7 there is the final configuration (240-word
chunks, overlap 1, M=16, prefix on, 4,598 chunks, 1,316 hatnote units excluded): article
recall@1/5/10/20 63/74/74/74 of 75, evidence recall@1/5/10/20 39/55/60/65 of 66, SQL p50/p95 at
k=10 0.90/1.41 ms, query embedding 4.7 ms; hit lists identical to the exact ranking for 74/75
claims, and identical across a server restart for 74/75. Every number of the phase-3 section
above was measured on the pre-hatnote ingest and is superseded by sections 1 to 7 of the summary,
which repeat the measurements on the current code; M=16 included: at ef 20 it now gives article
recall@10 72/75 and evidence recall@20 61/66 (section 3), not the exact ranking's 74/75 and 63/66,
which it reaches at ef 100.

## Review pass (2026-09-23)

A code review changed behaviour that later runs will see (commits after 92f1838; the committed
`results/` predate it). The metrics at `k` now come from the `LIMIT k` query, which changes the
`overfetch` and `rrf` result files with a non-zero `prefix_mismatches`:
`strategy_overfetch10_history`, `strategy_overfetch50_history`, `final_rrf` and `rrf_120_m6` need
the experiments rerun (a read-only recomputation of `final_rrf` on the current database changes
only evidence recall@1, 35 to 34 of 66). Every other file, the `none` and `inline` ones and the
two overfetch runs under `min_words`, had 0 prefix mismatches and does not change. The protocol
rerun of 2026-10-22 regenerated every file, these four included, on the current code.
The rest changes no number: `wikilense eval --ef-search 0` keeps the server value as documented,
`WIKILENSE_INDEX_M` is gone (M is the schema's; `ingest_meta` records the live index), the model is
pinned to revision 5c38ec7c405e, the ingest checks settings and model before dropping anything,
the rrf full-text list breaks ties by `chunk_id` (hit lists unchanged for 375 of 375 claim / k
pairs), and `scripts/run_experiments.py` builds no SQL from strings.

## Page metadata and the brief's four filters (2026-10-22)

The brief's example predicates are "only articles in this category, longer than this, edited since
that date, linked from that other article". Length (`page.n_words`) and links (`link`) came with the
FEVEROUS pages; categories and edit dates did not. The pages carry no category, revision or date
field, and they stop before the category box and the "last edited" footer. A random 100-page sample
of the full `feverous_wikiv1.db` and three pages read by hand confirmed this.

- **Source.** `scripts/fetch_page_meta.py` asks the MediaWiki Action API, for every corpus page, for
  the revision whose text the corpus holds.
  - FEVEROUS scraped Wikipedia in December 2020 and kept a per-page date stamp, but did not release
    it (paper, Sec 7.7). Taking today's "last edited" date or today's categories would pair 2026
    metadata with 2020 text.
  - Pages without an edit during December 2020 have one candidate, the revision live at
    2020-12-31T23:59:59Z (7 pages).
  - For pages edited during December (93), the candidate is chosen by the most FEVEROUS sentence
    probes (first six words, links reduced) found in its wikitext. When several candidates hold
    equally many probes, the later one is taken, which assumes nothing about the scrape day beyond
    "December"; the file does not record how often that happened. A revision that holds none of
    the probes would be recorded as `unverified` with null metadata (none of the 100 is).
  - Three titles were moved after 2020 and are followed through the move log. All 100 pages were
    found, and 94.5% of the probes occur in the chosen revisions.
  - Example: on today's "Albert Einstein" page only 260 of 484 probes are found, against 466 in
    the December 2020 revision.
  - The categories are the `[[Category:...]]` links of that revision: 1,005 distinct names, 1,123
    page-category pairs, 72 names shared by two or more pages. Categories added by templates are
    not in the wikitext; they are mostly hidden maintenance categories.
  - Wikimedia Robot Policy: one request at a time and at most four per second, a User-Agent with the
    repository URL, gzip, `maxlag`, `Retry-After`, and a pause after slow requests. The 100 pages
    took 309 requests. The output, `data/corpus/page_meta.jsonl` and `PAGE_META.md`, is committed.
    A re-run finds the same revisions unless a page is deleted.
- **Schema.**
  - `page.revision_id` and `page.last_edited` (UTC, indexed), NULL when the file has no record.
  - `category(category_id, name UNIQUE utf8mb4_bin)`.
  - `page_category(page_id, category_id)`, with an index on (`category_id`, `page_id`).
  - The ingest reads the file when it is present and records its SHA-256, or `none`, in
    `ingest_meta`.
- **Filters.** `Filters.category` adds `STRAIGHT_JOIN (SELECT DISTINCT page_category.page_id ...
  WHERE category.name = %s) AS in_category`. `edited_since` / `edited_before` add
  `page.last_edited >= %s` / `< %s`. Aware datetimes are converted to UTC; naive ones and dates
  are read as UTC. CLI: `--category`, `--edited-since`, `--edited-before`; the API and the form
  use the same names.
- **Plan rule found on the way.** Without the DISTINCT, MariaDB 11.8.9 merges the category derived
  table into the outer join and reads `chunk` in full (EXPLAIN type `ALL`), even on the 12-chunk
  test data. With it, the table is materialised (`<derived2>`) and the vector index drives the
  scan, like the link filters. `tests/test_search.py` pins this plan.
- **When the optimizer keeps the vector index under a filter** (`results/planner_choices.md`,
  group `planner`): 75 claims, inline statement, k 10, ef 100.
  - Nine of the eleven filters, and the unfiltered statement, plan `chunk` through the index for
    every claim. The rows read grow as the filter gets more selective: 10 rows unfiltered; 601 at
    the median under `max_words 2000`, where 89 of 4,598 chunks pass.
  - The two most selective range predicates on indexed `page` columns make it scan all 4,598
    chunks and sort instead: `max_words 500` (25 chunks pass) and `edited_before 2020-06-01` (4
    pass). That gives the exact result at 5.2 / 3.1 ms p50, against 2.1 to 3.2 ms for the index
    plans.
  - No query returned fewer than min(k, passing) hits: under `edited_before 2020-06-01` only 4
    chunks pass, and each query returns those 4. The four filters together (377 chunks pass) keep
    the index and read 184 rows at the median, in 3.2 ms.
- **Searching rows the transaction has not committed** (`results/uncommitted_recall.md`, group
  `uncommitted`). A scratch table with the chunk table's vector column and index is filled inside
  a transaction and searched before COMMIT; the transaction is rolled back, the table filled
  again, committed and searched again, with three fresh builds.
  - With the 4,598 FINAL chunk embeddings and the 75 claims as queries, the overlap with the exact
    top 10 is 0.972 / 0.973 / 0.973 before COMMIT at `mhnsw_ef_search` 20 / 100 / 400, and
    0.990 / 1.000 / 1.000 after. Two earlier runs gave 0.968-0.976 before and 0.987-1.000 after.
    At ef 100 the worst query finds 4 of 10 before COMMIT and 10 after.
  - On 1,000 random vectors it is 0.583 / 0.623 / 0.623 before and 0.787 / 0.992 / 0.996 after.
  - Since 2026-10-22 each state gets an untimed warm-up pass before the timed ones. With it, the
    latency differs little (p50 0.44 to 0.86 ms before COMMIT against 0.69 to 1.03 ms after, on
    the chunks). The first runs, without warm-up, showed a threefold gap: a cold-cache effect that
    a grader pointed out.
  - ROLLBACK leaves 0 rows in every build. Why the index finds fewer true neighbours before
    COMMIT is not known.
  - Practical rule: write and search in separate transactions, or commit before searching.
  - `tests/test_db.py` commits before its NumPy cross-check for this reason.
- **Scale with distractor pages** (`results/scale.md`, group `scale`, 2026-10-22).
  - The corpus, then the corpus plus 5,000 and 20,000 random pages of the full FEVEROUS Wikipedia
    (seeded, nested samples; `WIKILENSE_FEVEROUS_DB`), each ingested with the FINAL settings: 4,598
    / 26,700 / 89,650 chunks, HNSW graph 14 / 42 / 113 MB.
  - HNSW search p50 0.83 / 1.05 / 1.23 ms; exact joined statement 2.52 / 10.27 / 19.17 ms; inline
    under the History filter 3.30 / 11.54 / 21.30 ms (never short); overfetch 10 under it 1.03 /
    1.44 / 1.70 ms, short for 69 / 68 / 65 of 75 queries.
  - Article recall@10 74 / 74 / 71 of 75 for both HNSW and exact; HNSW hit lists equal the exact
    ones for 75 / 73 / 72 claims. The ingest of 20,100 pages took 328.5 s (240.1 s embedding).
  - So the README's "number to watch" is measured: the joined form's cost grows with the table
    (7.6 times for 19 times the chunks), the bare HNSW search's barely (1.5 times).
- **Vector arithmetic checked.** On 300 random, unnormalised vectors, `VEC_DISTANCE_COSINE`
  equals NumPy's `1 - cos` for every row (within 1e-5). A scan that ignores the index returns
  NumPy's exact top 10 in order, and the HNSW index at ef 200 finds all of them
  (`tests/test_db.py`).
