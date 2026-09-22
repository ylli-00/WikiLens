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
  config.py                Settings from environment / .env (DB connection, model, chunk and index parameters)
  corpus.py                Read data/corpus/*.jsonl; element-id parsing; corpus selection logic
  wikitext.py              FEVEROUS page parsing: page order, section tree, text units, links, markup cleaning
  chunking.py              Sentence-window chunker -> Chunk objects that remember their sentence keys
  embedding.py             Embedder (sentence-transformers) -> float32 numpy; vector <-> bytes/text helpers
  db.py                    PyMySQL connection, schema apply/reset, vector parameter helpers, bulk inserts
  ingest.py                Pipeline: corpus -> page/section/sentence/link/claim tables -> chunks -> embeddings
  search.py                Query functions: knn, hybrid (predicates + joins), join back to sentences
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
- Fill a chunk with consecutive units until adding the next unit would exceed `max_words`; a
  single unit longer than `max_words` becomes its own chunk. The next chunk starts
  `overlap_units` units before the end of the previous chunk, but only when the previous chunk
  has more than `overlap_units` units (no infinite loops, no duplicate chunks).
- `Chunk(ordinal, section_ordinal, element_keys, text, n_words)`; `text` is the cleaned units
  joined by a single space; `ordinal` counts chunks per page from 0.
- The text that is embedded is `f"{title} > {section_path}: {text}"` when the section path is
  non-empty, else `f"{title}: {text}"`. The stored `chunk.text` stays the plain text. Ingest adds
  the prefix; it was tested against no prefix and kept (exact rankings: article recall@10 74/75
  both ways, evidence recall@20 63/66 with the prefix against 59/66 without; see Phase 3).

## Embedding (`embedding.py`)

- Model: `BAAI/bge-small-en-v1.5`, 384 dimensions, MIT licence, 33M parameters (chosen in phase
  1 and kept; no other model was measured, so the choice rests on the reasons below).
  Reason: strong on passage retrieval for its size, fast enough on CPU for a grader, and it fits
  the 8 GB GPU with room to spare. Normalised embeddings, so cosine distance equals
  `1 - dot`.
- `Embedder(model_name=..., device=None)` loads lazily; `dim` property; `embed_passages(texts,
  batch_size=64) -> np.ndarray[float32, (n, dim)]`; `embed_queries(texts)` prepends the model's
  query instruction (`"Represent this sentence for searching relevant passages: "` for bge).
- Helpers, independent of the model: `vector_to_bytes(v) -> bytes` (float32 little-endian, the
  MariaDB storage format), `bytes_to_vector(b) -> np.ndarray`, `vector_to_text(v) -> "[...]"` for
  `VEC_FromText`.

## Database (`sql/schema.sql`, `db.py`, `config.py`)

All tables InnoDB, `utf8mb4`. Title columns use `utf8mb4_bin` so that matching is exact.

| Table | Columns (PK first) | Notes |
|---|---|---|
| `page` | `page_id`, `title` UNIQUE, `n_sentences`, `n_items`, `n_words`, `n_chars`, `n_sections`, `n_tables`, `n_lists` | index on `n_words` for length filters |
| `section` | `section_id`, `page_id` FK, `ordinal`, `heading`, `level`, `path` | UNIQUE (`page_id`, `ordinal`); index on `heading` |
| `sentence` | `sentence_id`, `page_id` FK, `section_id` FK, `element_key`, `ordinal`, `text` | UNIQUE (`page_id`, `element_key`); one row per text unit (sentence or list item) |
| `chunk` | `chunk_id`, `page_id` FK, `section_id` FK, `ordinal`, `text`, `n_words`, `embedding VECTOR(384) NOT NULL` | UNIQUE (`page_id`, `ordinal`); `VECTOR INDEX (embedding) M=16 DISTANCE=cosine` (M=6 in phases 1 and 2; 16 chosen with the M sweep, `results/SUMMARY.md` section 3) |
| `chunk_sentence` | (`chunk_id`, `sentence_id`) | the chunk-to-sentence map; index on `sentence_id` |
| `link` | `link_id`, `from_page_id` FK, `to_title`, `to_page_id` NULL FK, `source_element` | `to_page_id` resolved when the target is in the corpus; indexes on `from_page_id`, `to_page_id`, `to_title` |
| `claim` | `claim_id` (FEVEROUS id), `split`, `text`, `label`, `challenge` | |
| `claim_evidence` | (`claim_id`, `evidence_set`, `position`), `element_id`, `page_title`, `element_type`, `page_id` NULL, `sentence_id` NULL | resolved ids when the page / text unit is in the corpus; indexes on `page_id`, `sentence_id` |
| `ingest_meta` | `key`, `value` | embedding model, dim, chunk parameters, index M, corpus sha256, ingested_at |

Vector dimension 384 is written literally in the DDL and checked against `Embedder.dim` at
ingest; a mismatch is an error with a clear message.

`config.Settings` (from environment, `.env` loaded with python-dotenv): `WIKILENSE_DB_HOST`,
`WIKILENSE_DB_PORT`, `WIKILENSE_DB_USER`, `WIKILENSE_DB_PASSWORD`, `WIKILENSE_DB_NAME`,
`WIKILENSE_EMBEDDING_MODEL` (default above), `WIKILENSE_CHUNK_MAX_WORDS` (240; 120 until
2026-09-17), `WIKILENSE_CHUNK_OVERLAP_UNITS` (1), `WIKILENSE_VECTOR_DIM` (384), `WIKILENSE_INDEX_M`
(16; 6 until 2026-09-17) and `WIKILENSE_EF_SEARCH` (100, `Settings.ef_search`: the
`mhnsw_ef_search` value the application passes per query; the server default stays 20).

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
the server with 512 MB because the graph of the 8,868-chunk corpus is already 16 MB on disk (the
size does not change the results, see the stability bullet under Phase 2).

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

- `knn(conn, qvec, k)`: `SELECT chunk_id, page_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance
  FROM chunk ORDER BY distance LIMIT %s`.
- Hybrid variants, each returning the same row shape, each with its `EXPLAIN` recorded:
  predicate on page length (`page.n_words >= %s`), on section heading (`section.heading LIKE %s`
  or `path`), on link structure (`chunk.page_id IN (SELECT to_page_id FROM link WHERE
  from_page_id = ...)`), and combinations; joined back to `page`, `section` and, through
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
- **Database** (MariaDB 11.8.9, PyMySQL 2.2.8): binding a `VECTOR` parameter as plain bytes works
  only on a connection opened with `binary_prefix=True` (`db.connect` sets it); otherwise MariaDB
  rejects it (errors 1292 / 4079). `VEC_FromText(%s)` and `UNHEX(%s)` work on any connection. The
  knn query `ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT n` gives `EXPLAIN` type `index`, key
  `embedding`; the alias form `ORDER BY distance` is identical. Resolution references
  (`link.to_page_id`, `claim_evidence.page_id`, `claim_evidence.sentence_id`) use `ON DELETE SET
  NULL`, ownership references cascade. Explicit indexes on every FK column. `claim.split`,
  `claim.label`, `claim_evidence.element_type` are ENUMs over the complete value sets found in the
  files. `section.heading` uses the server default collation (`utf8mb4_uca1400_ai_ci`), so a
  heading filter with `LIKE` is case- and accent-insensitive on purpose. `mhnsw_ef_search` outside
  its range is clamped with a warning, not an error. Schema apply 59 ms, reset 87 ms.
- **Embedding** (bge-small-en-v1.5, revision 5c38ec7c405e, 33.4M parameters, 256 MB on disk):
  RTX 4060 1,059 passages/s, CPU 78 passages/s for 100-word texts; peak GPU memory 253 MiB; the
  outputs are bitwise deterministic on both devices; model load 3 to 6 s from a warm cache.

## Ingest (`ingest.py`), contract for phase 2

`run_ingest(settings, corpus_dir=DEFAULT_CORPUS_DIR, reset=True, embedder=None, batch_size=64,
use_prefix=True, progress=True) -> IngestReport` with counts `n_pages, n_sections, n_sentences,
n_units_empty, n_chunks, n_links, n_links_resolved, n_links_skipped, n_claims, n_evidence,
n_evidence_page_resolved, n_evidence_sentence_resolved` and `seconds` per stage (parse, embed,
load, resolve). Steps, each committed when it completes:

1. `apply_schema(reset)`; refuse to run against the test database.
2. Parse every page (`parse_page`, `chunk_page`); insert `page`, `section` (lead included),
   `sentence` (every unit, empty ones too), `link` (`to_title` over 255 characters skipped and
   counted). Keep chunks in memory with their `embedding_text` (prefix on by default).
3. Embed all chunk texts in batches (`embed_passages`), then insert `chunk` rows with
   `vec_param` bytes and `chunk_sentence` rows through `insert_rows`. Assert `embedder.dim ==
   settings.vector_dim` before embedding.
4. Resolve `link.to_page_id` with one `UPDATE link JOIN page ON page.title = link.to_title`.
5. Insert `claim` and `claim_evidence` (`parse_element_id`; `page_id` by title, `sentence_id` by
   (`page_id`, `element_key`); cells and captions keep `sentence_id` NULL).
6. Write `ingest_meta`: `embedding_model`, `embedding_dim`, `embedding_prefix`, `chunk_max_words`,
   `chunk_overlap_units`, `index_m`, `index_distance`, `corpus_dir`, `corpus_pages_sha256`,
   `corpus_claims_sha256`, `mariadb_version`, `wikilense_version`, `ingested_at` (UTC ISO 8601).

## Search (`search.py`), contract for phase 2

- `Hit(chunk_id, page_id, title, section_path, chunk_ordinal, n_words, distance, text)`.
- `Filters(min_words=None, max_words=None, heading_like=None, path_like=None, linked_from=None,
  links_to=None, titles=None)`; every field optional; `heading_like` and `path_like` are SQL LIKE
  patterns supplied by the caller (parameters, never interpolated).
- `search(conn, qvec, k=10, filters=None, strategy="inline", overfetch=10, ef_search=None) ->
  list[Hit]` with strategies: `inline` (predicates and joins in the one statement that carries
  `ORDER BY VEC_DISTANCE_COSINE(...) LIMIT k`), `overfetch` (an inner index-driven query with
  `LIMIT k * overfetch`, filtered and re-limited in the outer query), `none` (no filters).
  `ef_search` sets the session variable for that call and restores it afterwards.
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
- `evaluate(conn, embedder, ks=(1, 3, 5, 10, 20), repeats=5, strategy=..., filters=None,
  ef_search=None) -> EvalResult` with, per k: article recall (a gold page among the pages of the
  top-k chunks), evidence recall (every unit of at least one sentence-only set covered by the
  top-k chunks, over the 65 eligible claims), unit coverage (share of gold units covered), and
  latency: embedding time and SQL time separately, p50 / p95 / mean over `repeats` warm runs of
  every claim, plus the parameters, `ingest_meta`, machine and versions.
- `write_results(result, out_dir="results", name=...)` writes `<name>.json` and `<name>.md` (a
  table per metric).
- Sweeps are separate CLI runs (`--ef-search`, `--strategy`, `--k`), not hidden loops.

## CLI and web (`cli.py`, `web.py`), contract for phase 2

`wikilense init-db [--reset]`, `wikilense ingest [--corpus-dir] [--no-reset] [--batch-size]
[--no-prefix]`, `wikilense query "text" [--k 5] [--min-words N] [--heading PATTERN]
[--linked-from TITLE] [--strategy inline|overfetch|none] [--ef-search N] [--explain] [--json]`,
`wikilense eval [--k 1,3,5,10,20] [--repeats 5] [--strategy ...] [--ef-search N] [--out results]
[--name NAME]`, `wikilense serve [--host 127.0.0.1] [--port 8000]`. The web page is one HTML form
(query, k, filters) that calls `GET /api/search` and shows hits with title, section path,
distance, the chunk text and the SQL that ran; no JavaScript framework, no external assets.

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
claims, and identical across a server restart for 74/75. The M=6 numbers of the phase-3 section
above were measured on the pre-hatnote ingest and are superseded by sections 1 to 3 of the
summary, which repeat them on the current code.
