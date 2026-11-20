# WikiLense design

WikiLense does retrieval over Wikipedia inside MariaDB 11.8. It chunks pages, embeds the chunks,
stores them in a `VECTOR` column with a cosine `VECTOR INDEX`, queries them with
`VEC_DISTANCE_COSINE`, and combines the semantic search with ordinary SQL predicates and joins in
the same statement. FEVEROUS claims are the queries, and their gold evidence is the ground truth
for recall.

This file is the contract between modules and the record of each design decision, with its reason
and its evidence. The settings were chosen on the 75-claim development set and the 100-page corpus
("Chosen defaults and their evidence"), and the scaled run keeps them fixed. The numbers are
those of the current result files: `results/SUMMARY.md` (rebuilt from the JSON files by
`scripts/run_experiments.py summary`) for the demo corpus and `results/scaled/` for the scaled
run, which the README and `docs/EVALUATION.md` quote.

## Repository layout

Every file and what it does is listed in [ARCHITECTURE.md](ARCHITECTURE.md) ("File by file");
`sql/schema.sql` is the single source of truth for the DDL, and `db.apply_schema()` executes it.
The scaled run adds `scripts/scaled_*.py` and `results/scaled/`. The LLM fact-checker extension
is in `wikilense/llm/` and is documented separately.

Rules for all code: Python 3.10+, the standard library where it is enough, type hints, docstrings
that say what a function returns, no SQL built by string formatting (parameters only, and
identifiers only from fixed literals in the code), no credentials in code, and no debug flags left
on. Tests live in `tests/` and are real tests.

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
`table_caption`, `item`. A title can contain an underscore, so parse with a regex anchored at the
end: `^(?P<page>.+?)_(?P<type>sentence|cell|header_cell|table_caption|item)_(?P<nums>\d+(?:_\d+)*)$`.
The page group is non-greedy: with a greedy one, every `header_cell` id parses as page
`X_header`, type `cell` (measured on all 383,137 train+dev ids). `corpus.parse_element_id(el)`
returns `(page_title_nfc, type, key)`, where `key` is the page-object key (`sentence_7`,
`item_0_1`, `cell_0_1_1` ...). Normalise page titles to Unicode NFC before any comparison. Apply
NFC to parsed strings, never to raw JSON text: NFC can compose the letter of an escape (`\n`) with
a combining mark that follows, which breaks the JSON.

**Text unit**: one sentence (`sentence_N`) or one list item (`item_N_M`), in page order. Tables
and captions are not text units; they are counted in the page statistics only. This is a
documented limitation: 27 of the 114 evidence ids of the 75 development claims are table cells.

## Parsing rules (`wikitext.py`)

- Walk `order`. Keep a section stack keyed by `level`; a new section pops every level >= its own.
  Section ordinal 0 is the lead (heading `""`, level 1, path `""`). Every later `section_N` opens
  a new section whose `path` is the headings from the top of the stack, joined by `" > "`.
- A text unit is emitted for every `sentence_N` and every item of every `list_N`, numbered in page
  order from 0.
- `clean_text(raw)`: `[[target|shown]]` -> `shown`; `[[target]]` -> `target` with `_` -> space;
  remove any remaining `[[`/`]]`; collapse whitespace; strip. On the whole shard
  `wiki_000.jsonl` (654,115 text units), 86 units (0.013%) still contain markup characters after
  cleaning, and in all of them the characters are real text (`|` in scores, formulas, IPA).
- Links are read from sentences, list items and table cells. The target is the text before `|`,
  with a `#fragment` dropped, `_` -> space and NFC applied; links stay in page order, duplicates
  included. Links with an empty target (`[[#Section|x]]`, `[[]]`) and bare links truncated by the
  sentence splitter are skipped. Table captions are not read, because they have no element id.
- Page statistics count sentences, items, words (over cleaned text units), characters, sections,
  tables and lists. Sections are counted by heading, so `section` holds one row more per page: the
  lead.

## Chunking rules (`chunking.py`)

- `chunk_units(units, max_words, overlap_units=1)` is a pure function over one page's text
  units. Ingest passes `Settings.chunk_max_words` (240: at equal retrieved text the 60-, 120- and
  240-word chunkings are close, and 240 needs about half the vectors; `results/SUMMARY.md` section 4) and
  `Settings.chunk_overlap_units` (1, kept without a sweep). A chunk never crosses a section
  boundary.
- Only *chunkable* units take part (`TextUnit.chunkable`). A unit that is empty after cleaning, or
  a hatnote (`wikitext.HATNOTE_RE`: "Main article: X", "See also: Y", "For other uses, see Z",
  ...), is in no chunk and in no `chunk_sentence` row, but stays a `sentence` row so evidence ids
  resolve (1,316 of the 33,637 units of the corpus are hatnotes).
- Fill a chunk with consecutive units until adding the next unit would exceed `max_words`; a
  single unit longer than `max_words` becomes its own chunk. The next chunk starts
  `overlap_units` units before the end of the previous chunk, but only when the previous chunk
  has more than `overlap_units` units, which prevents infinite loops and duplicate chunks. A chunk
  that would only repeat units of the previous chunk, or has zero words, is not emitted. A chunk
  keeps its element keys; its text is the cleaned units joined by a single space.
- The embedded text is `f"{title} > {section_path}: {text}"` when the section path is non-empty,
  else `f"{title}: {text}"`; the stored `chunk.text` stays the plain text. The prefix was
  tested against no prefix and kept (results/SUMMARY.md section 5, "Chosen defaults and their
  evidence").

## Embedding (`embedding.py`)

- Model: `BAAI/bge-small-en-v1.5`, 384 dimensions, MIT licence, 33M parameters. No other model was
  measured, so the choice rests on these reasons: it is strong on passage retrieval for its size,
  fast enough on a CPU-only machine, and fits the 8 GB GPU with room to spare. Measured on
  100-word texts, it embeds 1,059 passages/s on the RTX 4060 and 78 passages/s on the CPU, with a
  peak GPU memory of 253 MiB. The outputs are bitwise deterministic on both devices. The
  embeddings are normalised, so cosine distance equals `1 - dot`.
- The default model is pinned to the Hub commit `config.DEFAULT_EMBEDDING_REVISION`
  (5c38ec7c405e..., the snapshot every result was measured with), so a later update of the model
  repository cannot change the vectors. `embed_queries` prepends the model's query instruction
  (`"Represent this sentence for searching relevant passages: "` for bge); passages are embedded
  without it. The `Embedder` API and the text helpers for `VEC_FromText` / `VEC_ToText` are in
  `wikilense/embedding.py`. The bytes bound for a `VECTOR` parameter (float32 little-endian, the
  MariaDB storage format) come from `db.vec_param(v)`.

## Database (`sql/schema.sql`, `db.py`, `config.py`)

All tables are InnoDB with `utf8mb4`. Title columns use `utf8mb4_bin` so that matching is exact.

| Table | Columns (PK first) | Notes |
|---|---|---|
| `page` | `page_id`, `title` UNIQUE, `n_sentences`, `n_items`, `n_words`, `n_chars`, `n_sections`, `n_tables`, `n_lists`, `revision_id` NULL, `last_edited` NULL | index on `n_words` for length filters; `revision_id` / `last_edited` (UTC) name the December 2020 Wikipedia revision the text comes from (`data/corpus/page_meta.jsonl`), index on `last_edited` for the edit-date filters |
| `section` | `section_id`, `page_id` FK, `ordinal`, `heading`, `level`, `path` | UNIQUE (`page_id`, `ordinal`); index on `heading` |
| `sentence` | `sentence_id`, `page_id` FK, `section_id` FK, `element_key`, `ordinal`, `text` | UNIQUE (`page_id`, `element_key`); one row per text unit (sentence or list item) |
| `chunk` | `chunk_id`, `page_id` FK, `section_id` FK, `ordinal`, `text`, `n_words`, `embedding VECTOR(384) NOT NULL` | UNIQUE (`page_id`, `ordinal`); `VECTOR INDEX (embedding) M=16 DISTANCE=cosine` (16 chosen with the M sweep, `results/SUMMARY.md` section 3); `FULLTEXT KEY ft_chunk_text (text)` for the `rrf` strategy |
| `chunk_sentence` | (`chunk_id`, `sentence_id`) | the chunk-to-sentence map; index on `sentence_id` |
| `link` | `link_id`, `from_page_id` FK, `to_title`, `to_page_id` NULL FK, `source_element` | `to_page_id` resolved when the target is in the corpus; indexes on `from_page_id`, `to_page_id`, `to_title` |
| `category` | `category_id`, `name` UNIQUE (`utf8mb4_bin`) | Wikipedia category names, no "Category:" prefix |
| `page_category` | (`page_id`, `category_id`) both FK | the many-to-many map; index (`category_id`, `page_id`) for the category filter |
| `claim` | `claim_id` (FEVEROUS id), `split`, `text`, `label`, `challenge` | |
| `claim_evidence` | (`claim_id`, `evidence_set`, `position`), `element_id`, `page_title`, `element_type`, `page_id` NULL, `sentence_id` NULL | resolved ids when the page / text unit is in the corpus; indexes on `page_id`, `sentence_id` |
| `ingest_meta` | `key`, `value` | embedding model, dim, chunk parameters, index M, corpus sha256 (pages, claims, page_meta or `none`), ingested_at |

Vector dimension 384 is written literally in the DDL and checked against `Embedder.dim` at
ingest; a mismatch is an error with a clear message. Resolution references (`link.to_page_id`,
`claim_evidence.page_id`, `claim_evidence.sentence_id`) use `ON DELETE SET NULL`, ownership
references cascade, and every FK column has an explicit index. `claim.split`, `claim.label` and
`claim_evidence.element_type` are ENUMs over the complete value sets found in the files.
`section.heading` uses the server default collation (`utf8mb4_uca1400_ai_ci`), so a heading filter
with `LIKE` is deliberately case- and accent-insensitive.

Settings come from `WIKILENSE_*` variables (the full list is in `.env.example` and
`wikilense/config.py`). The ones that carry a decision are `WIKILENSE_CHUNK_MAX_WORDS` (240; see
`results/SUMMARY.md` section 4), `WIKILENSE_CHUNK_OVERLAP_UNITS` (1), `WIKILENSE_EF_SEARCH` (100,
passed as `mhnsw_ef_search` per query, 1 to 10000; 0 keeps the server's session value, default 20;
anything else is a `SettingsError`), and `WIKILENSE_TEST_DB_NAME` (the test suite resets that
database, so `tests/conftest.py` refuses a name that equals the main database or does not end in
`_test`). The index `M` (16; see `results/SUMMARY.md` section 3) is written literally in
`sql/schema.sql` (`db.VECTOR_INDEX_M`). It is not a setting, and `ingest_meta` records the `M` of
the index that is in place.

Connections have autocommit off, and queries are parameterised (`%s`) only. The binding rule for
`db.vec_param(v)` is pinned by a round-trip test. Plain bytes work with PyMySQL 1.2.3 and later,
which escape bytes as `_binary X'...'` by themselves, and with 1.2.0 to 1.2.2 only on a connection
opened with `binary_prefix=True` (`db.connect` sets it); otherwise MariaDB rejects them (errors
1292 / 4079). `VEC_FromText(%s)` and `UNHEX(%s)` work on any connection.

The MariaDB docs give the rule for index use: the index drives a query only for
`ORDER BY VEC_DISTANCE_COSINE(embedding, <constant>) LIMIT n` sorted ascending. Wrapping the
distance in an expression, or a bare `WHERE VEC_DISTANCE(...) < t`, defeats it. The application
refuses an `mhnsw_ef_search` outside the server's range, which the server would clamp with only a
warning. `mhnsw_max_cache_size` (16 MB default, global) bounds the index cache.
`docker-compose.yml` starts the server with 512 MB, because the graph must fit in it (`cache`
under "How MariaDB 11.8.9 uses the vector index").

## Corpus selection (`corpus.py`, `scripts/build_corpus.py`)

This section gives the rule for the 100-page corpus and its 75 claims, the development set. The
scaled run selects its claims and pages differently ("The scaled run: 3,000 claims on two larger
corpora").

Inputs: one or more shards, and the train and dev claim files. Rule:

1. A claim is *in-shard* when every page cited in every `content` id of every evidence set is a
   shard title (NFC). On `wiki_000.jsonl` there are 75 such claims (65 train, 10 dev; 7 SUPPORTS,
   58 REFUTES, 10 NEI), all citing a single page; 66 have an evidence set made only of text units.
2. The evidence pages are the union of those pages: 54.
3. Filler pages: link targets of the evidence pages that are shard titles and not evidence pages,
   ranked by the number of distinct evidence pages linking to them (descending), then title
   (ascending); take up to 46. If fewer exist, continue in shard order with pages of at least 20
   sentences. Reason: linked pages are realistic distractors and make the "linked from" SQL filter
   meaningful. All 46 came from links (202 candidates).
4. Write `pages.jsonl` (raw shard lines, shard order; 100 pages, 8.0 MB), `claims.jsonl` (raw
   records plus `split`), and `MANIFEST.md` (rule, counts, sizes, SHA-256 of both files, the page
   titles). The build is deterministic, uses the standard library only, recomputes every count
   from the files, and is byte-identical on a re-run.

## Ingest (`ingest.py`)

`run_ingest` is the single entry point. Before it touches the database it refuses the test
database, a vector dimension other than the schema's literal, `chunk_max_words < 1`,
`chunk_overlap_units < 0`, an embedding model of another dimension and a malformed
`page_meta.jsonl`, so a mistake in `.env` or in that file never drops the ingest in place
(`pages.jsonl` and `claims.jsonl` are read while the steps run). Then it runs seven steps and
commits each one when it completes; a failure rolls back the uncommitted rows and closes the
connection:

1. `apply_schema(reset)`; `reset=False` is for a schema made by `init-db` and refuses a database
   that already holds pages.
2. Parse every page (`parse_page`, `chunk_page`) and insert `page`, `section` (lead included),
   `sentence` (every unit, empty ones and hatnotes too), `link` (a `to_title` over 255 characters
   is skipped and counted), and the metadata from `page_meta.jsonl` when the corpus directory has
   it (NULL and empty, with a warning, otherwise).
3. Embed all chunk texts in batches (`embed_passages`), then insert `chunk` rows with
   `vec_param` bytes and `chunk_sentence` rows through `insert_rows`.
4. Resolve `link.to_page_id` with one `UPDATE link JOIN page ON page.title = link.to_title`.
5. `ANALYZE TABLE` for `chunk`, `page`, `section`, `link`, `category` and `page_category`, so the
   optimizer plans the search statements from fresh statistics. With stale statistics right after
   a bulk insert, it started from `page` and lost the vector index.
6. Insert `claim` and `claim_evidence` (`parse_element_id`; `page_id` by title, `sentence_id` by
   (`page_id`, `element_key`); cells and captions keep `sentence_id` NULL).
7. Write `ingest_meta`: the model, its revision and dimension, the prefix and chunk settings,
   `index_m` and `index_distance` (read from `SHOW CREATE TABLE chunk`, so they describe the index
   that is in place), the hatnote pattern and count, the `ANALYZE` tables and seconds, the corpus
   directory and the SHA-256 of `pages.jsonl`, `claims.jsonl` and `page_meta.jsonl` (`none`
   without it), the MariaDB and WikiLense versions, and `ingested_at` (UTC ISO 8601).

## Search (`search.py`)

The API (`Hit`, `Filters`, `search()`, `explain_search()`, `hit_sentences()`) is in
`wikilense/search.py`. These are the contracts a decision rests on, each pinned by
`tests/test_search.py`:

- The k-nearest-neighbour core, `search.KNN_SQL`: `SELECT chunk_id, VEC_DISTANCE_COSINE(embedding,
  %s) AS distance FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s`.
- `search(conn, qvec, k=10, filters=None, strategy="inline", ...)`. Strategies: `inline`
  (predicates and joins in the one statement that carries `ORDER BY VEC_DISTANCE_COSINE(...) LIMIT
  k`; the default, see "Chosen defaults and their evidence"), `overfetch` (an inner index-driven
  query with `LIMIT k * overfetch`, filtered in the outer query; `overfetch` at most 1000), `none`
  (no filters), `rrf` (the vector top-N and the FULLTEXT top-N of `query_text`, N = `k *
  overfetch`, fused by reciprocal rank fusion, `1 / (60 + rank)` summed over both lists; the
  filters apply to the fused list).
- `Filters` fields: `min_words`, `max_words`, `heading_like`, `path_like` (SQL LIKE patterns, passed
  as parameters), `linked_from`, `links_to`, `titles`, and the metadata filters `category`,
  `edited_since`, `edited_before` ("Page metadata and the four filters").
- Ties are broken by `chunk_id`: in Python for the vector strategies, because `ORDER BY distance,
  chunk_id` loses the index; in SQL for `rrf`. A query vector of the wrong dimension is a
  `ValueError`: MariaDB would return NULL distances, not an error.
- `ef_search` (1 to 10000; None leaves the session alone) is set for that call and restored
  afterwards, even when the statement fails. An error from the restoring `SET` never replaces the
  block's own error, because after a lost connection or a Ctrl-C it would hide what happened. A
  missing value takes `Settings.ef_search`, and 0 means the server's session value, in the CLI,
  the web page and the harness alike. Every SQL statement is a named module constant, so the
  README can quote it.

Statement shapes and their reasons (`docs/VECTOR_INDEX.md`, "The statements"):

- **The inline statement uses `STRAIGHT_JOIN` with `chunk` first.** With a plain `JOIN` on stale
  statistics the optimizer started from `page` and lost the vector index. Left free, the optimizer
  starts from the filter's own table under 17 of 18 filters on S1 and sorts the chunks that pass:
  0.86 to 3.13 ms instead of 129.95 to 1,387.73 ms when 662 or fewer chunks pass, but 1,566.76 and
  1,689.04 ms instead of 26.68 and 26.40 ms when half of them pass
  (`results/scaled/run1/s1/join_order.md`, 200 claims; on the demo corpus `results/join_order.md`).
  The fixed order keeps the cost of the joined index search under a loose filter and is slow under
  a very selective one. A rule for when to leave the optimizer free would need a count of the passing
  chunks before the search; we did not build or measure one, because the settings are fixed.
- **The link and category filters join a `SELECT DISTINCT` derived table.** The `IN (subquery)`
  form becomes a semi-join that starts from `link` and loses the vector index. The `DISTINCT` keeps
  a chunk from coming back once per link row: without it the derived table is merged and the index
  is still walked for 200 of 200 claims, but on S1 up to 147 of 200 lists name a chunk twice
  (`results/scaled/run1/s1/join_order.md`). `tests/test_search.py` pins both plans.

## How MariaDB 11.8.9 uses the vector index

This is the main finding of the project. `docs/VECTOR_INDEX.md` has the statements and every
table.

- **The bare statement is the HNSW search.** `ORDER BY VEC_DISTANCE_COSINE(...) LIMIT k`
  (strategies `none` and `overfetch`) reads exactly `LIMIT` rows: SQL p50 0.40 ms at k=1 on the
  120-word demo ingest at ef_search 20 (`results/ef_20.md`), 1.11 ms (S0) and 1.43 ms (S1) at k=5
  and ef_search 100 (`results/scaled/eval/`). `mhnsw_ef_search` sets its accuracy: at ef 20 that
  M=6 graph found 61/66/67/68/68 of 75 gold pages at k=1/3/5/10/20 against 66/72/74/74/74 for the
  exact ranking of the same vectors (`results/inline_ef_20.md`), so the gap comes from HNSW and
  not from the embedding.
- **A join makes the ranking exact, while the graph fits in `mhnsw_max_cache_size`.** With another
  table in the statement (strategy `inline`), `EXPLAIN` still reports `type index, key embedding`,
  but the server ranks exactly over the index, at a cost that grows with the table: 2.24 ms on the
  4,598 chunks of the demo corpus (`results/final_inline.md`), 15.74 ms on S0 and 25.81 ms on S1
  (`results/scaled/timing/`). The walk then continues until `k` rows have passed the joins and
  predicates (`ANALYZE` reports `r_rows` = `k` for `chunk` unfiltered, tests/test_search.py). The
  joined hit list equals NumPy's exact ranking, in order, for 75 of 75 claims on the demo corpus
  (`results/separate_index.md`) and for 3,000 of 3,000 on S0 and S1 (`results/scaled/run1/`).
  Under a filter it is close but not always exact. On the demo corpus the statement returned the
  scan's chunks with two near-equal neighbours swapped in 2 of 75 lists under the four filters
  (`results/join_order.md`), so `search()` sorts the rows it gets by the reported distance and the
  chunk id. On S1, after that sort, the list is the exact one for 2,996 of 3,000 claims under the
  History heading and 2,962 under the four filters (`results/scaled/run1/s1/separate_index.md`).
  With a cache that does not hold the graph the lists are no longer exact (2,725 to 2,734 of 3,000
  on S1 at 160 MB, `results/scaled/run1/s1/cache_size.md`).

`scripts/measure_index_behaviour.py` has five parts:

- **`forms` (`results/statement_forms.md`).** A predicate on `chunk` itself (`n_words`, no index)
  never comes back short, but it is approximate: it gave the exact filtered top 10 for only 51 of
  75 claims when 304 chunks pass, against 75 of 75 with a join. So copying a filter column onto
  the vector table without an index of its own would not give an exact filtered search.
- **`join_order` (`results/scaled/run1/s{0,1}/join_order.md`)** measures the decision under
  "Search", on a fixed sample of 200 claims, because a 20-claim pilot took 2,426 s and all 3,000
  would take about 100 hours (`results/scaled/pilot/`). Each form is timed alone, because on the
  demo corpus taking turns with a form that reads thousands of rows slowed the other one (2.11
  against 2.61 ms, `results/join_order.md`).
- **`isolation` (`results/isolation.md`).** A REPEATABLE READ transaction searches, another
  connection commits, and the first transaction searches again in its old snapshot and ends.
  After that, index searches from new connections at any isolation level missed the committed
  rows in 10 of 10 trials, until the next committed write or `FLUSH TABLES` (three runs; the five
  controls were clean). The stale index is MariaDB issue
  [MDEV-41321](https://jira.mariadb.org/browse/MDEV-41321). It was measured on one patch release
  in one container, and `results/isolation.md` has the statements to repeat it. WikiLense's
  connections keep autocommit off and REPEATABLE READ: `wikilense
  eval` runs many searches in one transaction, but only an ingest commits writes to the corpus,
  nothing in WikiLense searches while an ingest runs, and no measured number depends on it. An
  application that writes while it searches must not search the index inside a REPEATABLE READ
  transaction whose snapshot predates another connection's commit.
- **`builds` (`results/scaled/run1/s1/build_variance.md`).** On S1, twenty fresh builds at `M=16`
  and `ef_search` 100 found a gold page for 2,922 to 2,937 of 3,000 claims (the exact ranking
  2,943, the graph in the database 2,933), and twenty builds with the server defaults found 2,218 to
  2,433. Which build a reader gets moves the headline recall inside the first range, and far more
  at the defaults. On the demo corpus all twenty builds at `M=16` found 74 of 75, the exact
  ranking's number (`results/build_variance.md`).
- **`cache` (`results/scaled/run1/s1/cache_size.md`).** On S1, whose graph file is 156.0 MB, every
  count at 192 and 256 MB equals the one at 512 MB. At 160 MB, just above the file, the joined
  statement is exact for 2,725 to 2,734 of 3,000 claims and takes 351.45 ms instead of 27.53; at
  the 16 MB default the bare search finds a gold page for 2,658 claims instead of 2,933. On the
  demo corpus, 8 MB or less for its 13.0 MB graph left 57 to 67 of 75 lists exact
  (`results/cache_size.md`). The 16 MB default holds neither scaled graph (56.0 and 156.0 MB),
  which is why `docker-compose.yml` sets 512 MB, more than three times S1's graph.

`tests/test_search.py` pins five of the eight rows of the table "Checked against the
documentation" in `docs/VECTOR_INDEX.md`. The experiment protocol measures the following:

- **Uncommitted rows** (`results/uncommitted_recall.md`). A table searched inside the transaction
  that filled it finds fewer true neighbours than after COMMIT (overlap with the exact top 10:
  0.973 before, 1.000 after, at ef 100). The cause is not known. Each state gets an untimed
  warm-up pass; without it, the first runs showed a threefold gap, which was a cold-cache effect.
  The practical rule is to write and search in separate transactions, or to commit before
  searching, as `tests/test_db.py` does.
- **Scale with random distractor pages** (`results/scale.md`). From 4,598 to 89,650 chunks the
  exact joined statement grows 7.6 times (2.52 to 19.17 ms p50) and the bare HNSW search 1.5
  times (0.83 to 1.23 ms). Article recall@10 is 74 / 74 / 71 of 75 at the three sizes.
- **Vector arithmetic.** `VEC_DISTANCE_COSINE` equals NumPy's `1 - cos` within 1e-5 on 300 random
  vectors; `tests/test_db.py` requires a mean HNSW overlap of at least 95% with the exact top 10,
  because every graph build is another approximation.

## Evaluation (`evaluate.py`)

The API is in `wikilense/evaluate.py`; the contracts below are pinned by `tests/test_evaluate.py`.

- The queries are the corpus claims. The ground truth per claim is its gold pages and its
  sentence-only sets, from resolved ids only (a set is sentence-only when every element is a
  `sentence` or `item` with a resolved `sentence_id`). The default is `ks=(1, 3, 5, 10, 20)`;
  per-claim filters give the oracle (the gold page as a `titles` filter).
- For each k the harness reports article recall (a gold page among the pages of the top-k
  chunks), evidence recall (every unit of at least one sentence-only set covered by the top-k
  chunks, over the 66 eligible claims, with list items counted as text units), unit coverage
  (share of gold units covered), and latency (embedding and SQL time separately, p50 / p95 / mean
  over warm repeated runs).
- The metrics at `k` come from the `LIMIT k` search of each claim, the statement whose latency is
  reported at `k`. The first `k` hits of a `LIMIT max(ks)` search would be the same list for `none`
  and `inline` but not for `overfetch` and `rrf`, whose inner candidate list is `k * overfetch`
  long; `prefix_mismatches` counts the claims where the two differ.
- The pages and sentences a heading filter can reach are read through `chunk`
  (`PAGES_HEADING_SQL`, `SENTENCES_HEADING_SQL`; a database test pins it), because the filter keeps
  chunks and a section whose text is all in its subsections has no chunk. Under
  `heading LIKE '%History%'` 26 claims are reachable: the ceiling is 26/75, and `inline` reaches 25.
- Each run writes `<name>.json` and `<name>.md` with its parameters, `ingest_meta`, the machine
  and the versions. Sweeps are separate CLI runs, not hidden loops.

## Experiment protocol (`scripts/run_experiments.py`)

The groups are `stability`, `ef_search`, `index_m`, `chunk_size`, `prefix`, `filters`, `final`,
`restore`, `uncommitted`, `planner`, `scale` and `summary`; `all` runs them in that order without
`scale`. `summary` rebuilds `results/SUMMARY.md` from the JSON files and loads no settings, so it
needs no database password. The check that the database holds the default ingest also compares
the digests of `pages.jsonl` and `claims.jsonl`, so pages left by an interrupted `scale` group do
not pass as the default ingest. Sizes are binary MB, the unit MariaDB's documentation uses for
`mhnsw_max_cache_size`. Timings on this laptop drift (the p50 of one statement moved from 0.54 to
1.03 ms within a minute), so the methods compared take turns query by query. A NumPy scan runs on
one thread, because with its thread pool the same scan took 0.03 ms while the threads were
spinning and up to 2 ms after a pause.

## Chosen defaults and their evidence

The defaults are in the code (`sql/schema.sql`, `wikilense/config.py`, `.env.example`). They were
chosen on the 100-page development set and kept for the scaled run. The sweeps are sections 1 to
7 of `results/SUMMARY.md`, and [EVALUATION.md](EVALUATION.md#parameters-and-why) has the table.

- **Index `M` = 16.** On the same vectors at ef_search 20, article recall@10 is 67, 74 and 73 of 75
  for M 6, 16 and 32, and evidence recall@20 54, 64 and 62 of 66, against 74 and 63 for the exact
  ranking. At ef_search 100 the M=16 graph returns the exact hit list for 69 of 75 claims (M=32:
  73). The build takes 2.9 s (M=6 1.1 s, M=32 9.1 s) and the graph is 17 MB. Section 3,
  `results/index_m_rebuild.md`.
- **`mhnsw_ef_search` = 100 as the application default** (the server default of 20 is left as it
  is). On the final ingest the hit lists equal the exact ranking for 74 of 75 claims, against 65
  at 20, for 0.51 ms instead of 0.45 ms at k=10. Section 7, `results/final_ef_100.md`,
  `results/final_ef_20.md`.
- **`chunk_max_words` = 240, overlap 1.** At equal retrieved text (chunk size x k) the 60-, 120-
  and 240-word chunkings are close: article recall 74, 74 and 74 of 75 and evidence recall 54, 57
  and 55 of 66 at 1,200 words (60, 63 and 60 at 2,400). The 240-word chunks store 4,598 vectors
  instead of 8,658 or 19,065 and a 13 MB graph instead of 17 or 29. Section 4,
  `results/chunk60_ef_100.md`, `results/chunk120_ef_100.md`, `results/chunk240_ef_100.md`.
- **Embedding prefix kept** (`"title > section path: "`). The exact ranking, which does not depend
  on a graph, decides it: the prefix raises evidence recall@20 from 61 to 63 of 66 and leaves
  article recall@10 at 74 of 75 (section 5, `results/prefix_on_inline.md`,
  `results/prefix_off_inline.md`).
- **Filtered queries: strategy `inline`**, `overfetch` as the low-latency option. Inline always
  fills `k`; overfetch filters after the index search and comes back short under a selective
  filter. Under `heading LIKE '%History%'` on the demo corpus (117 of 4,598 chunks pass) inline
  takes 3.07 ms at k=10 and is never short, overfetch 10 takes 0.78 ms and is short for 69 of 75
  claims (section 6, `results/strategy_*.md`). On the 3,000 claims at k=5 the same filter gives
  16.49 ms (S0) and 26.59 ms (S1) for inline, and 1.63 and 1.87 ms for overfetch 10, which is
  short for 2,472 and 2,425 of them (`results/scaled/timing/`).
- **Final configuration** (section 7: 4,598 chunks, 1,316 hatnote units excluded): article
  recall@1/5/10/20 63/74/74/74 of 75, evidence recall@1/5/10/20 39/55/60/65 of 66; SQL p50/p95 at
  k=10 0.51/0.94 ms, query embedding 4.58 ms; hit lists identical to the exact ranking for 74/75.
- **Reproducibility.** Every graph build is another approximation (on S1, see `builds` above), and
  within one server process the results are deterministic, except right after the table is
  reopened (on S1, 10 of 3,000 bare hit lists changed between two passes,
  `results/scaled/run1/s1/cache_size.md`). A container restart changed 1 of 75 hit lists on the M=6 index
  and none on M=16 at ef_search 100, without changing recall (`results/stability_comparison.md`,
  `results/final_restart_comparison.md`). Running `wikilense eval` after `wikilense ingest` with
  the defaults checks the counts.

## CLI and web (`cli.py`, `web.py`)

The subcommands are `init-db`, `ingest`, `query`, `eval` and `serve`; [EXAMPLES.md](EXAMPLES.md)
shows every query option, the web page and its API. The exit codes are pinned by
`tests/test_cli.py`: 0 for success, 1 for an expected failure (one line on stderr), 2 for bad
arguments and 130 for an interrupt. The web page is one HTML form that calls `GET /api/search` and
shows the hits and the SQL that ran. It uses no JavaScript framework and no external assets. A
link with the form's parameters fills the form and runs the search. Two things are kept on
purpose. The web API's 503 answer repeats the driver's connection error, because the server binds
to 127.0.0.1 by default and the text is what a reader needs to fix a setup. The Swagger page loads
its script from a CDN, which docs/EXAMPLES.md states. `make examples` runs `sql/examples.sql`
through the `mariadb` client in the container, which is the path for a reader without Python.

## Page metadata and the four filters

The four metadata filters are: only articles in this category, longer than this, edited since that
date, linked from that other article. Length (`page.n_words`) and links (`link`) come with the
FEVEROUS pages. Categories and edit dates do not: the pages have no such field, and they stop
before the category box and the "last edited" footer.

- **Source: each page's December 2020 revision from the MediaWiki API**
  (`scripts/fetch_page_meta.py`, committed as `data/corpus/page_meta.jsonl` and `PAGE_META.md`).
  FEVEROUS scraped Wikipedia in December 2020 and did not release its per-page date stamp (paper,
  Sec 7.7). Today's "last edited" date or categories would pair 2026 metadata with 2020 text. The
  chosen revision is the one whose wikitext holds the most FEVEROUS sentence probes: 94.5% of the
  probes occur in the chosen revisions, and today's "Albert Einstein" page holds 260 of 484,
  against 466 in the December 2020 revision. Its `[[Category:...]]` links are the categories.
  Categories that templates add when the page is rendered are not in the wikitext, so the category
  filter does not see them; most of those are hidden maintenance categories.
  `docs/DATA.md` has the details.
- **Schema and filters.** The schema has `page.revision_id` and `page.last_edited` (UTC, indexed,
  NULL without a record), plus `category` and `page_category` as in the table above.
  `Filters.category` adds
  `STRAIGHT_JOIN (SELECT DISTINCT page_category.page_id ... WHERE category.name = %s) AS
  in_category`; `edited_since` / `edited_before` add `page.last_edited >= %s` / `< %s`, with dates
  and naive datetimes read as UTC.
- **When the optimizer keeps the vector index under a filter** (`results/planner_choices.md`).
  Nine of the eleven filters plan `chunk` through the index for every claim. The two most selective
  range predicates on indexed `page` columns, `max_words 500` (25 chunks pass) and `edited_before
  2020-06-01` (4 pass), make it scan all 4,598 chunks and sort instead, which gives the exact result
  at 5.2 / 3.1 ms p50. No query returned fewer than min(k, passing) hits.

## Setup, packaging and the README

- **The README's first screen answers one question**: the setup command, one query and the output
  of a recorded run (`results/fresh_clone_run.md`), so a reader who only wants to type one question
  sees the whole path; the reference material is in `docs/`. `tests/test_docs.py` keeps that block
  equal to the record, caps the README at 350 lines, and checks every relative link and anchor of
  every Markdown file and the number of tests the README and ARCHITECTURE.md state.
- **The README has a section "Where this approach loses"**, so the case for searching inside the
  database is weighed against its cost. `scripts/compare_separate_index.py` runs MariaDB against a
  copy of the chunk vectors in the application's memory (an exact NumPy scan on one thread,
  pre-filtered, followed by the `SELECT` of the hit rows): under the History filter and the four
  metadata filters MariaDB's joined form costs 3.20 / 3.56 and 3.46 / 3.45 ms in two runs and the
  copy 0.78 / 1.04 and 0.71 / 0.73 ms, with equally complete results (`results/separate_index.md`,
  `results/separate_index_repeat.md`). No dedicated vector store was measured.
- **`make setup` generates the `.env` passwords** with `scripts/make_env.py`, which writes two
  random 128-bit hex strings with mode 0600, never prints them and never touches an existing
  file. The setup is one command, which is what the README's first screen shows. When a new
  `.env` meets a data volume created with other passwords, the run ends with one line that names
  the cause and both ways out (`cli.ACCESS_DENIED_HINT`). `make setup` stops with one line when
  the Python is too old for what it installs, and rebuilds a venv whose interpreter no longer
  runs.
- **`make setup` installs the version ranges; the lock files stay opt-in.** A lock file as the
  default would make the setup fail on Python 3.10 and 3.11, on macOS 13, and (for the GPU lock)
  everywhere but Linux. The cost is that a new release of a dependency can break the default path;
  `make setup CPU=1` is the pinned way in, and CI installs it on every push.
- **`requires-python` is `>=3.10`.** The package's own code needs nothing newer, and the ranges
  resolve for Python 3.10 to 3.14 on Linux, Windows and macOS 14. Narrowing it to the lock files'
  3.12 (numpy 2.5.3) would refuse setups that work, and `make setup` reports that a lock file
  needs 3.12 before pip does. It has run on Python 3.12 (CI and development) and 3.14 (a fresh
  clone).
- **`requirements-cpu.lock` has per-platform PyTorch lines**, so `make setup CPU=1` resolves on
  macOS: `torch==2.14.0+cpu` exists for Linux and Windows only, so macOS gets `torch==2.14.0`.
  Checked by resolution only, not run on a Mac or on Windows (`docs/SETUP.md`, "Platforms").
- **The MariaDB image is pinned to `mariadb:11.8.9`** in `docker-compose.yml` and in CI. Every
  measured behaviour and every plan a test pins is that of 11.8.9, a later patch release may change
  them, and a reader should get the server the numbers describe. The cost is that security fixes
  arrive only when the pin is moved, and moving it means rerunning the tests and
  `scripts/measure_index_behaviour.py`.
- `make eval`'s `results/baseline.*` is ignored by git; `data/corpus/NOTICE.md` states the licence
  and attribution of the committed Wikipedia text and FEVEROUS claims.

## The scaled run: 3,000 claims on two larger corpora

The 75 claims on 100 pages are the development set; the scaled run measures the same design on
more claims and larger corpora. [docs/DATA.md](DATA.md) says how the claims and the distractor
pages were picked, [docs/EVALUATION.md](EVALUATION.md) has every table, and
[results/scaled/](../results/scaled/) holds the result files.

- **Claims.** The run uses 3,000 FEVEROUS train+dev claims, 1,000 per label. Per label, 700 have
  every evidence set on one page and 300 have every set spanning two or more pages.
  - Only claims with a text-only evidence set (sentences and list items) are used, because tables
    are not indexed. Of the FEVEROUS train+dev claims, 56% (43,957 of 79,181) have no text-only
    evidence set.
  - Only claims that can be found are used: every element of a text-only set must be inside a
    chunk. Hatnotes are kept out of chunks by design, so 112 claims were left out before sampling.
  - The claims are sampled with `random.Random(20261111)`. The challenge tags are recorded, not
    balanced on.
  - The label mix is not FEVEROUS's (there 3.5% of claims are NOT ENOUGH INFO), so results are
    reported per group and also reweighted to FEVEROUS's mix (in docs/EVALUATION.md).
- **Corpora: two sizes.** S0 is the gold pages only (3,851 pages, 43,134 chunks). S1 is S0 plus
  one distractor page per gold article, picked by "rule D": each gold article's most-linked page
  (its "favourite link"). S1 has 7,686 pages and 132,359 chunks. Random pages were not used
  because they barely distract (see docs/DATA.md).
- **Settings not re-chosen.** 240-word chunks, overlap 1, bge-small-en-v1.5 with the title prefix,
  M=16, `mhnsw_ef_search` 100 and strategy `inline` were chosen on the 75-claim development set
  (the 100-page corpus) and stay fixed for the scaled run.
- **Headline k = 5**, matching FEVEROUS's own k = 5 and the 5 chunks an LLM fact-checker reads.
  The full table at k = 1, 3, 5, 10, 20 is still given, and the 100-page results keep their
  k = 10.
- **Two evidence scores, and page coverage.** Ours requires a complete gold set inside the top-k
  chunks. The FEVEROUS-style score takes the best 5 sentences and 25 list items of the top 10
  chunks, ranked by embedding each sentence against the claim. Page coverage is defined as in the
  FEVEROUS baseline code. Even the FEVEROUS-style score is easier than the paper's, because the
  claims are text-only and the corpus is nearly three orders of magnitude smaller.
- **Page metadata at scale.** Categories and edit dates were fetched for every S1 page from the
  MediaWiki API (each page's December 2020 revision, as in "Page metadata and the four filters"),
  so the four filters work at scale.
- **Timing under the laptop's heat-safe setting** (CPU capped at 50%, GPU at 2,100 MHz), so the
  scaled times are compared only with each other, not with the 100-page ones.
- **Database measurements at scale.** Filter values are picked per database by a stated rule:
  target shares of 50%, 5%, 0.5% and 0.05% of the chunk table (`scripts/scaled_filters.py`). The
  join-order measurement runs on a fixed random sample of 200 claims, because by the 20-claim pilot
  (`results/scaled/pilot/`) all 3,000 would take about 100 hours per run. Each claim gets one timed
  pass, since 3,000 claims already give 3,000 timings per median. Each measurement ran once so
  far; a rerun goes into `results/scaled/run2/`.
