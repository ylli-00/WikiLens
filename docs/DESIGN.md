# WikiLense design (phase 1, 2026-09-17)

Retrieval over Wikipedia inside MariaDB 11.8: chunk pages, embed chunks, store them in a
`VECTOR` column with a cosine `VECTOR INDEX`, query with `VEC_DISTANCE_COSINE`, and combine the
semantic search with ordinary SQL predicates and joins in the same statement. FEVEROUS claims are
the queries; their gold evidence is the ground truth for recall.

This file is the contract between modules. Every value marked *provisional* is a working choice
made so that coding does not stall; the final choice is made with the experiments and documented
in the README.

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
`^(?P<page>.+)_(?P<type>sentence|cell|header_cell|table_caption|item)_(?P<nums>\d+(?:_\d+)*)$`,
so that a title containing an underscore still parses. `corpus.parse_element_id(el)` returns
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

- `chunk_units(units, max_words=120, overlap_units=1)` (both *provisional*) is a pure function
  over one page's text units. A chunk never crosses a section boundary.
- Fill a chunk with consecutive units until adding the next unit would exceed `max_words`; a
  single unit longer than `max_words` becomes its own chunk. The next chunk starts
  `overlap_units` units before the end of the previous chunk, but only when the previous chunk
  has more than `overlap_units` units (no infinite loops, no duplicate chunks).
- `Chunk(ordinal, section_ordinal, element_keys, text, n_words)`; `text` is the cleaned units
  joined by a single space; `ordinal` counts chunks per page from 0.
- The text that is embedded is `f"{title} > {section_path}: {text}"` when the section path is
  non-empty, else `f"{title}: {text}"`. The stored `chunk.text` stays the plain text. Ingest adds
  the prefix (*provisional*, to be tested against no prefix).

## Embedding (`embedding.py`)

- Model *provisional*: `BAAI/bge-small-en-v1.5`, 384 dimensions, MIT licence, 33M parameters.
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
| `chunk` | `chunk_id`, `page_id` FK, `section_id` FK, `ordinal`, `text`, `n_words`, `embedding VECTOR(384) NOT NULL` | UNIQUE (`page_id`, `ordinal`); `VECTOR INDEX (embedding) M=6 DISTANCE=cosine` (*provisional* M) |
| `chunk_sentence` | (`chunk_id`, `sentence_id`) | the chunk-to-sentence map; index on `sentence_id` |
| `link` | `link_id`, `from_page_id` FK, `to_title`, `to_page_id` NULL FK, `source_element` | `to_page_id` resolved when the target is in the corpus; indexes on `from_page_id`, `to_page_id`, `to_title` |
| `claim` | `claim_id` (FEVEROUS id), `split`, `text`, `label`, `challenge` | |
| `claim_evidence` | (`claim_id`, `evidence_set`, `position`), `element_id`, `page_title`, `element_type`, `page_id` NULL, `sentence_id` NULL | resolved ids when the page / text unit is in the corpus; indexes on `page_id`, `sentence_id` |
| `ingest_meta` | `key`, `value` | embedding model, dim, chunk parameters, index M, corpus sha256, ingested_at |

Vector dimension 384 is written literally in the DDL and checked against `Embedder.dim` at
ingest; a mismatch is an error with a clear message.

`config.Settings` (from environment, `.env` loaded with python-dotenv): `WIKILENSE_DB_HOST`,
`WIKILENSE_DB_PORT`, `WIKILENSE_DB_USER`, `WIKILENSE_DB_PASSWORD`, `WIKILENSE_DB_NAME`,
`WIKILENSE_EMBEDDING_MODEL` (default above), `WIKILENSE_CHUNK_MAX_WORDS` (120),
`WIKILENSE_CHUNK_OVERLAP_UNITS` (1), `WIKILENSE_VECTOR_DIM` (384), `WIKILENSE_INDEX_M` (6).

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
index looks at; `mhnsw_max_cache_size` (16 MB default, global) bounds the index cache.

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
