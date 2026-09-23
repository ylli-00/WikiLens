A project for area 4, "Vector search and RAG, natively" for the [MariaDB student database projects, 2026-09](https://mariadb.org/bachelor_hackathon_2026-09/).

# WikiLense

> **Scope.** The 75 claims are every FEVEROUS train and dev claim whose evidence lies inside one
> Wikipedia shard, a development-sized set on which every statement below is measured. Multi-page
> claims, a chosen benchmark claim set and a larger corpus are the next phase (see Limitations).

Semantic search over Wikipedia inside MariaDB. Wikipedia pages are cut into chunks, each chunk is
embedded with a small open model and stored in a `VECTOR(384)` column with a cosine `VECTOR INDEX`,
and a query is one SQL statement: `ORDER BY VEC_DISTANCE_COSINE(...)` combined with ordinary
predicates (page length, section heading, links between pages) and joins back to the relational
tables. Fact-checking claims from the FEVEROUS benchmark are the queries, and their gold evidence
sentences are the ground truth, so "it works" is a recall number and a latency number.

Developed against **MariaDB 11.8.9** (Docker image `mariadb:11.8`), Python 3.12, PyMySQL (1.2.0
for the numbers in `results/`, 1.2.3 pinned in the lock files), sentence-transformers 6.0 with
`BAAI/bge-small-en-v1.5`. Everything in this README is measured on
the committed corpus (100 pages, 75 claims). The headline and the `final_*` runs use its 240-word
ingest (4,598 chunks); the `M`, `ef_search`, stability and prefix sweeps use the 120-word ingest
(8,658 chunks), and the chunk-size sweep uses one ingest per size. The numbers come from
[results/SUMMARY.md](results/SUMMARY.md), produced by [scripts/run_experiments.py](scripts/run_experiments.py).

## What we found

- **Retrieval works and is fast.** With the chosen settings, the gold page of a claim is among the
  pages of the top 5 chunks for 74 of 75 claims, and the complete gold evidence is inside the top
  10 chunks for 60 of 66 claims (65 of 66 at k=20). The vector query itself takes 0.9 ms at the
  median and 1.4 ms at the 95th percentile for k=10; embedding the query on the GPU takes 4.7 ms.
- **The approximate index can be made exact for this corpus.** On the 120-word ingest, MariaDB's
  HNSW index with its server defaults (`M=6`, `mhnsw_ef_search=20`) found the gold page for 68 of
  75 claims at k=10 where an exact ranking of the same vectors finds 74; rebuilt with `M=16` on the
  same vectors it finds 72, and 74 with `mhnsw_ef_search=100`. On the final ingest, `M=16` with
  100 candidates per query returns the exact ranking's hit list for 74 of 75 claims at the same
  sub-millisecond cost. That is the largest single lever we measured.
- **Joins change how the index is used.** A bare `ORDER BY VEC_DISTANCE_COSINE ... LIMIT k` is
  the HNSW search proper. Add a join to `page` or `section` in the same statement and MariaDB
  still reports `key: embedding`, but it now ranks exactly over the index, about 2.6 ms here,
  independent of `k` and of `ef_search`, plus a walk that continues until `k` rows pass the
  predicates. That gives exact, filter-complete results at a cost that grows with the table.
- **The remaining misses are inside the right page.** When the gold page is handed to the query as
  a filter (an oracle), the gold sentences are in the first chunk for 42 of 66 claims and within 20
  chunks for all 66. Chunk ranking within a page, not page retrieval, is where recall is lost.
- **Results are reproducible within a few claims, not to the chunk.** Every HNSW build is a
  different graph, and a server restart changed 3 of 75 hit lists on the `M=6` index (1 of 75 on
  `M=16` at `ef_search=100`), without changing recall. The index cache size did not change a
  single hit list. The README says which numbers are exact and which are approximate.

## Quick start

```
git clone https://github.com/ylli-00/WikiLens.git && cd WikiLens
make setup
```

`make setup` does four things, which you can also run by hand:

1. `cp .env.example .env` and stops so you can set the two passwords in `.env` (the MariaDB root
   password and the application user's password). Run `make setup` again afterwards.
2. `docker compose up -d --wait`: MariaDB 11.8 with `--mhnsw-max-cache-size=536870912` (512 MB),
   bound to
   `127.0.0.1:3306`, with the databases `wikilense` (the corpus) and `wikilense_test` (the test
   suite). The healthcheck waits until InnoDB is ready.
3. `python3 -m venv .venv` (make creates it before starting the container) and
   `.venv/bin/pip install -r requirements-dev.txt -e .`. The package reads `sql/schema.sql` and `data/corpus/` from the checkout, so the editable
   install is required. On a machine without a GPU, run `make setup CPU=1` instead, which installs the pinned CPU build
   from `requirements-cpu.lock` (what CI does).
4. `.venv/bin/wikilense ingest`: creates the tables, parses the 100 committed pages, embeds the
   chunks and loads everything. The embedding model (about 134 MB) downloads to `~/.cache/huggingface`
   on first use. The run takes 21 s on an RTX 4060 laptop GPU, of which 16 s is embedding; on a
   CPU the embedding runs at about 78 passages per second, so expect about a minute.

The ingest ends with the seconds per stage and a report you can compare against this one (the
1,316 hatnote units kept out of chunks are recorded in `ingest_meta`):

```
  pages 100, sections 3173, sentences 33637 (empty 372), chunks 4598
  links 40542 (resolved 373, skipped 0)
  claims 75, evidence ids 114 (page resolved 114, sentence resolved 87)
```

Then `make test` (255 tests, 20 s), `make eval` (the headline table below, about 15 s),
`make serve` (the web page at http://127.0.0.1:8000), `make experiments` (the whole protocol,
about 10 minutes) and `make down`.

## Try it

`wikilense query` embeds the text, runs one SQL statement and prints the hits with their distance,
page title, section path and text. `--explain` prints the statement, its parameters and the
`EXPLAIN` rows.

```
$ wikilense query "Which river rises and ends entirely within Switzerland?" --k 3 --explain
 1. 0.3091  Aare  [chunk_id 1, 127 words]
    The Aare (German: [ˈaːrə]) or Aar (German: [aːɐ̯] (listen)) is a tributary of the High Rhine and
    the longest river that both rises and ends entirely within Switzerland. ...
 2. 0.3593  Aare > Course  [chunk_id 4, 158 words]
    ...
 3. 0.3620  Aare > Course  [chunk_id 3, 228 words]
    ...
SQL:
    SELECT chunk.chunk_id, chunk.page_id, page.title, section.path AS section_path, chunk.ordinal AS
    chunk_ordinal, chunk.n_words, VEC_DISTANCE_COSINE(chunk.embedding, %s) AS distance, chunk.text
    FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id STRAIGHT_JOIN section ON
    section.section_id = chunk.section_id ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s
parameters: <vector[384]>, <vector[384]>, 3
EXPLAIN:
    id  select_type  table    type    possible_keys                           key        key_len  ref                         rows  Extra
    1   SIMPLE       chunk    index   uq_chunk_page_ordinal,ix_chunk_section  embedding  1538     NULL                        3
    1   SIMPLE       page     eq_ref  PRIMARY                                 PRIMARY    4        wikilense.chunk.page_id     1
    1   SIMPLE       section  eq_ref  PRIMARY                                 PRIMARY    4        wikilense.chunk.section_id  1
3 hits; model load 6135 ms, embedding 6.1 ms, SQL 4.8 ms (BAAI/bge-small-en-v1.5 on cuda, strategy inline, ef_search 100)
```

The vector index is the `key` of the first row; `page` and `section` are joined by primary key.
The model load happens once per process (the web server pays it once at start-up). The timings
are one sample and vary from run to run (the first statement of a new connection pays plan
warm-up); the CLI's SQL figure also includes the `mhnsw_ef_search` session round trips, which the
harness numbers under Evaluation (warm p50 2.6 ms for this statement) exclude.

**A predicate on the section heading**, in the same statement (`WHERE section.heading LIKE %s`
appears in the SQL and `Using where` on the `section` row of the EXPLAIN):

```
$ wikilense query "Roman settlement and medieval castle" --k 3 --heading '%History%'
 1. 0.3490  Lincoln, England > History > Roman history: Lindum Colonia  [chunk_id 3306, 208 words]
    The Romans conquered this part of Britain in AD 48 and shortly afterwards built a legionary
    fortress high on a hill ...
 2. 0.3500  Andorra > History > Prehistory  [chunk_id 388, 209 words]
```

**A predicate on the link graph**: only pages that the page "AFC Ajax" links to.

```
$ wikilense query "football club league titles" --k 3 --linked-from "AFC Ajax"
 1. 0.2935  Juventus F.C. > Honours  [chunk_id 4074, 214 words]
    Italy's most successful club of the 20th century ... have won the Italian League Championship ...
```

**The hybrid strategy** fuses the vector ranking with a `FULLTEXT` `MATCH ... AGAINST` ranking by
reciprocal rank fusion, in one statement with three CTEs (the vector list, the full-text list
and their fusion):

```
$ wikilense query "Coppa Italia finals won" --k 3 --strategy rrf
 1. 0.2217  Juventus F.C. > History > Historic four consecutive doubles ... [chunk_id 4054, 236 words, rrf score 0.0323]
    On 17 May 2017, Juventus won their 12th Coppa Italia title in a 2–0 win over Lazio ...
```

**The join back to the sentences** a chunk was built from (`--sentences`), which is how the
evaluation matches chunks to FEVEROUS evidence ids:

```
$ wikilense query "Aare hydroelectric plants" --k 1 --sentences
 1. 0.2491  Aare  [chunk_id 1, 127 words]
    ...
    sentences (chunk_sentence -> sentence, page order):
        sentence_2  The Aare (German: [ˈaːrə]) or Aar (German: [aːɐ̯] (listen)) is a tributary of ...
        sentence_3  Its total length from its source to its junction with the Rhine comprises about ...
```

Other options: `--title TITLE` (repeatable, exact page titles), `--min-words` / `--max-words`
(page length), `--path` (section path pattern),
`--links-to TITLE` (pages that link to a title), `--strategy inline|overfetch|none|rrf`,
`--overfetch N`, `--ef-search N` (0 keeps the server value), `--json` for scripts.

`wikilense serve` starts a web page with the same query and filters; the API behind it,
`GET /api/search?q=...&k=5&heading=%25History%25&sentences=1`, returns `hits`, the `sql` that ran,
its `explain` rows, the effective `ef_search` and `timing_ms`. `/docs` is enabled so the API can be
tried from the browser.

## The data

- `data/corpus/pages.jsonl` (8.0 MB): 100 Wikipedia pages in the FEVEROUS format (December 2020
  articles with sentence, section, table and list elements). `data/corpus/claims.jsonl` (141 KB):
  75 FEVEROUS claims with their evidence sets. Both are committed, with SHA-256 checksums and the
  page list in [data/corpus/MANIFEST.md](data/corpus/MANIFEST.md).
- **Selection rule.** The 75 claims (65 train, 10 dev; 7 SUPPORTS, 58 REFUTES, 10 NOT ENOUGH INFO)
  are all claims of FEVEROUS train and dev whose evidence pages are all inside the first Wikipedia
  shard, `wiki_000.jsonl`. Their evidence cites 54 distinct pages. The other 46 pages are the pages
  those 54 link to most often, so the corpus contains realistic distractors and the link filters
  have something to work on. 65 claims have an evidence set made only of sentences; 66 claims count for
  evidence recall, because one more claim's set mixes sentences and list items, which are both text
  units. The 27 evidence ids that point at table cells cannot be matched: tables are not indexed
  in this phase.
- **Rebuilding it** (about 5 s, byte-identical): download the shard with
  `python3 scripts/fetch_wiki_shard.py wiki_000.jsonl` (44 MB, fetched from inside the 9.9 GB
  archive with HTTP range requests and CRC-checked), put `feverous_train_challenges.jsonl` and
  `feverous_dev_challenges.jsonl` from https://fever.ai/dataset/feverous.html into `data/feverous/`,
  and run `python3 scripts/build_corpus.py`.
- FEVEROUS: Aly et al., 2021 (https://arxiv.org/abs/2106.05707); Wikipedia text under CC BY-SA 3.0.

## The data model

Nine InnoDB tables, all in [sql/schema.sql](sql/schema.sql), which the code executes as it is
(`wikilense.db.SCHEMA_COLUMNS` mirrors it and a test compares both against the live server).

| Table | Holds | Why it is shaped this way |
|---|---|---|
| `page` | one row per article: title, counts of sentences, items, words, characters, sections, tables, lists | `n_words` is indexed so "articles longer than N" is a range scan; `title` is `utf8mb4_bin` so matching is exact |
| `section` | the heading tree: ordinal, heading, level, path such as `History > Roman history` | the lead is section 0; `heading` keeps the server's accent- and case-insensitive collation on purpose, so `LIKE '%history%'` matches `History` |
| `sentence` | one row per text unit (a sentence or a list item), with its FEVEROUS element key | this is the unit of gold evidence; every unit is stored, even empty ones and hatnotes, so evidence ids always resolve |
| `chunk` | the retrieval unit: text, word count, `embedding VECTOR(384) NOT NULL`, the `VECTOR INDEX` and a `FULLTEXT` index | see the DDL below |
| `chunk_sentence` | which sentences each chunk covers (33,783 rows) | the join that turns "chunk in the top k" into "gold sentence in the top k" |
| `link` | every `[[link]]` in a page, with the target resolved to a `page_id` when it is in the corpus | "pages linked from X" and "pages linking to X" become joins |
| `claim` | the FEVEROUS claim, label, split, challenge | the queries of the evaluation |
| `claim_evidence` | every evidence element id, with `page_id` and `sentence_id` resolved when possible | the ground truth of the evaluation, joined by id rather than by text |
| `ingest_meta` | model, dimension, chunk size, index parameters, corpus checksums, server version, time | every results file records the ingest it was measured on |

The chunk table, verbatim:

```sql
CREATE TABLE IF NOT EXISTS chunk (
    chunk_id   INT UNSIGNED NOT NULL AUTO_INCREMENT,
    page_id    INT UNSIGNED NOT NULL,
    section_id INT UNSIGNED NOT NULL,
    ordinal    INT UNSIGNED NOT NULL,
    text       TEXT NOT NULL,
    n_words    INT UNSIGNED NOT NULL,
    embedding  VECTOR(384) NOT NULL,
    PRIMARY KEY (chunk_id),
    UNIQUE KEY uq_chunk_page_ordinal (page_id, ordinal),
    KEY ix_chunk_section (section_id),
    FULLTEXT KEY ft_chunk_text (text),
    VECTOR INDEX (embedding) M=16 DISTANCE=cosine,
    CONSTRAINT fk_chunk_page FOREIGN KEY (page_id)
        REFERENCES page (page_id) ON DELETE CASCADE,
    CONSTRAINT fk_chunk_section FOREIGN KEY (section_id)
        REFERENCES section (section_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

Ownership references cascade (deleting a page deletes its sections, sentences, chunks and vectors
in one statement; a test checks it). Resolution references (`link.to_page_id`,
`claim_evidence.page_id`, `claim_evidence.sentence_id`) are `ON DELETE SET NULL`, because a link or
a piece of gold evidence must survive the page it points to being re-ingested. The vector
dimension is literal in the DDL and checked against the model at ingest.

## The SQL

Every statement is a named constant in [wikilense/search.py](wikilense/search.py), assembled from
fixed fragments; values are always bound as parameters, and a vector parameter is bound as the
raw float32 bytes (`db.vec_param`), sent as a `_binary X'...'` literal (see "Binding" below).

The bare index query, the HNSW search proper; strategies `none` and `overfetch` run it as a derived
table joined back to `page` and `section` for the hit metadata (`none` is `overfetch` with factor 1
and no filters):

```sql
SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, ?) AS distance
FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT ?
```

Predicates and joins in the same statement (strategy `inline`, the default of `search()` and of the
CLI, with or without filters; `WHERE` fragments are added only for the filters that are given):

```sql
SELECT chunk.chunk_id, chunk.page_id, page.title, section.path AS section_path,
       chunk.ordinal AS chunk_ordinal, chunk.n_words,
       VEC_DISTANCE_COSINE(chunk.embedding, ?) AS distance, chunk.text
FROM chunk
  STRAIGHT_JOIN page ON page.page_id = chunk.page_id
  STRAIGHT_JOIN section ON section.section_id = chunk.section_id
  [STRAIGHT_JOIN (SELECT DISTINCT link.to_page_id AS page_id FROM link
                  JOIN page AS src ON src.page_id = link.from_page_id
                  WHERE src.title = ? AND link.to_page_id IS NOT NULL) AS linked
                 ON linked.page_id = chunk.page_id]
WHERE page.n_words >= ? AND section.heading LIKE ? ...
ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, ?) LIMIT ?
```

The bounded form (strategy `overfetch`): the index returns `k * overfetch` candidates in a derived
table, and the outer query filters, joins and re-limits:

```sql
SELECT ... , knn.distance, chunk.text
FROM (SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, ?) AS distance
      FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT ?) AS knn
JOIN chunk ON chunk.chunk_id = knn.chunk_id
JOIN page ON page.page_id = chunk.page_id
JOIN section ON section.section_id = chunk.section_id
WHERE ... ORDER BY knn.distance, chunk.chunk_id LIMIT ?
```

The hybrid (strategy `rrf`): the vector top-N and the full-text top-N, each ranked with
`ROW_NUMBER()`, fused by reciprocal rank fusion:

```sql
WITH vec AS (SELECT knn.chunk_id, ROW_NUMBER() OVER (ORDER BY knn.distance, knn.chunk_id) AS rnk
             FROM (SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, ?) AS distance
                   FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT ?) AS knn),
     ft AS (SELECT matched.chunk_id, ROW_NUMBER() OVER (ORDER BY matched.relevance DESC, matched.chunk_id) AS rnk
            FROM (SELECT chunk_id, MATCH(text) AGAINST (? IN NATURAL LANGUAGE MODE) AS relevance
                  FROM chunk WHERE MATCH(text) AGAINST (? IN NATURAL LANGUAGE MODE)
                  ORDER BY relevance DESC, chunk_id LIMIT ?) AS matched),
     fused AS (SELECT ranked.chunk_id, SUM(CAST(1 AS DOUBLE) / (60 + ranked.rnk)) AS score
               FROM (SELECT chunk_id, rnk FROM vec UNION ALL SELECT chunk_id, rnk FROM ft) AS ranked
               GROUP BY ranked.chunk_id)
SELECT ..., VEC_DISTANCE_COSINE(chunk.embedding, ?) AS distance, chunk.text, fused.score
FROM fused JOIN chunk ON chunk.chunk_id = fused.chunk_id JOIN page ... JOIN section ...
ORDER BY fused.score DESC, chunk.chunk_id LIMIT ?
```

The join back to evidence: `chunk_sentence JOIN sentence` for the hit chunks, ordered by page
position (`--sentences`).

The area text names categories and edit dates as example predicates. The FEVEROUS dump carries
neither (no category or revision fields in the page objects), so the relational predicates here
are page length (`page.n_words`), section heading and path, the link graph (`link`) and the title;
a category or last-edit column would join the same way.

[sql/examples.sql](sql/examples.sql) has the same statement shapes with a stored chunk's vector as
the query, for the `mariadb` client alone, without Python or the model.

## How MariaDB 11.8.9 uses the vector index, measured

Everything here was measured on this corpus with `EXPLAIN`, `ANALYZE` and the harness during
development; the tests in [tests/test_db.py](tests/test_db.py) and
[tests/test_search.py](tests/test_search.py) assert the positive facts on synthetic data (`key:
embedding`, `r_rows` = `k`, both indexes for `rrf`, session scope of `ef_search`).

- **What drives the index.** Only `ORDER BY VEC_DISTANCE_COSINE(column, constant) LIMIT n`, sorted
  ascending (the alias form `ORDER BY distance` is identical). Wrapping the distance in an
  expression, a `WHERE VEC_DISTANCE(...) < t` without `ORDER BY ... LIMIT`, or a second `ORDER BY`
  key (`..., chunk_id`) all drop the index, so ties are broken in Python.
- **Two cost regimes.** The bare query (`none`, `overfetch`) is the HNSW search: 0.6 to 1.1 ms on
  4,598 chunks, reads exactly `LIMIT` rows, and its accuracy is set by `mhnsw_ef_search`. With a
  join in the statement (`inline`), `EXPLAIN` still says `type: index, key: embedding`, but the
  server ranks exactly over the index: 2.6 ms here at every `k` and every `ef_search`, with hit
  lists that do not depend on `ef_search` (identical at 20 and 100 for 75 of 75 claims) and equal to
  the known exact order on the synthetic test corpus, and then walks until `k` rows pass the
  predicates (`ANALYZE` reports `r_rows` = `k` when nothing is filtered, more rows under a heading
  filter that keeps 2.5% of the chunks, 117 of 4,598, and the whole table when the filter matches
  nothing).
- **Bounded versus complete.** `overfetch` stays at the bare cost (1.0 ms under a filter that keeps
  almost every chunk) but returns fewer than `k` rows when the filter is selective: under
  `heading LIKE '%History%'` (117 of 4,598 chunks pass) it came back short for 69 of 75 queries
  with `overfetch 10` and 20 with `overfetch 50`, `inline` never, at 3.6 ms median and 4.9 ms p95.
- **Join order matters.** With a plain `JOIN`, the optimizer started from `page` on stale
  statistics right after a bulk insert and lost the vector index; the statements use
  `STRAIGHT_JOIN` with `chunk` first and the ingest runs `ANALYZE TABLE`. The `IN (subquery)` form
  of the link filter becomes a semi-join that starts from `link` and loses the index too; it is
  faster only for a very selective filter, where sorting the few matching chunks costs less than
  the index walk. A materialised `SELECT DISTINCT` derived table keeps the index in every run.
- **Parameters.** `mhnsw_ef_search` is a session variable (set per query, restored after); values
  outside its range are clamped with a warning, not an error. `mhnsw_max_cache_size` is global;
  the 16 MB default is already the size of the `M=6` graph of the 8,658-chunk 120-word ingest
  (17 MB with `M=16`), so the Docker configuration starts the server with 512 MB. The graph lives in a hidden InnoDB table
  (`chunk#i#NN`), not in `SHOW TABLE STATUS`'s `Index_length`.
- **Binding.** Raw float32 bytes bind as a `_binary` literal (`INSERT` and inside
  `VEC_DISTANCE_COSINE`): PyMySQL 1.2.3 and later send `_binary X'<hex>'` by themselves; 1.2.0 to
  1.2.2 send `_binary'...'` only on a connection opened with `binary_prefix=True`, which
  `db.connect` sets for every version; without it, PyMySQL 1.2.0 sends bytes as a string and
  MariaDB answers "Incorrect vector value";
  `VEC_FromText(?)` with a JSON list and `UNHEX(?)` work everywhere. A query vector of the wrong
  dimension returns `NULL` distances, silently.
- **Transactions.** A chunk inserted inside a transaction is visible to the index in the same
  connection and gone after `ROLLBACK`; `UPDATE` of an embedding changes the distance; `DELETE`
  removes it from the results; deleting a page cascades to its vectors. No synchronisation job, no
  second store.
- **Approximation and state.** Every build of the HNSW graph is a different approximation: two
  `M=6` builds of the same vectors agreed on 16 of 75 hit lists. Within one server process the
  results are deterministic, and the cache size does not change them (75 of 75 identical at 16 MB
  and 512 MB); a `docker compose restart` changed 3 of 75 hit lists on the `M=6` index and 1 of 75
  on `M=16` at `ef_search=100`, without changing recall.

## Parameters and why

| Parameter | Chosen | Evidence (details in results/SUMMARY.md) |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5`, 384 dimensions, 33M parameters, MIT licence | chosen in phase 1 for passage retrieval at this size; no second model was measured. 1,059 passages/s on the RTX 4060 and 78/s on the CPU for 100-word texts (phase-1 measurement in docs/DESIGN.md, not part of `results/`), so a grader without a GPU can ingest; normalised vectors, so cosine distance is `1 - dot` |
| Embedded text | `title > section path: chunk text` | on the exact ranking the prefix raises evidence recall@20 from 61 to 63 of 66 and leaves article recall@10 at 74 of 75 (section 5) |
| Chunk size | 240 words, overlap 1 unit, never across a section boundary | at equal retrieved text the 60-, 120- and 240-word chunkings tie: article recall 74/74/74 of 75 and evidence recall 54/57/55 of 66 at 1,200 words; 240 stores 4,598 vectors instead of 8,658 or 19,065 and a 13 MB graph instead of 17 or 29 (section 4) |
| Hatnotes | "Main article: ...", "See also: ...", "For other uses ..." are stored as sentences but kept out of chunks | 1,316 of 33,637 units; before the change they formed short chunks that ranked high under section filters (found in the phase-4 review; not recorded in `results/`) |
| Index `M` | 16 (server default 6) | on the same vectors at `ef_search=20`: article recall@10 69/72/74 of 75 and evidence recall@20 55/61/63 of 66 for M 6/16/32, against 74 and 63 exact; at `ef_search=100` M=16 matches the exact hit list for 71 of 75 claims; build 2.9 s (M=6 1.1 s, M=32 9.0 s), graph 17 MB (section 3) |
| `mhnsw_ef_search` | 100 per query (server default 20) | on the final ingest hit lists identical to the exact ranking for 74 of 75 claims against 64 at 20, for 0.90 ms instead of 0.69 ms at k=10 (section 7) |
| Strategy with filters | `inline`; `overfetch` when latency matters more than a full result | inline always fills `k` at 2.7 to 3.6 ms; overfetch 10 costs 1.0 ms but comes back short under selective filters (section 6) |
| Index cache | 512 MB (`--mhnsw-max-cache-size`) | capacity only: the size changed no hit list; the graph of the 120-word ingest is 16 MB with `M=6` and 17 MB with `M=16`, at or above the default limit |

## Evaluation

**Method.** Each of the 75 claims is embedded once and searched once per `k` with `LIMIT k`; those
hits give the metrics at `k`, and the same statement is the one timed. (The committed `results/`
predate this rule and took the first `k` hits of `LIMIT 20`; that differs only for `overfetch` and
`rrf`, whose four affected files are listed in docs/DESIGN.md, "Review pass".) *Article recall@k*: a gold page of the claim is among the pages of the
top-k chunks (all 75 claims cite one page). *Evidence recall@k*: every sentence or list item of at
least one gold evidence set is covered by the top-k chunks, through `chunk_sentence`, over the 66
claims whose gold set is made of text units. *Unit coverage@k*: the share of gold units covered.
Latency is the SQL round trip of one `search()` call, warm, 5 repeats per claim (375 samples per
`k`), reported as p50 and p95; the query embedding is timed separately. Machine: i9-13900H, 15 GB,
RTX 4060 Laptop GPU, Ubuntu 24.04, MariaDB 11.8.9 in Docker.

**Headline** (`results/final_ef_100.md`: 240-word chunks, M=16, `ef_search=100`, strategy `none`):

| k | article recall | evidence recall | unit coverage | SQL p50 | SQL p95 |
|---|---|---|---|---|---|
| 1 | 63/75 (0.84) | 39/66 (0.59) | 0.60 | 0.83 ms | 1.27 ms |
| 3 | 74/75 (0.99) | 52/66 (0.79) | 0.81 | 0.81 ms | 1.23 ms |
| 5 | 74/75 (0.99) | 55/66 (0.83) | 0.85 | 0.82 ms | 1.27 ms |
| 10 | 74/75 (0.99) | 60/66 (0.91) | 0.92 | 0.90 ms | 1.41 ms |
| 20 | 74/75 (0.99) | 65/66 (0.99) | 0.99 | 1.05 ms | 1.59 ms |

Query embedding: 4.7 ms p50 on the GPU.

**The same ingest, other statements** (article recall@10, evidence recall@20, SQL p50/p95 at k=10):

| run | statement | article@10 | evidence@20 | SQL p50 / p95 |
|---|---|---|---|---|
| `final_inline` | exact ranking (joined statement, no filter) | 74/75 | 65/66 | 2.63 / 2.87 ms |
| `final_ef_100` | index, ef_search 100 | 74/75 | 65/66 | 0.90 / 1.41 ms, hit lists = exact for 74/75 |
| `final_ef_20` | index, ef_search 20 | 74/75 | 65/66 | 0.69 / 0.87 ms, hit lists = exact for 64/75 |
| `final_rrf` | vector + full text, RRF | 74/75 | 64/66 | 10.84 / 13.94 ms |
| `final_oracle_titles` | inline, gold page as `titles` filter | 75/75 | 66/66 | 2.65 / 9.87 ms; evidence in the first chunk for 42/66, within 5 for 59/66 |
| `final_ef_100_after_restart` | index, ef_search 100, after `docker compose restart` | 74/75 | 65/66 | hit lists identical to before for 74/75 |

**Reading the numbers.** The one claim whose gold page never appears (claim 87976, a NOT ENOUGH
INFO claim about a by-election whose gold page is "Lincoln, England") is missed by the embedding
itself: the exact ranking misses it too (`results/final_ef_100.md`, worst-cases table). The hybrid
does not beat the vector ranking on these claims and costs ten times more. The oracle row is the
bound on what better chunk ranking could give: with the page known, 42 of 66 gold sets are already
in the first chunk. Compared with the server defaults on the 120-word ingest of the same corpus (`results/ef_20.md`,
8,658 chunks), `M=6` and `ef_search=20` found 68 of 75 gold pages at k=10 where the exact ranking of
those vectors finds 74; on the 240-word ingest the chosen settings find 74, which is what the exact
ranking finds there too.

**Reproduce.** `wikilense eval --name mine` after `wikilense ingest` writes `results/mine.json` and
`.md` with the headline table for the current ingest; the counts match the table above up to the
HNSW build (a fresh graph differs in a few chunk lists, not in recall). `scripts/run_experiments.py
all` reruns every group in `results/SUMMARY.md` (about 10 minutes with a GPU): stability across cache
sizes and a restart, the `ef_search` sweep, the `M` sweep with rebuild times, chunk sizes at equal
retrieved text, the prefix, the filter strategies, and the final configuration. Every results file
records the ingest parameters, the effective `ef_search`, the cache size, the index `M` and the
chunk count it was measured on.

## What the database-native approach gave

- **Predicates, joins and the vector search in one statement.** "Chunks about X, on pages longer
  than N words, under a History heading, on pages linked from Y" is one SQL statement that the
  optimizer plans, with the vector index as its first access path, and `EXPLAIN` shows it. A
  separate vector store answers the vector part and leaves the rest to application code with
  post-filtering, which is exactly the `overfetch` strategy we measured: cheaper, but short of
  `k` rows under selective filters.
- **The ground truth is a join.** Evidence recall is `chunk_sentence JOIN claim_evidence`, so the
  evaluation is SQL over the same tables, not a second pipeline that re-matches text.
- **One transaction, one backup, one set of foreign keys.** Vectors are rows: they roll back with
  their transaction, disappear with their page, and are covered by the same dump. There is no
  synchronisation job between a store and the metadata it describes.
- **Two rankings, one engine.** The `FULLTEXT` index and the vector index live on the same table and
  are fused with window functions and CTEs, without a second system.
- **Honest costs.** The joined form is exact and about three times the bare index query on this
  corpus, and it grows with the table; the bare form is approximate and needs `M` and `ef_search`
  tuned, which we did with measurements rather than adjectives.

## Limitations and next phase

- Tables and captions are not indexed: 27 of the 114 evidence ids point at table cells and cannot
  be matched. Lists are.
- 100 pages is small enough that the exact ranking costs 2.6 ms; the number to watch when the
  corpus is scaled is the `inline` cost, which grows with the table, against the bare index query,
  which does not.
- The corpus holds only claims that cite one page (the shard-based selection rule); multi-page
  claims are the next phase, together with the full FEVEROUS claim set and the LLM fact-checker
  comparison (no retrieval, retrieved chunks, gold evidence).
- HNSW results are approximate by construction; the README states which numbers are exact.
- MariaDB 11.8 has one vector index per table and no `INFORMATION_SCHEMA` view for it; the index
  size was read from the hidden tablespace (`chunk#i#NN`) as root.
- Generation is out of scope: the area's project text asks for retrieval, and the ground truth
  measures retrieval. The LLM fact-checker comparison (no retrieval, retrieved chunks, gold
  evidence) is the planned extension.

## Repository

```
wikilense/          the package: corpus, wikitext (parsing), chunking, embedding, db, ingest,
                    search, evaluate, cli, web, config
sql/schema.sql      the data model; sql/docker-init/ creates the test database on first start
scripts/            build_corpus.py, fetch_wiki_shard.py, run_experiments.py
data/corpus/        the committed 100 pages and 75 claims, with MANIFEST.md
results/            every experiment (JSON + Markdown) and SUMMARY.md
docs/DESIGN.md      the design contract, every deviation forced by the data, the measured outcomes
tests/              255 tests: pure tests, `db` tests against wikilense_test, `slow` tests that load the model
analysis/feverous/  the FEVEROUS analysis that chose the benchmark (earlier phase)
```

`make test` runs the suite (20 s; `pytest -m "not slow"` skips the model tests; `db` tests skip
when the server is unreachable), `make lint` runs ruff. [CI](.github/workflows/ci.yml) runs both on
every push to `main` and on every pull request against a `mariadb:11.8` service container with the CPU build of torch, installing from
`requirements-cpu.lock`.

## Team

Ylli Rada, Constructor University, Databases and Web Services course. Code, experiments and
documentation were produced with Claude Code (Anthropic) directed by the author; the decisions,
their reasons and every deviation the data forced are recorded in [docs/DESIGN.md](docs/DESIGN.md)
and [CLAUDE.md](CLAUDE.md). Licence: MIT ([LICENSE](LICENSE)).
