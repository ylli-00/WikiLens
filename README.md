A project for area 4, "Vector search and RAG, natively" for the [MariaDB student database projects, 2026-09](https://mariadb.org/bachelor_hackathon_2026-09/).

# WikiLense

Semantic search over Wikipedia inside MariaDB. Wikipedia pages are cut into chunks, each chunk is
embedded with a small open model and stored in a `VECTOR(384)` column with a cosine `VECTOR INDEX`,
and a query is one SQL statement: `ORDER BY VEC_DISTANCE_COSINE(...)` combined with ordinary
predicates and joins back to the relational tables. The predicates are the four the brief names:
**only articles in this category, longer than this, edited since that date, linked from that
other article**. Fact-checking claims from the FEVEROUS benchmark are the queries and their gold
evidence is the ground truth, so "it works well" is a recall number and a latency number.

> **The ground-truth set.** The brief asks for "a small ground-truth set — questions whose correct
> source article you already know". Ours is the 75 FEVEROUS claims whose gold evidence lies inside
> our 100 pages, selected by a fixed rule ([The data](#the-data)). Each claim is a statement to
> verify rather than a question, and for each one, annotators who searched all of Wikipedia marked
> the exact sentences that prove or refute it. So we know the correct source article *and* the
> correct sentences, and can measure evidence recall as well as article recall. We chose this over
> writing questions ourselves for two reasons. Questions written while reading an article reuse
> its words, which flatters recall. And the claims' labels (supported, refuted, not enough
> information) serve the planned LLM fact-checking extension. It is a development-sized set, and
> claims that cite several pages are the next phase ([Limitations](#limitations-and-next-phase)).

Developed against **MariaDB 11.8.9** (Docker image `mariadb:11.8`), Python 3.12, PyMySQL 1.2.3,
sentence-transformers 6.0 with `BAAI/bge-small-en-v1.5`. Every number in this README is measured on
the committed corpus (100 pages, 75 claims) and read from [results/SUMMARY.md](results/SUMMARY.md),
which [scripts/run_experiments.py](scripts/run_experiments.py) writes. The headline and the `final_*`
runs use the 240-word ingest (4,598 chunks). The `M`, `ef_search`, stability and prefix sweeps use
the 120-word ingest (8,658 chunks), and the chunk-size sweep uses one ingest per size.

**Contents:** [What we found](#what-we-found) · [Quick start](#quick-start) · [Try it](#try-it) ·
[The data](#the-data) · [The data model](#the-data-model) · [The SQL](#the-sql) ·
[How MariaDB uses the vector index](#how-mariadb-1189-uses-the-vector-index-measured) ·
[Parameters and why](#parameters-and-why) · [Evaluation](#evaluation) ·
[What the database-native approach gave](#what-the-database-native-approach-gave) ·
[Limitations](#limitations-and-next-phase) · [Repository](#repository) · [Team](#team)

## What we found

- **Retrieval works and is fast.**
  - With the chosen settings, the gold page of a claim is among the pages of the top 5 chunks for
    74 of 75 claims.
  - The complete gold evidence is inside the top 10 chunks for 60 of 66 claims, and inside the top
    20 for 65 of 66.
  - The vector query takes 0.51 ms at the median and 0.94 ms at the 95th percentile for k=10.
    Embedding the query on the GPU takes 4.6 ms.
- **The brief's four predicates run inside the vector search, in one statement.** "Chunks about X,
  only in pages of the category *Member states of the United Nations*, longer than 1,000 words,
  edited since 1 December 2020, linked from *Asia*" is one SQL statement.
  - `EXPLAIN` shows the vector index as its first access path, and the category and link filters
    as materialised derived tables.
  - Over the 75 claims it runs in 3.5 ms at the median and never returns fewer than k rows.
- **The optimizer decides, per filter, whether the vector index survives.**
  - Nine of the eleven filters we measured keep it, as does the unfiltered statement. The rows
    read grow with the filter's selectivity: 10 rows unfiltered, 601 when only 89 of 4,598 chunks
    pass.
  - The two most selective range filters, 25 and 4 passing chunks, make MariaDB scan all 4,598
    chunks and sort instead. The result is exact, but the cost grows with the table.
  - A category filter keeps the index only when its derived table is `SELECT DISTINCT`: without
    it, MariaDB merges the table into the join and scans in full.
- **Searching rows your own transaction has not committed finds fewer true neighbours.**
  - On our chunk embeddings at `mhnsw_ef_search=100`, the k-NN search over rows inserted in the
    open transaction finds 0.971 of the exact top 10, against 1.000 after `COMMIT`.
  - On random vectors it is 0.651 against 0.985.
  - A larger `mhnsw_ef_search` does not close the gap.
  - `ROLLBACK` removes every row and index entry.
- **Tuning the index matters more than the hardware.**
  - On the 120-word ingest, the server defaults (`M=6`, `mhnsw_ef_search=20`) find the gold page
    for 68 of 75 claims at k=10, where the exact ranking of the same vectors finds 74.
  - With `M=16`, the same vectors give 74 already at `ef_search=20`.
  - On the final ingest, `ef_search=100` returns the exact ranking's hit list for 74 of 75 claims,
    at 0.51 ms against 2.24 ms for the exact statement.
- **The remaining misses are inside the right page.** With the gold page handed to the query as a
  filter, the gold sentences are in the first chunk for 42 of 66 claims and within 20 chunks for
  all 66. Ranking chunks within a page, not finding the page, is where recall is lost.
- **Results are reproducible up to the graph build.**
  - Every HNSW build is a different approximation: two `M=6` builds of the same vectors agreed on
    12 of 75 hit lists.
  - Within one server process the results are deterministic, and the index cache size changed
    none of them.
  - A container restart changed 1 of 75 hit lists on the `M=6` index and none on `M=16` at
    `ef_search=100`.

## Quick start

```
git clone https://github.com/ylli-00/WikiLens.git && cd WikiLens
make setup
```

`make setup` does four things, which you can also run by hand:

1. `cp .env.example .env`, then it stops so you can set the two passwords in `.env` (the MariaDB
   root password and the application user's password). Run `make setup` again afterwards.
2. `docker compose up -d --wait`: MariaDB 11.8 with `--mhnsw-max-cache-size=536870912` (512 MB),
   bound to `127.0.0.1:3306`, with the databases `wikilense` (the corpus) and `wikilense_test`
   (the test suite). The healthcheck waits until InnoDB is ready.
3. `python3 -m venv .venv` and `.venv/bin/pip install -r requirements-dev.txt -e .`.
   - `make` creates the venv before starting the container.
   - The package reads `sql/schema.sql` and `data/corpus/` from the checkout, so the editable
     install is required.
   - On a machine without a GPU, run `make setup CPU=1` instead. It installs the pinned CPU build
     from `requirements-cpu.lock`, as CI does.
4. `.venv/bin/wikilense ingest`: creates the tables, parses the 100 committed pages, loads their
   categories and edit dates, embeds the chunks and loads everything.
   - The embedding model, about 134 MB, downloads to `~/.cache/huggingface` on first use.
   - The run takes 21 s on an RTX 4060 laptop GPU, of which 17 s is embedding.
   - On a CPU the embedding runs at about 78 passages per second, so expect about a minute.

The ingest ends with the seconds per stage and a report you can compare against this one (the
1,316 hatnote units kept out of chunks are recorded in `ingest_meta`):

```
  pages 100, sections 3173, sentences 33637 (empty 372), chunks 4598
  links 40542 (resolved 373, skipped 0)
  claims 75, evidence ids 114 (page resolved 114, sentence resolved 87)
  page metadata 100 of 100 pages, categories 1005 (page_category 1123)
```

Then:
- `make test`: 345 tests, about 20 s.
- `make eval`: the headline table below, about 15 s.
- `make serve`: the web page at http://127.0.0.1:8000.
- `make experiments`: the whole protocol, about 10 minutes.
- `make down`: stops MariaDB.

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
 3. 0.3620  Aare > Course  [chunk_id 3, 228 words]
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
3 hits; model load 4316 ms, embedding 5.8 ms, SQL 4.5 ms (BAAI/bge-small-en-v1.5 on cuda, strategy inline, ef_search 100)
```

The vector index is the `key` of the first row; `page` and `section` are joined by primary key.
- **Model load:** it happens once per process; the web server pays it once at start-up.
- **SQL time:** the CLI's figure is one sample and includes the `mhnsw_ef_search` session round
  trips. The harness numbers under [Evaluation](#evaluation) exclude them.

**The brief's four predicates in one statement.** This query asks for pages in a category, longer
than 1,000 words, edited since a date and linked from another page:

```
$ wikilense query "independence from the Soviet Union" --k 3 \
    --category "Member states of the United Nations" --min-words 1000 \
    --edited-since 2020-12-01 --linked-from Asia --explain
 1. 0.2522  Azerbaijan > History > Independence  [chunk_id 1145, 224 words]
    Following the politics of glasnost, initiated by Mikhail Gorbachev, civil unrest and ethnic
    strife grew in various regions of the Soviet Union, including Nagorno-Karabakh, ...
 2. 0.3012  Azerbaijan > History > Contemporary history  [chunk_id 1143, 226 words]
 3. 0.3087  Kazakhstan  [chunk_id 3693, 220 words]
SQL:
    ... FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id STRAIGHT_JOIN section ON
    section.section_id = chunk.section_id STRAIGHT_JOIN (SELECT DISTINCT link.to_page_id AS page_id
    FROM link JOIN page AS src ON src.page_id = link.from_page_id WHERE src.title = %s AND
    link.to_page_id IS NOT NULL) AS linked ON linked.page_id = chunk.page_id STRAIGHT_JOIN (SELECT
    DISTINCT page_category.page_id FROM page_category JOIN category ON category.category_id =
    page_category.category_id WHERE category.name = %s) AS in_category ON in_category.page_id =
    chunk.page_id WHERE page.n_words >= %s AND page.last_edited >= %s ORDER BY
    VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s
parameters: <vector[384]>, 'Asia', 'Member states of the United Nations', 1000, datetime.datetime(2020, 12, 1, 0, 0), <vector[384]>, 3
EXPLAIN:
    id  select_type  table          type    key                        rows  Extra
    1   PRIMARY      chunk          index   embedding                  3
    1   PRIMARY      page           eq_ref  PRIMARY                    1     Using where
    1   PRIMARY      section        eq_ref  PRIMARY                    1
    1   PRIMARY      <derived2>     ref     key0                       1
    1   PRIMARY      <derived3>     ref     key0                       1
    3   DERIVED      category       const   uq_category_name           1     Using index
    3   DERIVED      page_category  ref     ix_page_category_category  10    Using where; Using index
    2   DERIVED      src            const   uq_page_title              1     Using index
    2   DERIVED      link           range   ix_link_to_page            373   Using index condition; ...
3 hits; model load 4679 ms, embedding 5.1 ms, SQL 4.3 ms (BAAI/bge-small-en-v1.5 on cuda, strategy inline, ef_search 100)
```

(The EXPLAIN is trimmed to the columns that matter here.) How each part of the plan works:
- **The vector index drives the scan.** `chunk` is read through `embedding`, ahead of everything
  else.
- **Length and edit date** are checked on the `page` row (`Using where`).
- **The link and category filters** are two small materialised tables (`<derived2>`,
  `<derived3>`), each looked up by `page_id`. The category name resolves through its unique index
  (`const`), and the pages of that category through `ix_page_category_category`.

**A predicate on the section heading** and **one on the link graph**:

```
$ wikilense query "Roman settlement and medieval castle" --k 3 --heading '%History%'
 1. 0.3490  Lincoln, England > History > Roman history: Lindum Colonia  [chunk_id 3306, 208 words]
 2. 0.3500  Andorra > History > Prehistory  [chunk_id 388, 209 words]

$ wikilense query "football club league titles" --k 3 --linked-from "AFC Ajax"
 1. 0.2935  Juventus F.C. > Honours  [chunk_id 4074, 214 words]
```

**The hybrid strategy** fuses the vector ranking with a `FULLTEXT` `MATCH ... AGAINST` ranking by
reciprocal rank fusion, in one statement with three CTEs (the vector list, the full-text list and
their fusion):

```
$ wikilense query "Coppa Italia finals won" --k 3 --strategy rrf
 1. 0.2217  Juventus F.C. > History > Historic four consecutive doubles ... [chunk_id 4054, 236 words, rrf score 0.0323]
```

**The join back to the sentences** a chunk was built from (`--sentences`) is how the evaluation
matches chunks to FEVEROUS evidence ids:

```
$ wikilense query "Aare hydroelectric plants" --k 1 --sentences
 1. 0.2491  Aare  [chunk_id 1, 127 words]
    sentences (chunk_sentence -> sentence, page order):
        sentence_2  The Aare (German: [ˈaːrə]) or Aar (German: [aːɐ̯] (listen)) is a tributary of ...
        sentence_3  Its total length from its source to its junction with the Rhine comprises about ...
```

All options:
- **Filters:**
  - `--category NAME` (exact name, no "Category:");
  - `--edited-since` / `--edited-before YYYY-MM-DD` (UTC);
  - `--min-words` / `--max-words` (page length);
  - `--linked-from` / `--links-to TITLE`;
  - `--heading` / `--path` (SQL `LIKE` patterns);
  - `--title TITLE` (repeatable, exact titles).
- **Search:** `--strategy inline|overfetch|none|rrf`, `--overfetch N`, and `--ef-search N` (0 keeps
  the server value).
- **Output:** `--json` for scripts.

`wikilense serve` starts a web page with the same query and filters.
- **The API behind it:** `GET /api/search?q=...&k=5&category=...&edited_since=2020-12-01&sentences=1`
  returns `hits`, the `sql` that ran, its `explain` rows, the effective `ef_search` and `timing_ms`.
- **Swagger:** `/docs` is enabled, so the API can be tried from the browser.

[sql/examples.sql](sql/examples.sql) has ten worked statements for the `mariadb` client alone (no
Python, no model). They include the four-filter statement with its `EXPLAIN`, a page's categories
and edit date, and a transaction that shows the vector index following `ROLLBACK`. A test runs
every statement of the file against the schema.

## The data

- **`data/corpus/pages.jsonl`** (8.0 MB): 100 Wikipedia pages in the FEVEROUS format: December 2020
  articles with sentence, section, table and list elements.
- **`data/corpus/claims.jsonl`** (141 KB): 75 FEVEROUS claims with their evidence sets. It and
  `pages.jsonl` are committed, with SHA-256 checksums and the page list in
  [data/corpus/MANIFEST.md](data/corpus/MANIFEST.md).
- **The selection rule.**
  - The 75 claims (65 train, 10 dev; 7 SUPPORTS, 58 REFUTES, 10 NOT ENOUGH INFO) are all claims of
    FEVEROUS train and dev whose evidence pages are all inside the first Wikipedia shard,
    `wiki_000.jsonl`. Their evidence cites 54 distinct pages.
  - The other 46 pages are the pages those 54 link to most often, so the corpus contains realistic
    distractors and the link filters have something to work on.
  - 65 claims have an evidence set made only of sentences. 66 count for evidence recall, because one
    more claim's set mixes sentences and list items, which are both text units.
  - The 27 evidence ids that point at table cells cannot be matched, because tables are not
    indexed.
- **`data/corpus/page_meta.jsonl`**: each page's Wikipedia revision id, edit date and categories.
  FEVEROUS keeps none of them; its pages stop before the category box and the "last edited" footer.
  - [scripts/fetch_page_meta.py](scripts/fetch_page_meta.py) asks the MediaWiki API for the
    revision whose text the corpus holds. That is the one live in December 2020, when FEVEROUS
    scraped Wikipedia, not today's page. Its timestamp is the "last edited" date of the text we
    have, and its `[[Category:...]]` links are the categories.
  - For the 93 pages edited during December 2020, the revision whose wikitext contains the most of
    the page's FEVEROUS sentences is chosen. Three renamed pages are followed through the move log.
  - All 100 pages were found, and 94.5% of the sentence probes occur in the chosen revisions. On
    today's "Albert Einstein" page only 260 of 484 probes match, against 466 in the December 2020
    revision.
  - The result: 1,005 categories, 1,123 page-category pairs, edit dates from 2019-12-23 to
    2020-12-31. [data/corpus/PAGE_META.md](data/corpus/PAGE_META.md) has the summary.
  - The script follows the Wikimedia Robot Policy: one request at a time, at most four per second,
    a User-Agent with this repository's URL, `maxlag`, `Retry-After`. The 100 pages took 309
    requests.
- **Rebuilding it** (about 5 s, byte-identical):
  1. Download the shard with `python3 scripts/fetch_wiki_shard.py wiki_000.jsonl`. It is 44 MB,
     fetched from inside the 9.9 GB archive with HTTP range requests and CRC-checked.
  2. Put `feverous_train_challenges.jsonl` and `feverous_dev_challenges.jsonl` from
     https://fever.ai/dataset/feverous.html into `data/feverous/`.
  3. Run `python3 scripts/build_corpus.py`.
  4. Run `python3 scripts/fetch_page_meta.py` to refetch the metadata (a few minutes).
- FEVEROUS: Aly et al., 2021 (https://arxiv.org/abs/2106.05707); Wikipedia text under CC BY-SA 3.0.

## The data model

Eleven InnoDB tables, all in [sql/schema.sql](sql/schema.sql), which the code executes as it is.
`wikilense.db.SCHEMA_COLUMNS` mirrors the file, and a test compares both against the live server.

| Table | Holds | Why it is shaped this way |
|---|---|---|
| `page` | one row per article: title, counts of sentences, items, words, characters, sections, tables, lists; `revision_id` and `last_edited` of the December 2020 revision | `n_words` and `last_edited` are indexed, so "longer than N" and "edited since D" are range predicates; `title` is `utf8mb4_bin`, so matching is exact |
| `section` | the heading tree: ordinal, heading, level, path such as `History > Roman history` | the lead is section 0; `heading` keeps the server's accent- and case-insensitive collation on purpose, so `LIKE '%history%'` matches `History` |
| `sentence` | one row per text unit (a sentence or a list item), with its FEVEROUS element key | this is the unit of gold evidence; every unit is stored, even empty ones and hatnotes, so evidence ids always resolve |
| `chunk` | the retrieval unit: text, word count, `embedding VECTOR(384) NOT NULL`, the `VECTOR INDEX` and a `FULLTEXT` index | see the DDL below |
| `chunk_sentence` | which sentences each chunk covers (33,783 rows) | the join that turns "chunk in the top k" into "gold sentence in the top k" |
| `link` | every `[[link]]` in a page, with the target resolved to a `page_id` when it is in the corpus | "pages linked from X" and "pages linking to X" become joins |
| `category` | one row per Wikipedia category name (1,005) | `name` is unique and `utf8mb4_bin`: a category is matched exactly, like a title |
| `page_category` | which categories each page is in (1,123 rows) | a many-to-many table: the primary key answers "categories of P", the index (`category_id`, `page_id`) answers "pages in C", which the category filter joins |
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

How the references behave:
- **Ownership references cascade.** Deleting a page deletes its sections, sentences, chunks,
  vectors and category memberships in one statement; a test checks it.
- **Resolution references are `ON DELETE SET NULL`** (`link.to_page_id`, `claim_evidence.page_id`,
  `claim_evidence.sentence_id`). A link or a piece of gold evidence must survive the page it points
  to being re-ingested.
- **The vector dimension** is literal in the DDL and checked against the model at ingest.

## The SQL

Every statement is a named constant in [wikilense/search.py](wikilense/search.py), assembled from
fixed fragments. Values are always bound as parameters, and a vector parameter is bound as the
raw float32 bytes (`db.vec_param`), sent as a `_binary X'...'` literal.

The bare index query is the HNSW search proper. Strategies `none` and `overfetch` run it as a
derived table joined back to `page` and `section` for the hit metadata (`none` is `overfetch` with
factor 1 and no filters):

```sql
SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, ?) AS distance
FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT ?
```

Strategy `inline` puts the predicates and joins in the same statement. It is the default of
`search()` and of the CLI, with or without filters. A fragment is added only for a filter that is
given:

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
  [STRAIGHT_JOIN (SELECT DISTINCT page_category.page_id FROM page_category
                  JOIN category ON category.category_id = page_category.category_id
                  WHERE category.name = ?) AS in_category
                 ON in_category.page_id = chunk.page_id]
WHERE page.n_words >= ? AND page.last_edited >= ? AND section.heading LIKE ? ...
ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, ?) LIMIT ?
```

In the bounded form (strategy `overfetch`), the index returns `k * overfetch` candidates in a
derived table, and the outer query filters, joins and re-limits:

```sql
SELECT ... , knn.distance, chunk.text
FROM (SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, ?) AS distance
      FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT ?) AS knn
JOIN chunk ON chunk.chunk_id = knn.chunk_id
JOIN page ON page.page_id = chunk.page_id
JOIN section ON section.section_id = chunk.section_id
WHERE ... ORDER BY knn.distance, chunk.chunk_id LIMIT ?
```

The hybrid (strategy `rrf`) takes the vector top-N and the full-text top-N, ranks each with
`ROW_NUMBER()`, and fuses them by reciprocal rank fusion:

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

The join back to the evidence is `chunk_sentence JOIN sentence` for the hit chunks, ordered by
page position (`--sentences`).

## How MariaDB 11.8.9 uses the vector index, measured

Everything here was measured on this corpus with `EXPLAIN`, `ANALYZE` and the harness.
[tests/test_db.py](tests/test_db.py) and [tests/test_search.py](tests/test_search.py) assert the
facts that hold on synthetic data: `key: embedding`, `r_rows` = `k`, both indexes for `rrf`, the
session scope of `ef_search`, the transactions, and the plan of the category filter.

- **What drives the index.** Only `ORDER BY VEC_DISTANCE_COSINE(column, constant) LIMIT n`, sorted
  ascending; the alias form `ORDER BY distance` is identical. These all drop the index, so ties
  are broken in Python:
  - wrapping the distance in an expression;
  - a `WHERE VEC_DISTANCE(...) < t` without `ORDER BY ... LIMIT`;
  - a second `ORDER BY` key (`..., chunk_id`).
- **The arithmetic is right.** On 300 random, unnormalised vectors:
  - `VEC_DISTANCE_COSINE` equals NumPy's `1 - cos` for every row, within 1e-5.
  - A scan that ignores the index (`IGNORE INDEX (embedding)`) returns NumPy's exact top 10 in
    order.
  - The HNSW index at `ef_search=200` finds all of them.
- **Two cost regimes.**
  - **The bare query** (`none`, `overfetch`) is the HNSW search: 0.45 to 0.64 ms on 4,598 chunks.
    It reads exactly `LIMIT` rows, and `mhnsw_ef_search` sets its accuracy.
  - **With a join in the statement** (`inline`), `EXPLAIN` still says `type: index, key:
    embedding`, but the server ranks exactly over the index: 2.24 ms here. Its hit lists do not
    depend on `ef_search` (identical at 20 and 100 for 75 of 75 claims), and on the synthetic test
    corpus they equal the known exact order. It then walks until `k` rows pass the predicates.
- **Which filters keep the index** (`results/planner_choices.md`: every claim, the inline statement,
  k=10).
  - The optimizer chooses per filter, from its estimate of how many rows pass.
  - Nine of the eleven filters keep the vector index, as does the unfiltered statement. The rows
    read grow as the filter gets selective: 10 unfiltered, 108 under the category, 121 under the
    link filter, 315 under the History heading, 601 under `max_words 2000` (89 chunks pass).
  - The two most selective range filters on indexed `page` columns make it scan all 4,598 chunks,
    compute every distance and sort: `max_words 500` (25 chunks pass) and `edited_before
    2020-06-01` (4 pass). That plan is exact, at 5.3 and 3.3 ms, and its cost grows with the table.
  - No case returned fewer than `k` rows.

  ![Chunk rows read under each filter](docs/img/planner_rows.svg)

- **Bounded versus complete.** `overfetch` stays near the bare cost but returns fewer than `k` rows
  when the filter is selective. Under `heading LIKE '%History%'` (117 of 4,598 chunks pass), at
  k=10:

  | Strategy | Queries short of 10 rows (of 75) | Median time |
  |---|---|---|
  | `overfetch 10` | 69 (18 of them with no row) | 1.02 ms |
  | `overfetch 50` | 20 | 2.37 ms |
  | `inline` | 0 | 3.34 ms (4.74 ms p95) |
- **Join order matters.**
  - With a plain `JOIN`, the optimizer started from `page` on stale statistics right after a bulk
    insert and lost the vector index. The statements use `STRAIGHT_JOIN` with `chunk` first, and
    the ingest runs `ANALYZE TABLE`.
  - An `IN (subquery)` filter becomes a semi-join that starts from `link` and loses the index too.
  - A derived table without `DISTINCT` is merged into the outer join, and `chunk` is scanned in
    full. We found this with the category filter, even on the 12-chunk test data. A `SELECT
    DISTINCT` derived table stays materialised and keeps the index; a test pins that plan.
- **Searching rows the transaction has not committed** (`results/uncommitted_recall.md`).
  - The test: a scratch table with the chunk table's vector column and index is filled inside one
    transaction and searched before and after `COMMIT`. There are three fresh builds each, and the
    overlap is measured against NumPy's exact top 10.
  - With our 4,598 chunk embeddings and the 75 claims, the search finds 0.968 to 0.971 of the
    exact top 10 before `COMMIT` at `mhnsw_ef_search` 20, 100 and 400, and 0.987 to 1.000 after.
  - At ef 100 the worst single query finds 5 of its 10 neighbours before `COMMIT` and 9 after.
  - On 1,000 random vectors it is 0.620 to 0.651 before and 0.782 to 0.993 after.
  - The uncommitted search is also faster: 0.13 to 0.24 ms p50 against 0.59 to 0.91 ms on the
    chunks. That suggests it explores less of the graph; we did not verify this in the server.
  - `ROLLBACK` leaves no row behind. Write and search in separate transactions, or commit first.

  ![Overlap with the exact top 10 before and after COMMIT](docs/img/uncommitted_recall.svg)

- **Parameters.**
  - `mhnsw_ef_search` is a session variable, set per query and restored after. Values outside its
    range are clamped with a warning, not an error.
  - `mhnsw_max_cache_size` is global. The 16 MB default is already the size of the `M=6` graph of
    the 8,658-chunk 120-word ingest (17 MB with `M=16`), so the Docker configuration starts the
    server with 512 MB.
  - The graph lives in a hidden InnoDB table (`chunk#i#NN`), not in `SHOW TABLE STATUS`'s
    `Index_length`.
- **Binding.**
  - Raw float32 bytes bind as a `_binary` literal, both in `INSERT` and inside
    `VEC_DISTANCE_COSINE`.
  - PyMySQL 1.2.3 and later send `_binary X'<hex>'` by themselves. Versions 1.2.0 to 1.2.2 do so
    only on a connection opened with `binary_prefix=True`, which `db.connect` sets for every
    version.
  - `VEC_FromText(?)` with a JSON list and `UNHEX(?)` work everywhere.
  - A query vector of the wrong dimension returns `NULL` distances, silently.
- **Transactions.**
  - A chunk inserted inside a transaction is visible to the index in the same connection, and gone
    after `ROLLBACK`.
  - Another connection sees it only after `COMMIT`.
  - `UPDATE` of an embedding changes the distance at once, `DELETE` removes it from the results,
    and deleting a page cascades to its vectors.
  - There is no synchronisation job and no second store. `sql/examples.sql` (example 8) shows it
    from the `mariadb` client.
- **Approximation and state.**
  - Every build of the HNSW graph is a different approximation: two `M=6` builds of the same
    vectors agreed on 12 of 75 hit lists.
  - Within one server process the results are deterministic, and the cache size does not change
    them (75 of 75 identical at 16 MB and 512 MB).
  - A `docker compose restart` changed 1 of 75 hit lists on the `M=6` index and none on `M=16` at
    `ef_search=100`, without changing recall.

## Parameters and why

| Parameter | Chosen | Evidence (details in results/SUMMARY.md) |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5`, 384 dimensions, 33M parameters, MIT licence | chosen in phase 1 for passage retrieval at this size; no second model was measured. 1,059 passages/s on the RTX 4060 and 78/s on the CPU for 100-word texts (phase-1 measurement in docs/DESIGN.md, not part of `results/`), so a grader without a GPU can ingest; normalised vectors, so cosine distance is `1 - dot` |
| Embedded text | `title > section path: chunk text` | on the exact ranking the prefix raises evidence recall@20 from 61 to 63 of 66 and leaves article recall@10 at 74 of 75 (section 5) |
| Chunk size | 240 words, overlap 1 unit, never across a section boundary | at equal retrieved text the 60-, 120- and 240-word chunkings tie: article recall 74/74/74 of 75 and evidence recall 54/57/55 of 66 at 1,200 words; 240 stores 4,598 vectors instead of 8,658 or 19,065 and a 13 MB graph instead of 17 or 29 (section 4) |
| Hatnotes | "Main article: ...", "See also: ...", "For other uses ..." are stored as sentences but kept out of chunks | 1,316 of 33,637 units; before the change they formed short chunks that ranked high under section filters (found in the phase-4 review; not recorded in `results/`) |
| Index `M` | 16 (server default 6) | on the same vectors at `ef_search=20`: article recall@10 67/74/73 of 75 and evidence recall@20 54/64/62 of 66 for M 6/16/32, against 74 and 63 exact; at `ef_search=100` M=16 returns the exact hit list for 69 of 75 claims (M=32: 73); build 2.9 s (M=6 1.1 s, M=32 9.1 s), graph 17 MB (section 3) |
| `mhnsw_ef_search` | 100 per query (server default 20) | on the final ingest hit lists identical to the exact ranking for 74 of 75 claims against 65 at 20, for 0.51 ms instead of 0.45 ms at k=10 (section 7) |
| Strategy with filters | `inline`; `overfetch` when latency matters more than a full result | inline always fills `k`, at 2.2 to 3.6 ms p50 while it walks the index and 5.3 ms when the optimizer scans instead; overfetch 10 costs 0.7 to 1.0 ms but comes back short under selective filters (sections 6 and 10) |
| Index cache | 512 MB (`--mhnsw-max-cache-size`) | capacity only: the size changed no hit list; the graph of the 120-word ingest is 16 MB with `M=6` and 17 MB with `M=16`, at or above the default limit |

## Evaluation

**Method.**
- **The queries.** Each of the 75 claims is embedded once and searched once per `k` with
  `LIMIT k`. Those hits give the metrics at `k`, and the same statement is the one timed.
- **Article recall@k:** a gold page of the claim is among the pages of the top-k chunks. All 75
  claims cite one page.
- **Evidence recall@k:** every sentence or list item of at least one gold evidence set is covered
  by the top-k chunks, through `chunk_sentence`. It is counted over the 66 claims whose gold set is
  made of text units.
- **Unit coverage@k:** the share of gold units covered.
- **Latency:** the SQL round trip of one `search()` call, warm, with 5 repeats per claim (375
  samples per `k`), reported as p50 and p95. The query embedding is timed separately.
- **Machine:** i9-13900H, 15 GB, RTX 4060 Laptop GPU, Ubuntu 24.04, MariaDB 11.8.9 in Docker.

**Headline** (`results/final_ef_100.md`: 240-word chunks, M=16, `ef_search=100`, strategy `none`):

| k | article recall | evidence recall | unit coverage | SQL p50 | SQL p95 |
|---|---|---|---|---|---|
| 1 | 63/75 (0.84) | 39/66 (0.59) | 0.60 | 0.50 ms | 0.86 ms |
| 3 | 74/75 (0.99) | 52/66 (0.79) | 0.81 | 0.45 ms | 0.78 ms |
| 5 | 74/75 (0.99) | 55/66 (0.83) | 0.85 | 0.45 ms | 0.79 ms |
| 10 | 74/75 (0.99) | 60/66 (0.91) | 0.92 | 0.51 ms | 0.94 ms |
| 20 | 74/75 (0.99) | 65/66 (0.99) | 0.99 | 0.64 ms | 1.10 ms |

Query embedding: 4.6 ms p50 on the GPU.

![Article and evidence recall at k](docs/img/recall_at_k.svg)

**The same ingest, other statements** (article recall@10, evidence recall@20, SQL p50/p95 at k=10):

| run | statement | article@10 | evidence@20 | SQL p50 / p95 |
|---|---|---|---|---|
| `final_inline` | exact ranking (joined statement, no filter) | 74/75 | 65/66 | 2.24 / 2.57 ms |
| `final_ef_100` | index, ef_search 100 | 74/75 | 65/66 | 0.51 / 0.94 ms, hit lists = exact for 74/75 |
| `final_ef_20` | index, ef_search 20 | 74/75 | 65/66 | 0.45 / 0.70 ms, hit lists = exact for 65/75 |
| `final_rrf` | vector + full text, RRF | 74/75 | 64/66 | 9.11 / 12.84 ms |
| `final_oracle_titles` | inline, gold page as `titles` filter | 75/75 | 66/66 | 2.45 / 8.59 ms; evidence in the first chunk for 42/66, within 5 for 59/66 |
| `final_ef_100_after_restart` | index, ef_search 100, after `docker compose restart` | 74/75 | 65/66 | hit lists identical to before for 75/75 |

**Reading the numbers.**
- **The one claim whose gold page never appears** is claim 87976, a NOT ENOUGH INFO claim about a
  by-election, whose gold page is "Lincoln, England". The embedding itself misses it: the exact
  ranking misses it too (`results/final_ef_100.md`, worst-cases table).
- **The hybrid** does not beat the vector ranking on these claims, and it costs eighteen times
  more.
- **The oracle row** bounds what better chunk ranking could give: with the page known, 42 of 66
  gold sets are already in the first chunk.
- **Against the server defaults** on the 120-word ingest of the same corpus (`results/ef_20.md`,
  8,658 chunks), `M=6` and `ef_search=20` found 68 of 75 gold pages at k=10, where the exact
  ranking of those vectors finds 74. On the 240-word ingest the chosen settings find 74, which is
  what the exact ranking finds there too.

**Reproduce.**
- `wikilense eval --name mine` after `wikilense ingest` writes `results/mine.json` and `.md` with
  the headline table for the current ingest. The counts match the table above up to the HNSW
  build: a fresh graph differs in a few chunk lists, not in recall.
- `scripts/run_experiments.py all` reruns every group in `results/SUMMARY.md` (about 10 minutes
  with a GPU):
  - stability across cache sizes and a restart;
  - the `ef_search` sweep;
  - the `M` sweep with rebuild times;
  - chunk sizes at equal retrieved text;
  - the prefix;
  - the filter strategies;
  - the final configuration;
  - the search before `COMMIT`;
  - the optimizer's choice per filter.
- Every results file records the ingest parameters, the effective `ef_search`, the cache size, the
  index `M` and the chunk count it was measured on.
- `scripts/make_charts.py` redraws the charts above from the JSON files.

## What the database-native approach gave

- **Predicates, joins and the vector search in one statement.** "Chunks about X, in a category,
  longer than N words, edited since D, linked from Y" is one SQL statement that the optimizer
  plans, and `EXPLAIN` shows the plan.
  - A separate vector store answers the vector part and leaves the rest to application code with
    post-filtering. That is exactly the `overfetch` strategy we measured: cheaper, but short of `k`
    rows under selective filters.
  - Here the optimizer can also choose the other way round, scanning the few qualifying rows and
    sorting them exactly, when a filter is selective enough. A post-filtering store cannot do that.
- **Metadata is just more tables.** Categories and edit dates were added as two tables and two
  columns, filled in the same ingest transaction as the vectors and joined like anything else. No
  vector-store "metadata payload" has to be kept in sync.
- **The ground truth is a join.** Evidence recall is `chunk_sentence JOIN claim_evidence`, so the
  evaluation is SQL over the same tables, not a second pipeline that re-matches text.
- **One transaction, one backup, one set of foreign keys.** Vectors are rows. They roll back with
  their transaction, disappear with their page, and are covered by the same dump. There is no
  synchronisation job between a store and the metadata it describes. We measured the one caveat:
  rows not yet committed are searched less accurately by their own transaction.
- **Two rankings, one engine.** The `FULLTEXT` index and the vector index live on the same table and
  are fused with window functions and CTEs, without a second system.
- **Honest costs.**
  - The joined form is exact and about four times the bare index query on this corpus, and it
    grows with the table.
  - The bare form is approximate and needs `M` and `ef_search` tuned, which we did with
    measurements rather than adjectives.

## Limitations and next phase

- **Ground truth.** The 75 claims are FEVEROUS statements, not hand-written questions, and every
  one cites a single page because of the shard-based selection rule. The next phase:
  - multi-page claims;
  - a chosen benchmark claim set over a larger corpus;
  - the LLM fact-checker comparison: no retrieval, retrieved chunks, gold evidence.
- **Tables and captions are not indexed.** 27 of the 114 evidence ids point at table cells and
  cannot be matched. Lists are indexed.
- **Scale.** At 100 pages the exact ranking costs 2.2 ms. When the corpus is scaled, the number to
  watch is the `inline` cost, which grows with the table and with the filter's selectivity,
  against the bare index query, which does not.
- **Categories** are the explicit `[[Category:...]]` links of each page's December 2020 revision.
  Categories that templates add at render time (mostly hidden maintenance categories) are not
  included.
- **Edit dates** are the timestamp of that revision: "last edited, as the corpus knows it". The
  day FEVEROUS scraped each page was not published. For pages edited during December 2020 the
  revision is chosen by text match, which leaves 5.5% of the sentence probes unmatched (mostly
  text produced by templates).
- **HNSW results are approximate by construction,** and a search over rows the same transaction has
  not committed finds fewer true neighbours. The README states which numbers are exact.
- **MariaDB 11.8** has one vector index per table and no `INFORMATION_SCHEMA` view for it; the index
  size was read from the hidden tablespace (`chunk#i#NN`) as root.
- **Generation is out of scope.** The area's project text asks for retrieval, and the ground truth
  measures retrieval. The LLM fact-checker comparison is the planned extension.

## Repository

```
wikilense/          the package: corpus, wikitext (parsing), chunking, embedding, db, ingest,
                    search, evaluate, cli, web, config
sql/schema.sql      the data model; sql/examples.sql the worked statements for the mariadb client;
                    sql/docker-init/ creates the test database on first start
scripts/            build_corpus.py, fetch_wiki_shard.py, fetch_page_meta.py (categories and edit
                    dates from the MediaWiki API), run_experiments.py, make_charts.py
data/corpus/        the committed 100 pages, 75 claims and page metadata, with MANIFEST.md and
                    PAGE_META.md
results/            every experiment (JSON + Markdown) and SUMMARY.md
docs/DESIGN.md      the design contract, every deviation forced by the data, the measured outcomes
docs/ARCHITECTURE.md  a beginner's guide to the code: the pipeline, every file, one query end to end
docs/img/           the charts of this README (scripts/make_charts.py)
tests/              345 tests: pure tests, `db` tests against wikilense_test, `slow` tests that load the model
analysis/feverous/  the FEVEROUS analysis that chose the benchmark (earlier phase)
```

- `make test` runs the suite in about 20 s. `pytest -m "not slow"` skips the model tests, and the
  `db` tests skip when the server is unreachable.
- `make lint` runs ruff.
- [CI](.github/workflows/ci.yml) runs both on every push to `main` and on every pull request, against
  a `mariadb:11.8` service container with the CPU build of torch, installing from
  `requirements-cpu.lock`.

## Team

Ylli Rada, Constructor University, Databases and Web Services course. Code, experiments and
documentation were produced with Claude Code (Anthropic), directed by the author. The decisions,
their reasons and every deviation the data forced are recorded in [docs/DESIGN.md](docs/DESIGN.md)
and [CLAUDE.md](CLAUDE.md). Licence: MIT ([LICENSE](LICENSE)).
