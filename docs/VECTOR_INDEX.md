# How MariaDB 11.8.9 uses the vector index, measured

The full account behind the summary in the [README](../README.md#how-mariadb-uses-the-vector-index):
the SQL statements WikiLense runs, and what we measured about how the server executes them.

## The statements

Every statement is a named constant in [wikilense/search.py](../wikilense/search.py), assembled from
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

## What we measured

Everything here was measured on this corpus with `EXPLAIN`, `ANALYZE` and the harness.
[tests/test_db.py](../tests/test_db.py) and [tests/test_search.py](../tests/test_search.py) assert the
facts that hold on synthetic data: `key: embedding`, `r_rows` = `k`, both indexes for `rrf`, the
session scope of `ef_search`, the transactions, and the plan of the category filter. They also pin
the statement shapes that lose the index, listed below, and the `NULL` distances of a
wrong-dimension vector.

- Only `ORDER BY VEC_DISTANCE_COSINE(column, constant) LIMIT n`, sorted ascending, drives the
  index; the alias form `ORDER BY distance` is identical. Wrapping the distance in an expression,
  a `WHERE VEC_DISTANCE(...) < t` without `ORDER BY ... LIMIT`, and a second `ORDER BY` key
  (`..., chunk_id`) all drop the index, so ties are broken in Python.
- The arithmetic is right: on 300 random, unnormalised vectors, `VEC_DISTANCE_COSINE` equals
  NumPy's `1 - cos` for every row, within 1e-5, and a scan that ignores the index
  (`IGNORE INDEX (embedding)`) returns NumPy's exact top 10 in order. In our runs the HNSW index at
  `ef_search=200` found all of them. The test requires at least 95% on average, because every graph
  build is a different approximation.
- Query cost falls into two regimes:
  - The bare query (`none`, `overfetch`) is the HNSW search: 0.45 to 0.64 ms on 4,598 chunks.
    It reads exactly `LIMIT` rows, and `mhnsw_ef_search` sets its accuracy.
  - With a join in the statement (`inline`), `EXPLAIN` still says `type: index, key:
    embedding`, but the server ranks exactly over the index: 2.24 ms here. Its hit lists do not
    depend on `ef_search` (identical at 20 and 100 for 75 of 75 claims on the 120-word ingest),
    and on the synthetic test corpus they equal the known exact order. It then walks until `k`
    rows pass the predicates.
- The optimizer chooses per filter whether to keep the index (`results/planner_choices.md`: every
  claim, the inline statement, k=10), from its estimate of how many rows pass.
  - Nine of the eleven filters keep the vector index, as does the unfiltered statement. The rows
    read grow as the filter gets selective: 10 unfiltered, 108 under the category, 121 under the
    link filter, 315 under the History heading, 601 under `max_words 2000` (89 chunks pass).
  - The two most selective range filters on indexed `page` columns make it scan all 4,598 chunks,
    compute every distance and sort: `max_words 500` (25 chunks pass) and `edited_before
    2020-06-01` (4 pass). That plan is exact, at 5.2 and 3.1 ms in that run, and reads every chunk.
    It is the plan under our fixed join order; "The join order we fix" below measures the plan
    the optimizer picks when the order is left to it.
  - No query returned fewer than min(k, passing) rows: under `edited_before 2020-06-01` only 4
    chunks pass, and each query returns those 4.

  ![Chunk rows read under each filter](img/planner_rows.svg)

- The scale test (`results/scale.md`) runs on the 100 pages, then on the same pages with
  5,000 and 20,000 random pages of the full FEVEROUS Wikipedia added as distractors. The settings,
  the claims and their gold pages are unchanged; k=10, `ef_search` 100, 3 timed runs per claim
  after one untimed run.

  | Pages | Chunks | HNSW graph | Article@10, HNSW / exact | HNSW ms p50 | Exact ms p50 | History filter: inline ms p50, overfetch 10 short |
  |---|---|---|---|---|---|---|
  | 100 | 4,598 | 13.0 MB | 74/75 / 74/75 | 0.83 | 2.52 | 3.30, 69/75 |
  | 5,100 | 26,700 | 40.0 MB | 74/75 / 74/75 | 1.05 | 10.27 | 11.54, 68/75 |
  | 20,100 | 89,650 | 108.0 MB | 71/75 / 71/75 | 1.23 | 19.17 | 21.30, 65/75 |

  The bare HNSW search grows little (1.5 times for 19 times the chunks). The exact joined
  statement, and the inline form of a filtered query, grow with the table (7.6 times). At every
  size, `inline` filled all `k` rows under the History filter, while `overfetch 10` stayed near
  1 to 2 ms and came back short for most queries. The HNSW hit lists equal the exact ones for
  75, 73 and 72 of 75 claims.

  ![Query time as the chunk table grows](img/scale_cost.svg)

- The bounded form, `overfetch`, stays near the bare cost but returns fewer than `k` rows when the
  filter is selective; the inline form is complete. Under `heading LIKE '%History%'`
  (117 of 4,598 chunks pass), at k=10:

  | Strategy | Queries short of 10 rows (of 75) | Median time |
  |---|---|---|
  | `overfetch 10` | 69 (18 of them with no row) | 0.78 ms |
  | `overfetch 50` | 20 | 2.28 ms |
  | `inline` | 0 | 3.07 ms (4.06 ms p95) |

  (Measured again on 2026-11-11, `results/strategy_*_history.md`; the counts are those of the
  October run. Under this filter only 26 of the 75 claims have a gold page with a chunk in a
  History section at all, and `inline` finds it for 25 of them.)
- Join order matters:
  - With a plain `JOIN`, the optimizer started from `page` on stale statistics right after a bulk
    insert and lost the vector index (observed in phase 2, docs/DESIGN.md). The statements use
    `STRAIGHT_JOIN` with `chunk` first, and the ingest runs `ANALYZE TABLE`.
  - An `IN (subquery)` filter becomes a semi-join and loses the index too (a test pins this).
  - The same filter as a joined derived table keeps the index (a test pins that plan). The
    derived tables are `SELECT DISTINCT`: without it MariaDB merges them into the join, which
    still walks the vector index but returns a chunk once per matching link row ("The join
    order we fix" below has the measurement, and the correction of what this file said before).
- A test searches rows the transaction has not committed (`results/uncommitted_recall.md`): it
  fills a scratch table with the chunk table's vector column and index inside one transaction and
  searches it before and after `COMMIT`. There are three fresh builds each, and the overlap is
  measured against NumPy's exact top 10. With our 4,598 chunk embeddings and the 75 claims, the
  search finds 0.972 to 0.973 of the exact top 10 before `COMMIT` at
  `mhnsw_ef_search` 20, 100 and 400, and 0.990 to 1.000 after. At ef 100 the worst single query
  finds 4 of its 10 neighbours before `COMMIT` and all 10 after. On 1,000 random vectors the
  overlap is 0.583 to 0.623 before and 0.787 to 0.996 after.

  Each state gets an untimed warm-up pass, and then the latency differs little: 0.44 to 0.86 ms p50
  before `COMMIT` against 0.69 to 1.03 ms after, on the chunks. An earlier run without the warm-up
  showed a threefold gap, which was a cold-cache effect. `ROLLBACK` leaves no row behind. Write and
  search in separate transactions, or commit first.

  ![Overlap with the exact top 10 before and after COMMIT](img/uncommitted_recall.svg)

- `mhnsw_ef_search` is a session variable, set per query and restored after. Values outside its
  range are clamped with a warning, not an error. `mhnsw_max_cache_size` is global. Its 16 MB
  default is already the size of the `M=6` graph of the 8,658-chunk 120-word ingest (17 MB with
  `M=16`), so the Docker configuration starts the server with 512 MB. The graph lives in a hidden
  InnoDB table (`chunk#i#NN`), not in `SHOW TABLE STATUS`'s `Index_length`.
- Raw float32 bytes bind as a `_binary` literal, both in `INSERT` and inside
  `VEC_DISTANCE_COSINE`. PyMySQL 1.2.3 and later send `_binary X'<hex>'` by themselves. Versions
  1.2.0 to 1.2.2 do so only on a connection opened with `binary_prefix=True`, which `db.connect`
  sets for every version. `VEC_FromText(?)` with a JSON list and `UNHEX(?)` work everywhere. A
  query vector of the wrong dimension returns `NULL` distances, silently.
- A chunk inserted inside a transaction is visible to the index in the same connection, and gone
  after `ROLLBACK`. Another connection sees it only after `COMMIT`. `UPDATE` of an embedding
  changes the distance at once, `DELETE` removes it from the results, and deleting a page cascades
  to its vectors. There is no synchronisation job and no second store. `sql/examples.sql`
  (example 8) shows it from the `mariadb` client.
- Every build of the HNSW graph is a different approximation: two `M=6` builds of the same vectors
  (120-word chunks) agreed on 12 of 75 hit lists. "The same vectors, built twenty times" below
  measures the spread on the final settings. Within one server process the results are
  deterministic, and the cache size does not change them (75 of 75 identical at 16 MB and 512 MB). A
  container restart (`docker restart`) changed 1 of 75 hit lists on the `M=6` index and none on
  `M=16` at `ef_search=100`, without changing recall.

## Three regimes of one statement

The section above says "two regimes": the bare statement, and the statement with a join. There is
a third, which WikiLense's own statements never use because every filter of theirs sits on a
joined table: a predicate on the indexed table itself.
[scripts/measure_index_behaviour.py](../scripts/measure_index_behaviour.py) runs the three shapes
on the corpus and compares every hit list with NumPy's exact ranking of the chunks that pass
([results/statement_forms.md](../results/statement_forms.md); 75 claims, k=10, the shapes taking
turns query by query; the two `ef_search` values were run one after the other, so compare times
inside one value). `EXPLAIN` gives `chunk` the type `index` and the key `embedding` in all six
shapes:

| Statement | Chunks passing | ef_search | p50 ms | p95 ms | Equal to exact | Short | Rows read |
|---|---|---|---|---|---|---|---|
| the bare statement on chunk | 4,598 | 20 | 0.36 | 0.77 | 75/75 | 0/75 | 10 |
| the same, joined to page | 4,598 | 20 | 2.16 | 3.26 | 75/75 | 0/75 | 10 |
| WHERE chunk.n_words < 60 | 766 | 20 | 1.12 | 3.43 | 61/75 | 0/75 | 88 |
| WHERE chunk.n_words < 60, joined to page | 766 | 20 | 2.25 | 3.48 | 75/75 | 0/75 | 84 |
| WHERE chunk.n_words < 30 | 304 | 20 | 3.99 | 13.72 | 41/75 | 0/75 | 276 |
| WHERE chunk.n_words < 30, joined to page | 304 | 20 | 2.45 | 3.77 | 75/75 | 0/75 | 270 |
| the bare statement on chunk | 4,598 | 100 | 0.76 | 1.06 | 75/75 | 0/75 | 10 |
| the same, joined to page | 4,598 | 100 | 2.42 | 3.38 | 75/75 | 0/75 | 10 |
| WHERE chunk.n_words < 60 | 766 | 100 | 2.10 | 4.97 | 68/75 | 0/75 | 86 |
| WHERE chunk.n_words < 60, joined to page | 766 | 100 | 2.46 | 3.23 | 75/75 | 0/75 | 84 |
| WHERE chunk.n_words < 30 | 304 | 100 | 6.71 | 18.46 | 51/75 | 0/75 | 277 |
| WHERE chunk.n_words < 30, joined to page | 304 | 100 | 2.68 | 3.39 | 75/75 | 0/75 | 270 |

- **Bare.** The HNSW search. It reads `LIMIT` rows.
- **A predicate on `chunk` itself** (`n_words`, a column without an index). The server does not
  stop at `LIMIT`: it keeps taking rows from the index until `k` of them pass
  (88 and 276 rows read at the median at `ef_search` 20). None of the 300 answers
  came back short, unlike a filter applied after the search; we ran nothing more selective than
  304 passing chunks. But the order the index hands rows over in is the approximate one, so the
  result is not the exact filtered top 10: 61 and 41 of 75 claims at `ef_search` 20,
  68 and 51 at 100. And each continuation costs time: with 304 passing chunks the statement
  takes longer than the joined form (6.71 against 2.68 ms at `ef_search` 100); with 766 it is still
  the faster one.
- **The same predicate with a join in the statement.** The exact top 10 for 75 of 75 claims at
  both `ef_search` values. It reads about as many rows (84 and 270), in the exact order.

For a schema this means: copying a filter column onto the vector table, without an index of its
own, did not buy an exact filtered search here, and under the more selective predicate not a fast
one either. With a join in the statement MariaDB ranked exactly in every case we measured. We did
not try an indexed column on `chunk`; the next section shows what the optimizer does with an
index on the filter column of a joined table.

## The join order we fix, and the plan the optimizer would pick

The filtered statement names its tables with `STRAIGHT_JOIN`, `chunk` first (the reason is under
"Join order matters" above). We had not measured what that costs.
`scripts/measure_index_behaviour.py join_order` runs every filter of the planner experiment in two
forms, as written and with every `STRAIGHT_JOIN` replaced by `JOIN`, and compares both with the
same statement under `IGNORE INDEX (embedding)`, which reads and sorts every chunk
([results/join_order.md](../results/join_order.md); 75 claims, k=10, `ef_search` 100). Each time is
the median of the form run alone; in brackets is the median when the two forms take turns query
by query. Taking turns guards against the laptop drifting between two blocks, but a form that
reads thousands of rows slows the other form's next run, so we give both:

| Filter | Chunks passing | As written | Rows read | p50 ms (taking turns) | Plain `JOIN` | Rows read | p50 ms (taking turns) |
|---|---|---|---|---|---|---|---|
| none | 4,598 | vector index | 10 | 2.11 (2.19) | vector index | 10 | 2.09 (2.19) |
| min_words_1000 | 4,544 | vector index | 10 | 2.08 (2.09) | vector index | 10 | 2.31 (2.10) |
| min_words_5000 | 3,891 | vector index | 17 | 2.11 (2.61) | `page` first | 3891 | 10.24 (10.37) |
| max_words_2000 | 89 | vector index | 601 | 3.29 (3.20) | `page` first | 89 | 0.77 (0.96) |
| max_words_500 | 25 | scan of `chunk` | 4598 | 4.76 (4.72) | `page` first | 25 | 0.50 (0.73) |
| heading_history | 117 | vector index | 315 | 2.99 (3.02) | vector index | 315 | 2.93 (3.03) |
| linked_from | 661 | vector index | 121 | 3.13 (3.17) | derived table first | 661 | 2.58 (2.66) |
| category | 695 | vector index | 108 | 2.41 (3.09) | derived table first | 695 | 2.34 (2.62) |
| edited_since_2020_12_01 | 4,525 | vector index | 10 | 2.54 (2.60) | vector index | 10 | 2.61 (2.60) |
| edited_before_2020_12_15 | 778 | vector index | 67 | 2.86 (2.81) | `page` first | 778 | 2.63 (2.79) |
| edited_before_2020_06_01 | 4 | scan of `chunk` | 4598 | 3.59 (3.31) | `page` first | 4 | 1.09 (0.84) |
| four_filters | 377 | vector index | 184 | 3.36 (3.47) | derived table first | 377 | 2.07 (2.12) |

A run made a minute earlier ([results/join_order_run1.md](../results/join_order_run1.md)) has the
same plans, rows read and agreement with the scan in every row. Its medians show how far a time
below a millisecond moves on this laptop: under `max_words 500` the free form took
0.54 ms there and the form as written 4.94 ms; under `min_words 5000`,
10.61 and 2.58 ms.

- **As written**, the server walks the vector index under 10 of the 12 filters. Under the two most
  selective range filters it reads all 4,598 chunks and sorts (`chunk` has type `ALL` in
  `EXPLAIN`; the `wikilense query ... --explain` output for `--max-words 500` also shows `page`
  read by `range` on `ix_page_n_words` through a join buffer).
- **With a plain `JOIN`**, the optimizer starts from another table under 8 of the 12 filters:
  `page`, or the derived table of the link or category filter. It reads `chunk` by
  `uq_chunk_page_ordinal`, exactly as many rows as pass the filter, and the vector index is not
  in the plan. This is what a vector store calls pre-filtering, chosen here by cost. We saw no
  plan that combines it with the vector index.
- **That is cheaper when few chunks pass:** 0.50 ms instead of
  4.76 ms when 25 pass, 1.09 instead of
  3.59 when 4 pass, 0.77 instead of
  3.29 when 89 pass, and 2.07 instead of
  3.36 under the brief's four filters (377 pass).
- **And dearer when many pass:** under `min_words 5000` (3,891 of 4,598 chunks pass) the free plan
  reads those 3,891 chunks and sorts, 10.24 ms instead of
  2.11 ms. The free plan reads every passing chunk, so its cost
  follows the number that pass; we did not measure it on a larger table. The fixed order took no
  such plan under any of the 12 filters, which is the reason it stays. Its own worst case is the
  scan of all 4,598 chunks under the two most selective filters.
- Under the link, category and `edited_before 2020-12-15` filters the plan moves too, and the
  times stay within a factor of 1.5, which we do not read as a difference on this laptop.
- Both forms return the same chunks as the scan for every claim under every filter (10 of them,
  or the 4 that pass `edited_before 2020-06-01`).
- **Order.** Under the four filters the statement as written returned the same 10 chunks in
  another order than the scan for 2 of 75 claims. In those 2 lists the rows are not in ascending
  order of the distance the statement itself reports, and where the list differs from the
  scan's the two reported distances differ by at most 3.9e-06. That fits the index ranking by
  its 16-bit copies of the vectors (its author's description, linked below) while the select
  list computes the distance from the stored 32-bit vector. We did not test that explanation
  and did not read the server source. `wikilense` sorts the returned rows by the reported
  distance and the chunk id, so its lists equal the exact ranking.

- **The derived tables need their `DISTINCT` for the result, not for the index.** The same
  statements with `SELECT DISTINCT` turned into `SELECT`: MariaDB merges the derived table into
  the join (no `<derived>` row in `EXPLAIN`) and still walks the vector index, for all 75 claims
  under the link, the category and the four filters. But under the link filter 75 of 75 hit
  lists then name the same chunk more than once (0 under the category filter, 75 under the four
  filters), because a page can link to another page several times. Earlier versions of the
  README and of this file said that the merged form loses the index. That does not reproduce on
  11.8.9, neither on the corpus nor on the test data, and the statement was removed.

What follows for the design: `STRAIGHT_JOIN` trades the best plan under a very selective filter
for a predictable one. A statement that leaves the order free only when few rows pass would be
faster there. We did not build it: the settings are fixed since the evaluation, and the rule for
"few" would need its own measurement at scale.

## The same vectors, built twenty times

"Every build of the HNSW graph is a different approximation" above rested on two builds. The
`builds` part of `scripts/measure_index_behaviour.py` copies the 4,598 chunk vectors into a scratch
table with the chunk table's vector column and index, ten times for `M=16` and ten times for the
server default `M=6`, and searches every build with the 75 claims (the bare statement, k=10). It
was run twice ([results/build_variance.md](../results/build_variance.md),
[results/build_variance_run1.md](../results/build_variance_run1.md)). Each cell is the smallest to
the largest value over the twenty builds of that `M` (for "two builds agree", over the pairs of
builds inside a run):

| M | ef_search | Article recall@10, of 75 | Top-10 list equal to exact, of 75 | Two builds agree, of 75 | Build, s |
|---|---|---|---|---|---|
| 16 | 100 | 74 | 74 to 75 | 74 to 75 | 1.4 to 2.3 |
| 16 | 20 | 73 to 74 | 70 to 74 | 68 to 74 | 1.4 to 2.3 |
| 6 | 100 | 72 to 74 | 50 to 63 | 39 to 57 | 0.4 to 0.6 |
| 6 | 20 | 65 to 73 | 38 to 51 | 26 to 44 | 0.4 to 0.6 |

- **At the chosen settings (`M=16`, `ef_search` 100) the build does not matter for recall.** Every
  one of the twenty builds found the gold page for 74 of 75 claims, which is what NumPy's exact
  ranking of the same vectors finds, and its top-10 lists equalled the exact ones for 74 or 75
  claims.
- **At `ef_search` 20 the same graphs differ more**: 70 to 74 lists equal to the exact ones, and
  73 to 74 gold pages. So the larger `ef_search` is also what makes a result repeatable across
  builds.
- **With the server defaults (`M=6`, `ef_search` 20) the build decides several claims**: 65 to 73
  gold pages, 38 to 51 lists equal to the exact ones, and two builds agree on only
  26 to 44 of 75 lists.
- Two more builds are in the other results files, measured differently (top-20 lists against the
  joined statement): the build behind the headline run gave 74 of 75 identical lists at
  `ef_search` 100 and 65 at 20 (`results/final_ef_100.md`, `final_ef_20.md`). The build now in the
  database, made by the ingest that followed, gave 75 of 75 top-10 lists equal to the exact ones at
  both values (`results/statement_forms.md`).
- The scratch table is filled in one transaction, in `chunk_id` order. We did not test whether
  the way a table is filled (batch size, commits between batches) changes the graph's quality.

## Isolation levels, and a reader that leaves the index stale

The documentation says that "all transaction isolation levels are supported". We checked what a
search sees while another connection writes, and found one sequence after which the index gave
wrong answers to every connection we tried ([results/isolation.md](../results/isolation.md); a
scratch table of 1,000 random vectors with the chunk table's index definition, in the test
database, one query, `ef_search` 100, the table rebuilt for every trial).

Writer B inserts 5 rows that are nearer to the query than every other row. Each cell is how many
of them reader A finds among its top 10 through the index / by a scan, in 5 trials each:

| A's isolation level | B has not committed | B has committed, A's transaction still open | A's next transaction | B's insert |
|---|---|---|---|---|
| READ UNCOMMITTED | 4 to 5 / 5 | 5 / 5 | 5 / 5 | went through |
| READ COMMITTED | 0 / 0 | 5 / 5 | 5 / 5 | went through |
| REPEATABLE READ | 0 / 0 | 0 / 0 | 0 / 5 | went through |
| SERIALIZABLE | 0 / 0 | 0 / 0 | 0 / 0 | blocked: Lock wait timeout exceeded; try restarting transaction (error 1205) |

At READ UNCOMMITTED and READ COMMITTED, and in the first two moments at REPEATABLE READ, the index
agrees with the scan and with what the level promises. One exception: at READ UNCOMMITTED the
index found 4 of B's 5 uncommitted rows in 1 of 5 trials; an approximate search can miss a row,
and we did not look for another cause. At SERIALIZABLE the search takes locks that made B's
insert wait and time out, so that row shows a table nothing was written to. The last moment of
the REPEATABLE READ row is the finding: in A's *next* transaction a scan finds the 5 rows and the
index finds none.

The sequence, which we repeated 10 times on fresh tables, with six controls of 5 trials:

1. A, at REPEATABLE READ, opens a transaction and searches the index.
2. B inserts the 5 rows and commits.
3. A searches again, still inside its old snapshot (and finds none of the new rows, as it should).
4. A commits and disconnects.
5. A new connection searches: does the index return none of the committed rows that a scan returns?

| Sequence | Stale at step 5 | New rows found through the index, per trial |
|---|---|---|
| as above | 10 of 10 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| control: A is at READ COMMITTED | 0 of 5 | [5, 5, 5, 5, 5] |
| control: A skips step 3 | 0 of 5 | [5, 5, 5, 5, 5] |
| control: A is in autocommit mode, so step 3 is a new transaction | 0 of 5 | [5, 5, 5, 5, 5] |
| control: A's statement in step 1 is `SELECT COUNT(*)`, not a search | 0 of 5 | [5, 4, 5, 5, 5] |
| control: A's statement in step 1 is the scan, not the index search | 0 of 5 | [5, 5, 5, 5, 5] |
| control: A commits but stays connected | 5 of 5 | [0, 0, 0, 0, 0] |

- **What it takes.** In every trial that went stale, a REPEATABLE READ transaction had searched
  the index before B's commit and again after it. No control without one of the two went stale:
  not at READ COMMITTED, not in autocommit mode, not without the second search, and not when A's
  first statement was a `COUNT(*)` or a scan. (In 1 control trial the index found 4 of the 5
  rows: an approximate search, not this state.) The last control shows that A's disconnect is
  not needed: the index was stale when A committed and stayed connected.
- **Who gets the wrong answer.** Everyone we tried. On 5 stale tables, new connections at
  REPEATABLE READ, READ COMMITTED and READ UNCOMMITTED and in autocommit mode each found 0 of the
  5 rows through the index. The isolation level of the connection that asks does not protect it.
- **It lasts.** Three further new connections each found 0 of the 5 rows, on six stale tables.
- **What ends it.** The next committed write to the table (then the index found all 6 of 6 new
  rows, 3 of 3 tables) or `FLUSH TABLES` on the table (5 of 5, 3 of 3 tables). `FLUSH TABLES`
  makes the server open the table again, so the index as stored was right. That fits a copy of
  the graph in the server's memory that was loaded inside A's old snapshot; we did not inspect
  the cache or read the server source.
- **With a `DELETE` in step 2** (B deletes the three nearest rows), the index then returned **no
  row at all** for that query in 10 of 10 trials, from a table of 997 rows where a scan returned
  10. Of 20 other queries none came back empty, but the shortest answer had 5 rows. A committed
  insert (3 of 3 tables) or `FLUSH TABLES` (once) brought the 10 rows back, and the control
  without A's second search returned 10 rows in 5 of 5.

What to do when other connections write to the table: no connection should search the index twice
inside one REPEATABLE READ transaction. Searching at READ COMMITTED or in autocommit mode, or ending
the transaction between searches, did not cause the state. PyMySQL opens connections with autocommit
off, so a connection that has run one `SELECT` is in a REPEATABLE READ transaction until it commits,
rolls back or closes. That is how WikiLense's own connections work. A web request runs one search
and closes. `wikilense query` runs a `COUNT(*)`, loads the model, runs one search and closes, which
is the shape of the `COUNT(*)` control. `wikilense eval` and the experiment scripts run many
searches in one transaction, so a committed write to the corpus by another connection during such a
run would start the sequence. Nothing here does that: only an ingest (`wikilense ingest`, or the one
inside an experiment group) commits writes to the corpus, and it recreates the tables;
`sql/examples.sql` updates one row and rolls back. The measurements are not affected. The experiment
was run three times with the same outcome in every row of the sequence table that the runs share
([results/isolation_run1.md](../results/isolation_run1.md) is the first run, made before three of
the controls were added). We found no MariaDB ticket that describes this (searched on 2026-11-11),
have not reported it, and measured it on 11.8.9 only, on this one table.

## The joined statement against a brute-force ranking

"The server ranks exactly" above rested on two observations: the hit lists of the joined statement
do not change with `mhnsw_ef_search`, and on the synthetic test corpus they equal the known order.
[scripts/compare_separate_index.py](../scripts/compare_separate_index.py) checks it directly on the
real corpus. It reads the 4,598 vectors out of MariaDB, ranks them for every claim with NumPy in
float64, and compares that list with the statement's ([results/separate_index.md](../results/separate_index.md),
two runs):

- the joined statement without a filter: the same 10 chunks in the same order for 75 of 75 claims;
- under `heading LIKE '%History%'` and under the brief's four filters: 75 of 75 again, against the
  exact ranking of the chunks that pass. These are the lists of `search()`, which sorts the rows
  it gets by the reported distance and the chunk id. The rows as the server returned them had two
  near-equal neighbours swapped for 2 of 75 claims under the four filters (see "Order" above);
- the bare HNSW statement at `ef_search=100`: 75 of 75 on this graph build. The October build gave
  74 of 75 against the joined statement (`results/final_ef_100.md`).

The same file measures what a copy of the vectors outside the database costs and saves; the README
section "Where this approach loses" reads it.

## Checked against the documentation on 2026-11-11

The MariaDB pages for vectors were read again
([Vector Overview](https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-overview),
[CREATE TABLE with Vectors](https://mariadb.com/docs/server/reference/sql-structure/vectors/create-table-with-vectors),
[Vector System Variables](https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-system-variables)),
and each statement tried on the 11.8.9 container:

| The documentation says | On 11.8.9 |
|---|---|
| The index is used only for `ORDER BY VEC_DISTANCE_*(column, vector)`, or its alias, ascending, with a `LIMIT` | confirmed: without `LIMIT`, and with `DESC`, `EXPLAIN` shows type `ALL` and a filesort (a test pins both) |
| A distance wrapped in an expression, and a `WHERE` on the distance without `ORDER BY ... LIMIT`, fall back to a full scan | confirmed (tests pin both, the expression ascending and descending) |
| A `WHERE` on the distance added to an `ORDER BY ... LIMIT` query still uses the index | confirmed: type `index`, key `embedding`, `Using where` (a test pins it) |
| A search with another distance function than the index was built for cannot use the index | confirmed: `VEC_DISTANCE_EUCLIDEAN` on our cosine index gives type `ALL`; the generic `VEC_DISTANCE` uses the index (a test pins both) |
| Only one vector index per table; the indexed column must be `NOT NULL` | our schema follows it |
| `M` from 3 to 200, default 6; `mhnsw_ef_search` from 1 to 10000, default 20; `mhnsw_max_cache_size` default 16 MB | the same values in `information_schema.SYSTEM_VARIABLES` (a test reads them) |
| `INFORMATION_SCHEMA.VECTOR_INDEXES` exists from MariaDB 13.1 | the table does not exist in 11.8.9, which is why the graph size was read from the tablespace file |
| "Concurrent reads/writes and all transaction isolation levels are supported" | we measured two things it does not mention: the lower recall over a transaction's own uncommitted rows, and the index left stale by a REPEATABLE READ reader ("Isolation levels" above) |

Not in the reference, and measured here: the second `ORDER BY` key and the `IN (subquery)` that
lose the index; the exact ranking in every joined statement we compared,
and the approximate, continuing search under a predicate on the indexed table itself; the
optimizer's switch to a scan under selective range filters, and its choice of the filter's own
index over the vector index when the join order is left free; the hit lists that change with the
graph build (and how much, over twenty builds) and, for `M=6`, across a restart; the stale index
after a REPEATABLE READ reader.

Two facts about the index come from its author's description, not from our measurements
([MariaDB Vector: How it works, part II](https://mariadb.org/mariadb-vector-how-it-works-part-ii/)
and [part III](https://mariadb.org/mariadb-vector-how-it-works-part-iii/)): the graph stores each
vector as 16-bit integers with a scale factor, and `ef_construction` is fixed at 10.
