# How MariaDB 11.8.9 uses the vector index, measured

The full account behind the summary in the [README](../README.md#how-mariadb-uses-the-vector-index):
the SQL statements WikiLense runs, and what we measured about how the server executes them.

The measurements are on the two databases of the 3,000-claim run
([EVALUATION.md](EVALUATION.md#the-scaled-run)): **S1**, 7,686 pages and 132,359 chunks with a
156.0 MB graph, and **S0**, 3,851 pages and 43,134 chunks with a 56.0 MB graph, at k = 5. (MB are
of 1,048,576 bytes here, as MariaDB's documentation writes `mhnsw_max_cache_size`.) The
behaviour around transactions was measured on scratch tables in the test database. The files are
in [results/scaled/](../results/scaled/), and [SUMMARY.md](../results/scaled/SUMMARY.md) has every
table with every run. The earlier measurements on the 100-page demo corpus (4,598 chunks, 75
claims) are in [results/](../results/); this page names them where it uses them.

**How these were measured.** One measurement at a time, with the laptop's CPU capped at 50% and the
GPU at 2,100 MHz, so compare times only within this page. Each claim gives one timed run per
statement after one untimed run, which is 3,000 timings per median. The join-order measurement
runs 18 filters, two forms each, plus a scan of every passing chunk per claim. A pilot on 20 claims
took 2,426 s ([results/scaled/pilot/](../results/scaled/pilot/)), which puts all 3,000 claims at
about 100 hours, so that measurement uses a fixed random sample of 200 claims
(`random.Random(20261119)`, the same claims on S0 and S1). Each measurement here ran once so far;
a rerun goes into `results/scaled/run2/`, and `scripts/scaled_summary.py` adds it to
[SUMMARY.md](../results/scaled/SUMMARY.md) as a second column.

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

Everything here was measured with `EXPLAIN`, `ANALYZE` and the harness.
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
  (`IGNORE INDEX (embedding)`) returns NumPy's exact top 10 in order. The test requires the HNSW
  index to find at least 95% of them on average, because every graph build is a different
  approximation.
- Query cost falls into two regimes:
  - The bare query (`none`, `overfetch`) is the HNSW search: 1.11 ms at the median on S0 and
    1.43 ms on S1 in the main harness (`results/scaled/eval/`). It reads exactly `LIMIT` rows,
    and `mhnsw_ef_search` sets its accuracy.
  - With a join in the statement (`inline`), `EXPLAIN` still says `type: index, key: embedding`,
    but the server ranks exactly over the index: 15.74 ms on S0 and 25.81 ms on S1, against
    1.43 and 1.68 ms for the bare query in the same run (`results/scaled/timing/`). The cost grows
    with the table; "The joined statement against a brute-force ranking" below checks the lists.
    It then walks until `k` rows pass the predicates.
- Under a filter, the bounded form `overfetch` stays near the bare cost but returns fewer than `k`
  rows when the filter is selective; the inline form is complete. Under
  `heading LIKE '%History%'`, at k = 5, on the 3,000 claims:

  | Strategy | S0: short of 5 rows | S0 p50 | S1: short of 5 rows | S1 p50 |
  |---|---|---|---|---|
  | `overfetch 10` | 2,472 | 1.63 ms | 2,425 | 1.87 ms |
  | `inline` | 0 | 16.49 ms | 0 | 26.59 ms |

- A search that fewer rows pass than `LIMIT` is not capped on 11.8.9, and it can run away. The
  `hard_cap` part of `scripts/measure_index_behaviour.py` ran 30 claims at k = 10 in a throwaway copy
  of S1, with a 2 GB memory cap and a 10 s `max_statement_time`
  ([results/scaled/run1/s1/hard_cap.md](../results/scaled/run1/s1/hard_cap.md)).
  - **A distance threshold inside the statement.**
    - 30 chunks meet it: all 60 executions returned 10 rows (p50 1.08 ms).
    - Exactly 10 meet it: 54 returned them (p50 1.18 ms), 6 ran to the time limit.
    - Only 3 meet it: 57 of 60 executions ran to the time limit.
  - **A predicate on `chunk` itself.**
    - 694 chunks passing (0.52%): p50 1,357.45 ms, p95 5,260.05 ms, stopped in 4 of 60.
    - 133 chunks passing (0.10%): p50 2,389.48 ms, stopped in 32 of 60.
    - The same predicates joined to `page` were never stopped (p50 31.57 and 654.14 ms).
    - No answer that came back was short.
  - **The documentation's subquery form.** It read exactly its inner `LIMIT` rows, in about 1 ms,
    and came back short in 0 to 6 of 60 executions, where the index missed a qualifying chunk.
  - **Without a time limit**, such a statement grew `mariadbd` until it ran out of memory: to the
    2 GB cap in about 20 s in a throwaway container, and to 7.6 GB on the development machine's
    server, where the kernel's OOM killer stopped it.
  - **Other versions and the guard.** 12.3.3 and 13.0.2 kept their memory flat on the same data.
    `max_statement_time` stops the search ("A search that runs away" below). The web page and
    `wikilense query` set it to 5 s; their filters are joins.
- The optimizer chooses per filter whether to keep the index, from its estimate of how many rows
  pass; "The join order we fix" below has the plans under 18 filter levels.
- Join order matters:
  - With a plain `JOIN`, the optimizer started from `page` on stale statistics right after a bulk
    insert and lost the vector index. The statements use `STRAIGHT_JOIN` with `chunk` first, and
    the ingest runs `ANALYZE TABLE`.
  - An `IN (subquery)` filter becomes a semi-join and loses the index too (a test pins this).
  - The same filter as a joined derived table keeps the index (a test pins that plan). The
    derived tables are `SELECT DISTINCT`: without it MariaDB merges them into the join, which
    still walks the vector index but returns a chunk once per matching link row (on S1 in up to
    147 of 200 lists, "The join order we fix" below).
- Rows a transaction has not committed are found less well. `scripts/scaled_uncommitted.py` fills
  a scratch table in the test database with S0's 43,134 chunk vectors inside one transaction, with
  the chunk table's vector column and index, and searches it with the 3,000 claims before and after
  `COMMIT`, in three fresh builds, against NumPy's exact top 5
  ([results/scaled/run1/s0/uncommitted_recall.md](../results/scaled/run1/s0/uncommitted_recall.md)).
  Before `COMMIT` the search finds 0.872 to 0.881 of the exact top 5, at `mhnsw_ef_search` 20, 100
  and 400 alike, and at every value some query finds none of its 5 neighbours. After `COMMIT` it
  finds 0.973 to 0.983 at 20, 0.993 to 0.996 at 100 and 0.998 to 0.999 at 400. Before `COMMIT`
  the statement is also faster (0.32 against 0.83 ms p50 at 100). We did not look into how the
  server searches rows that are not committed. On 1,000 random vectors the overlap is 0.611 to
  0.655 before and 0.802 to 0.997 after. On the 100-page corpus it was 0.972 to 0.973 before and
  0.990 to 1.000 after (k = 10, `results/uncommitted_recall.md`). The 43,134 inserts took 21.9 to
  22.2 s, committed or not, and `ROLLBACK` leaves no row behind. Write and search in separate
  transactions, or commit first.
- `mhnsw_ef_search` is a session variable, set per query and restored after. Values outside its
  range are clamped with a warning, not an error. `mhnsw_max_cache_size` is global, and its 16 MB
  default is below the size of both graphs (56.0 MB on S0, 156.0 MB on S1), so the Docker
  configuration starts the server with 512 MB. The graph lives in a hidden InnoDB table
  (`chunk#i#NN`), not in `SHOW TABLE STATUS`'s `Index_length`.
- Raw float32 bytes bind as a `_binary` literal, both in `INSERT` and inside
  `VEC_DISTANCE_COSINE`. PyMySQL 1.2.3 and later send `_binary X'<hex>'` by themselves. Versions
  1.2.0 to 1.2.2 do so only on a connection opened with `binary_prefix=True`, which `db.connect`
  sets for every version. `VEC_FromText(?)` with a JSON list and `UNHEX(?)` work everywhere. A
  query vector of the wrong dimension returns `NULL` distances, silently. MariaDB knows this:
  [MDEV-35192](https://jira.mariadb.org/browse/MDEV-35192) was closed as documented.
- A chunk inserted inside a transaction is visible to the index in the same connection, and gone
  after `ROLLBACK`. Another connection sees it only after `COMMIT`. `UPDATE` of an embedding
  changes the distance at once, `DELETE` removes it from the results, and deleting a page cascades
  to its vectors. There is no synchronisation job and no second store. One open report,
  [MDEV-41227](https://jira.mariadb.org/browse/MDEV-41227), has rows become unreachable through a cosine vector index after
  `ON DELETE CASCADE`. Our ingest builds the tables fresh and never cascades while measuring. `sql/examples.sql`
  (example 8) shows it from the `mariadb` client.
- Every build of the HNSW graph is a different approximation; "The same vectors, built several
  times" measures the spread. Within one server process the results are deterministic while the
  graph fits in the index cache, except just after the table is reopened (10 of 3,000 bare hit
  lists on S1 changed between the first two passes, "The index cache below and above the graph").
  A container restart (`docker restart`) changed 1 of 75 hit lists
  on an `M=6` index of the demo corpus and none on `M=16` at `ef_search=100`, without changing
  recall.

## Three regimes of one statement

"What we measured" names two regimes: the bare statement, and the statement with a join. A third
is a predicate on the indexed table itself, which WikiLense's own statements never use because
every filter of theirs sits on a joined table. `scripts/measure_index_behaviour.py forms`
runs the three shapes on S1 and compares every hit list with NumPy's exact ranking of the chunks
that pass ([results/scaled/run1/s1/statement_forms.md](../results/scaled/run1/s1/statement_forms.md);
3,000 claims, k = 5, one timed pass per claim, the shapes taking turns query by query; the two
`ef_search` values ran one after the other, so compare times inside one value). `EXPLAIN` gives
`chunk` the type `index` and the key `embedding` in all six shapes:

| Statement | Chunks passing | ef_search | p50 ms | p95 ms | Equal to exact | Short | Rows read |
|---|---|---|---|---|---|---|---|
| the bare statement on chunk | 132,359 | 20 | 1.13 | 2.94 | 2,776/3,000 | 0 | 5 |
| the same, joined to page | 132,359 | 20 | 26.97 | 31.89 | 3,000/3,000 | 0 | 5 |
| WHERE chunk.n_words < 60 | 26,935 | 20 | 4.79 | 19.23 | 2,389/3,000 | 0 | 35 |
| WHERE chunk.n_words < 60, joined to page | 26,935 | 20 | 27.14 | 33.50 | 2,995/3,000 | 0 | 34 |
| WHERE chunk.n_words < 30 | 12,341 | 20 | 20.82 | 179.15 | 2,209/3,000 | 0 | 120 |
| WHERE chunk.n_words < 30, joined to page | 12,341 | 20 | 27.59 | 43.25 | 2,984/3,000 | 0 | 114 |
| the bare statement on chunk | 132,359 | 100 | 2.16 | 4.39 | 2,917/3,000 | 0 | 5 |
| the same, joined to page | 132,359 | 100 | 26.68 | 31.65 | 3,000/3,000 | 0 | 5 |
| WHERE chunk.n_words < 60 | 26,935 | 100 | 9.40 | 33.03 | 2,707/3,000 | 0 | 34 |
| WHERE chunk.n_words < 60, joined to page | 26,935 | 100 | 26.75 | 33.45 | 2,995/3,000 | 0 | 34 |
| WHERE chunk.n_words < 30 | 12,341 | 100 | 33.30 | 248.78 | 2,514/3,000 | 0 | 118 |
| WHERE chunk.n_words < 30, joined to page | 12,341 | 100 | 27.30 | 43.50 | 2,984/3,000 | 0 | 114 |

- **Bare.** The HNSW search. It reads `LIMIT` rows, and its top 5 is the exact one for 2,776
  claims at `ef_search` 20 and 2,917 at 100.
- **A predicate on `chunk` itself** (`n_words`, a column without an index). The server does not
  stop at `LIMIT`: it keeps taking rows from the index until `k` of them pass (35 and 120 rows read
  at the median at `ef_search` 20). None of the 12,000 answers came back short, unlike a filter
  applied after the search. But the index hands the rows over in its approximate order, so the
  result is often not the exact filtered top 5: 2,389 and 2,209 of 3,000 claims at `ef_search` 20,
  2,707 and 2,514 at 100. Each continuation also costs time: with 26,935 passing chunks the
  statement is still faster than the joined form (9.40 against 26.75 ms at `ef_search` 100), but
  with 12,341 it is slower (33.30 against 27.30 ms), with a p95 of 248.78 ms.
- **The same predicate with a join in the statement.** It takes about 27 ms in every shape,
  whatever passes, and gives the exact top 5 for 3,000 of 3,000 claims without the predicate and
  for 2,995 and 2,984 of 3,000 with it, at both `ef_search` values. On the demo corpus the joined
  form was exact for 75 of 75 claims in every shape (`results/statement_forms.md`). We did not
  look into the 5 and 16 lists that differ here.

So copying a filter column onto the vector table, without an index of its own, did not buy an
exact filtered search, and under the more selective predicate not a fast one either. With a join
in the statement MariaDB ranked exactly, or nearly so, at a cost that grows with the table (about
27 ms on S1) but not with the filter.

## The join order we fix, and the plan the optimizer would pick

The filtered statement names its tables with `STRAIGHT_JOIN`, `chunk` first (the reason is under
"Join order matters" above). `scripts/measure_index_behaviour.py join_order` runs every filter in
two forms, as written and with every `STRAIGHT_JOIN` replaced by `JOIN` ("free"), and compares
both with the same statement under `IGNORE INDEX (embedding)`, which reads and sorts every chunk.
The 200 claims of the fixed sample are the queries, at k = 5 and `ef_search` 100; each form was
timed alone, one untimed pass and one timed pass per claim.

The filter values follow one rule per database (`scripts/scaled_filters.py`,
[results/scaled/filters/](../results/scaled/filters/)): for page length, edit date, category and
linking page, the value whose passing chunks come nearest, on a log scale, to 50%, 5%, 0.5% and
0.05% of the chunk table. The four filters together take the 50% length and date values and the
category and linking page whose pair keeps the most chunks with them. The edit date is cut to the
second: on S1 the 5%, 0.5% and 0.05% levels all fall on 31 December 2020. No category or linking
page reaches 50%, so their loosest level is the largest there is (7.0% and 6.2% on S1, 11.8% and
3.8% on S0). On S0 that linking page is also the one nearest to 5%, so S0 has 17 cases and S1 18.

On S1 ([results/scaled/run1/s1/join_order.md](../results/scaled/run1/s1/join_order.md)); "same
rows / order" counts the claims, of 200, whose list holds the scan's chunks, and in the scan's
order. The free form gave the scan's list for 200 of 200 claims under every filter:

| Filter | Chunks passing | As written | Rows read | p50 ms | Same rows / order | Free | Rows read | p50 ms |
|---|---|---|---|---|---|---|---|---|
| none | 132,359 | vector index | 5 | 26.77 | 200 / 200 | vector index | 5 | 26.65 |
| length 50% | 66,171 | vector index | 17 | 26.68 | 200 / 199 | `page` first | 66,171 | 1,566.76 |
| length 5% | 6,616 | vector index | 158 | 37.09 | 200 / 199 | `page` first | 6,616 | 26.39 |
| length 0.5% | 626 | scan of `chunk` | 132,359 | 359.63 | 200 / 200 | `page` first | 626 | 2.82 |
| length 0.05% | 139 | scan of `chunk` | 132,359 | 335.33 | 200 / 200 | `page` first | 139 | 1.06 |
| edited 50% | 66,208 | vector index | 13 | 26.40 | 200 / 200 | `page` first | 66,208 | 1,689.04 |
| edited 5% | 6,599 | vector index | 132 | 34.41 | 200 / 200 | `page` first | 6,599 | 25.84 |
| edited 0.5% | 660 | vector index | 1,924 | 129.95 | 200 / 199 | `page` first | 660 | 3.04 |
| edited 0.05% | 77 | scan of `chunk` | 132,359 | 336.89 | 200 / 200 | `page` first | 77 | 0.86 |
| category 7.0% | 9,267 | vector index | 154 | 38.78 | 197 / 197 | derived table first | 9,267 | 36.12 |
| category 5% | 7,313 | vector index | 202 | 40.27 | 200 / 200 | derived table first | 7,313 | 29.64 |
| category 0.5% | 660 | vector index | 6,148 | 323.29 | 175 / 169 | derived table first | 660 | 2.95 |
| category 0.05% | 66 | vector index | 19,656 | 1,387.73 | 198 / 193 | derived table first | 66 | 0.99 |
| linked 6.2% | 8,203 | vector index | 170 | 40.06 | 200 / 200 | derived table first | 8,203 | 34.13 |
| linked 5% | 6,698 | vector index | 121 | 38.15 | 200 / 199 | derived table first | 6,698 | 27.97 |
| linked 0.5% | 662 | vector index | 1,888 | 142.39 | 193 / 193 | derived table first | 662 | 3.13 |
| linked 0.05% | 66 | vector index | 19,656 | 1,385.72 | 198 / 193 | derived table first | 66 | 0.96 |
| four filters | 3,107 | vector index | 440 | 54.47 | 197 / 197 | derived table first | 3,107 | 12.69 |

On S0 ([results/scaled/run1/s0/join_order.md](../results/scaled/run1/s0/join_order.md)), the same
200 claims; the free form again gave the scan's list for 200 of 200 claims under every filter:

| Filter | Chunks passing | As written | Rows read | p50 ms | Same rows / order | Free | Rows read | p50 ms |
|---|---|---|---|---|---|---|---|---|
| none | 43,134 | vector index | 5 | 16.16 | 200 / 200 | vector index | 5 | 16.11 |
| length 50% | 21,570 | vector index | 16 | 16.50 | 200 / 200 | `page` first | 21,570 | 98.72 |
| length 5% | 2,166 | vector index | 168 | 16.94 | 200 / 200 | `page` first | 2,166 | 8.46 |
| length 0.5% | 256 | scan of `chunk` | 43,134 | 307.35 | 200 / 200 | `page` first | 256 | 1.51 |
| length 0.05% | 139 | scan of `chunk` | 43,134 | 296.55 | 200 / 200 | `page` first | 139 | 1.07 |
| edited 50% | 21,578 | vector index | 13 | 16.33 | 200 / 200 | `page` first | 21,578 | 91.69 |
| edited 5% | 2,124 | vector index | 130 | 16.96 | 200 / 200 | `page` first | 2,124 | 8.20 |
| edited 0.5% | 219 | vector index | 1,552 | 24.97 | 198 / 198 | `page` first | 219 | 1.42 |
| edited 0.05% | 34 | scan of `chunk` | 43,134 | 275.00 | 200 / 200 | `page` first | 34 | 0.72 |
| category 11.8% | 5,081 | vector index | 84 | 17.21 | 199 / 199 | derived table first | 5,081 | 19.83 |
| category 5% | 1,857 | vector index | 174 | 18.01 | 200 / 200 | derived table first | 1,857 | 7.00 |
| category 0.5% | 215 | vector index | 4,482 | 52.76 | 192 / 189 | derived table first | 215 | 1.39 |
| category 0.05% | 22 | vector index | 12,133 | 145.22 | 200 / 200 | derived table first | 22 | 0.82 |
| linked 3.8% | 1,629 | vector index | 212 | 19.34 | 199 / 198 | derived table first | 1,629 | 6.57 |
| linked 0.5% | 216 | vector index | 2,098 | 36.33 | 198 / 198 | derived table first | 216 | 1.51 |
| linked 0.05% | 22 | vector index | 13,358 | 164.88 | 200 / 200 | derived table first | 22 | 0.91 |
| four filters | 824 | vector index | 446 | 21.45 | 199 / 198 | derived table first | 824 | 3.94 |

- **As written**, the server walks the vector index under 15 of the 18 filters on S1 (14 of 17 on
  S0), and reads more `chunk` rows the fewer pass. On S1: 5 without a filter, 13 and 17 at the 50%
  length and date levels, 121 to 202 at the 5% levels and the loosest category and link levels,
  1,888 to 6,148 at 0.5%, and 19,656 under both 0.05% levels that keep the index (we did not look
  into why the two medians are equal). The time follows: about 27 ms at the 50% length and date
  levels, 34 to 40 ms at 5%, 130 to 323 ms at 0.5%, and 1,386 to 1,388 ms when 66 chunks pass
  through the category or the link filter. Under the three most selective length and date filters
  the plan reads all 132,359 chunks and sorts them instead (`chunk` has type `ALL`), in 335 to
  360 ms. No list came back short, in either form, on either database.
- **Free**, the optimizer starts from another table under 17 of the 18 filters (16 of 17 on S0):
  `page`, or the derived table of the link or category filter. It reads `chunk` by
  `uq_chunk_page_ordinal`, exactly the chunks that pass, sorts them, and the vector index is not in
  the plan. Its cost follows the number that pass: 0.86 to 1.06 ms for 66 to 139 chunks, 2.82 to
  3.13 ms for 626 to 662, 12.69 ms for 3,107, 25.84 to 36.12 ms for 6,599 to 9,267, and 1,566.76
  and 1,689.04 ms for the 66,171 and 66,208 of the 50% length and date levels.
- **Each order is slow where the other is fast.** When 662 or fewer chunks pass, the free plan
  takes 0.86 to 3.13 ms and the fixed order 129.95 to 1,387.73 ms. When half the chunks pass, the
  free plan takes 1,566.76 and 1,689.04 ms and the fixed order 26.40 and 26.68 ms. Between them, at
  the 5% levels and the loosest category and link levels, the two stay within a factor of 1.5
  (25.84 to 36.12 against 34.41 to 40.27 ms). Under the four filters the free plan is faster,
  12.69 against 54.47 ms. On S0 the pattern is the same at a smaller scale: 0.72 to 1.51 ms free
  against 24.97 to 307.35 ms as written when 256 or fewer pass, and 91.69 and 98.72 ms free against
  16.33 and 16.50 ms when half pass.
- **The fixed order is not always exact under a filter.** Free, every list equals the scan's. As
  written, it holds the scan's chunks for 175 to 200 of 200 claims on S1, fewest under the
  0.5% category filter (175, and 169 in the scan's order), and for 192 to 200 on S0. Under the four
  filters on all 3,000 claims it is the exact top 5 for 2,962 ("The joined statement against a
  brute-force ranking" below). We did not look into which rows the walk misses.
- **Order.** As written, some lists hold the scan's chunks in another order (at most 6 of 200
  claims, under the 0.5% category filter on S1), and up to 7 lists per filter have rows that are
  not in ascending order of the distance the statement itself reports. Where such a list and the
  scan's name different chunks at a position, the two reported distances differ by at most 4.5e-06
  under 4 of the 7 filters where it happens on S1 and 2 of the 3 on S0. That fits the index ranking
  by its 16-bit copies of the vectors (its author's description, linked below) while the select
  list computes the distance from the stored 32-bit vector. Under the others (the 0.5% and 0.05%
  category filters and the 0.05% link filter on S1, the 0.5% category filter on S0) the gap is up
  to 2.6e-02 on S1 and 9.2e-03 on S0, which that explanation does not cover; we did not look into
  it. `wikilense` sorts the returned rows by the reported distance and the chunk id, so its order
  is the distance order.
- **The derived tables need their `DISTINCT` for the result, not for the index.** With every
  `SELECT DISTINCT` turned into `SELECT`, MariaDB merges the derived table into the join (no
  `<derived>` row in `EXPLAIN`) and still walks the vector index, for all 200 claims under every
  link, category and four-filter case, on both databases. But a page can link to another page
  several times, and the merged join then returns a chunk once per link row: on S1, 24, 135 and
  138 of 200 lists name a chunk twice under the 6.2%, 5% and 0.5% link filters, 147 under the four
  filters and none under the 0.05% link filter; on S0, 200 of 200 under the 0.05% link filter and
  none under the others. No category filter repeats a chunk.

`STRAIGHT_JOIN` therefore trades the best plan under a selective filter for a predictable one
under a loose filter. On the demo corpus the free plan's worst case was 10.24 ms against 2.11
(`results/join_order.md`). At 132,359 chunks it is 1,689.04 ms against 26.40, and the fixed
order's own worst case, 1,387.73 ms against 0.99, is now of the same size. A statement that counts
the passing chunks first and leaves the order free when few pass would be faster at that end. We
did not build it: the settings are fixed since the evaluation, and the count and the threshold
would need their own measurement.

## The same vectors, built several times

The `builds` part of `scripts/measure_index_behaviour.py` copies S1's 132,359 chunk vectors into a
scratch table that has the chunk table's vector column and index. It does this in `chunk_id`
order, in one transaction, twenty times for `M=16` and twenty times for the server default `M=6`.
Every build is then searched with the 3,000 claims (the bare statement, k = 5)
([results/scaled/run1/s1/build_variance.md](../results/scaled/run1/s1/build_variance.md)). One
build was timed alone first
([results/scaled/build_timing/s1/build_variance.md](../results/scaled/build_timing/s1/build_variance.md)).
The build took 85.6 s of a 717 s job. Most of the job was the fixed cost of exporting the vectors,
embedding the claims and ranking them exactly, so forty builds fit in one run, and the count is
twenty per `M`, as on the demo corpus. Each cell is the smallest to the largest value over the
twenty builds of that `M` (for "two builds agree", over the 190 pairs of builds). The build time
covers creating the table, the inserts and the commit:

| M | ef_search | Article recall@5, of 3,000 | Top-5 list equal to exact, of 3,000 | Two builds agree, of 3,000 | Build, s |
|---|---|---|---|---|---|
| 16 | 100 | 2,922 to 2,937 | 2,862 to 2,918 | 2,838 to 2,907 | 86.5 to 102.3 |
| 16 | 20 | 2,838 to 2,887 | 2,628 to 2,743 | 2,608 to 2,752 | 86.5 to 102.3 |
| 6 | 100 | 2,709 to 2,786 | 2,041 to 2,281 | 1,837 to 2,112 | 24.5 to 29.4 |
| 6 | 20 | 2,218 to 2,433 | 1,553 to 1,794 | 1,404 to 1,676 | 24.5 to 29.4 |

NumPy's exact ranking of the same vectors finds a gold page for 2,943 claims.

- **At the chosen settings (`M=16`, `ef_search` 100) recall moves little with the build, the hit
  lists more.** The twenty builds found a gold page for 2,922 to 2,937 of 3,000 claims, and none
  reached the exact ranking's 2,943. Their top-5 lists equalled the exact ones for 2,862 to 2,918
  claims, and two builds gave the same list for 2,838 to 2,907. The graph in the database is one
  more build: its bare search finds a gold page for 2,933
  ([results/scaled/eval/s1_full_run1.md](../results/scaled/eval/s1_full_run1.md)), inside that
  range, and the single timed build gave 2,928 and 2,869. On the demo corpus all twenty builds
  found 74 of 75 gold pages, the exact ranking's number (`results/build_variance.md`,
  `results/build_variance_run1.md`).
- **At `ef_search` 20 the same graphs differ more**: 2,628 to 2,743 lists equal to the exact ones,
  2,838 to 2,887 gold pages, and two builds agree on 2,608 to 2,752 lists. The larger `ef_search`
  also makes a result repeatable across builds.
- **With the server defaults (`M=6`, `ef_search` 20) the build decides hundreds of claims**: 2,218
  to 2,433 gold pages, 1,553 to 1,794 lists equal to the exact ones, and two builds agree on 1,404
  to 1,676 of 3,000 lists. At `ef_search` 100, `M=6` still finds fewer gold pages (2,709 to 2,786)
  than every `M=16` build at 20.
- **A build takes 86.5 to 102.3 s at `M=16` and 24.5 to 29.4 s at `M=6`**, for 132,359 inserts
  into an empty table, against 1.4 to 2.3 s and 0.4 to 0.6 s for the 4,598 vectors of the demo
  corpus (the same two files).
- The scratch table is filled in one transaction, in `chunk_id` order. We did not test whether
  the way a table is filled (batch size, commits between batches) changes the graph's quality.

## The index cache below and above the graph

`mhnsw_max_cache_size` bounds the memory the server keeps for a vector index's graph. Its default is
16 MB, and the Docker configuration sets 512 MB. The S1 graph is a file of 163,577,856 bytes
(156.0 MB; the hidden tablespace `chunk#i#04` in `information_schema.innodb_sys_tablespaces`).
`scripts/measure_index_behaviour.py cache` searched the 3,000 claims on S1 in two passes per cache
size, with the bare and the joined statement, k = 5 and `ef_search` 100, and compared each hit list
with NumPy's exact top 5
([results/scaled/run1/s1/cache_size.md](../results/scaled/run1/s1/cache_size.md)). The order was:
512 MB on the running server, 16 MB right after `SET GLOBAL` without reopening the table, then 256
down to 16 MB, each after `FLUSH TABLES chunk`, and 512 MB again at the end. The setting is global,
so nothing else ran on the server meanwhile, and it was 512 MB again afterwards. The times are of
the whole `search()` call.

| Cache | Bare: equal to exact, two passes | Bare: gold page found | Bare p50 ms | Joined: equal to exact, two passes | Joined: gold page found | Joined p50 ms |
|---|---|---|---|---|---|---|
| 512 MB, at the start | 2,917, 2,917 | 2,933 | 2.56 | 3,000, 3,000 | 2,943 | 28.43 |
| 256 MB | 2,902, 2,911 | 2,931 to 2,933 | 2.37 | 3,000, 3,000 | 2,943 | 27.20 |
| 192 MB | 2,902, 2,911 | 2,931 to 2,933 | 2.27 | 3,000, 3,000 | 2,943 | 27.53 |
| 160 MB | 2,819, 2,823 | 2,910 to 2,911 | 25.21 | 2,734, 2,725 | 2,840 | 351.45 |
| 128 MB | 2,722, 2,724 | 2,874 to 2,875 | 39.40 | 2,440, 2,436 | 2,743 to 2,750 | 493.34 |
| 64 MB | 2,561, 2,548 | 2,798 to 2,807 | 35.20 | 2,283, 2,275 | 2,685 to 2,690 | 166.27 |
| 16 MB | 2,163, 2,162 | 2,658 | 16.54 | 2,314, 2,312 | 2,690 to 2,695 | 21.92 |
| 16 MB, table not reopened | 2,151, 2,162 | 2,657 to 2,658 | 17.71 | 2,314, 2,311 | 2,691 to 2,694 | 22.00 |
| 512 MB, at the end | 2,902, 2,911 | 2,931 to 2,933 | 2.14 | 3,000, 3,000 | 2,943 | 27.31 |

All 3,000 claims got 5 rows at every size; the size changed which rows. The exact ranking finds a
gold page for 2,943 claims.

- **The cache has to be larger than the graph's file, by a margin.** At 192 MB and more, every
  count equals the one at 512 MB. At 160 MB, 2.6% above the 156.0 MB file, the joined statement is
  exact for 2,734 and 2,725 claims instead of 3,000, finds a gold page for 2,840 instead of 2,943,
  and takes 351.45 ms instead of 27.53. We did not measure how much memory the cache needs per byte
  of the file.
- **Below that, the search is less exact, slower and not repeatable.** From 160 down to 64 MB the
  bare statement takes 25.21 to 39.40 ms and the joined one 166.27 to 493.34 ms. The two passes
  give the same hit list for as few as 2,291 of 3,000 claims (joined, 64 MB;
  "Two passes agree" in the results file).
- **At the 16 MB default the search is fast again but the least exact.** The bare statement finds
  a gold page for 2,658 of 3,000 claims (0.886) instead of 2,933 (0.978), at 16.54 ms. We did not
  look into why it is faster here than at 64 to 160 MB.
- **A new size applies at once.** The 16 MB row measured right after `SET GLOBAL`, without
  reopening the table, gives the same counts as the one measured after `FLUSH TABLES chunk`, to
  within 12 claims.
- **Right after the table is reopened, the bare search moves a little.** With a cache that holds
  the graph, the bare statement is exact for 2,902 claims in the first pass and 2,911 in the second,
  against 2,917 in both passes on the running server before, and 10 claims change their hit list
  between the two passes. The joined statement stays exact for all 3,000. We did not look into why.

On the demo corpus the same holds at a smaller scale: with 8 MB or less for its 13.0 MB graph the
hit lists were exact for only 57 to 67 of 75 claims, bare or joined (`results/cache_size.md`). So
size `mhnsw_max_cache_size` with room above the graph's file: 512 MB holds S1's graph three times
over. [MDEV-41649](https://jira.mariadb.org/browse/MDEV-41649) asks MariaDB for a larger default, for speed; the loss of recall
below the graph's size is not in it.

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
  the index before B's commit and again after it. None of the controls that lacked one of the two
  went stale (A at READ COMMITTED, in autocommit mode, without the second search, or with a
  `COUNT(*)` or a scan as its first statement). That holds for this table only. In a plain-SQL repro
  with 8-dimensional vectors on 11.8.9 and 13.0.2, a transaction whose first statement was
  `COUNT(*)`, a scan or `START TRANSACTION WITH CONSISTENT SNAPSHOT`, and that searched the index
  once after B's commit, left it stale in 3 of 3 trials. A 384-dimensional rerun of that repro
  stayed correct for `COUNT(*)` and the scan (0 of 5 stale) but went stale in 2 of 5 trials for
  `CONSISTENT SNAPSHOT`. (In 1 control trial the index found 4 of the 5 rows: an approximate
  search, not this state.) The last control shows that A's disconnect is not needed: the index was
  stale when A committed and stayed connected.
- **Who gets the wrong answer.** Everyone we tried: on 5 stale tables, new connections at REPEATABLE
  READ, READ COMMITTED and READ UNCOMMITTED and in autocommit mode each found 0 of the 5 rows
  through the index, so the isolation level of the connection that asks does not protect it.
- **It lasts.** Three further new connections each found 0 of the 5 rows, on six stale tables.
- **What ends it.** The next committed write to the table (then the index found all 6 of 6 new
  rows, 3 of 3 tables) or `FLUSH TABLES` on the table (5 of 5, 3 of 3 tables). `FLUSH TABLES`
  makes the server open the table again, so the index as stored was right. That fits a copy of
  the graph in the server's memory that was loaded inside A's old snapshot; we did not inspect
  the cache or read the server source.
- **With a `DELETE` in step 2** (B deletes the three nearest rows), the index then returned no
  row at all for that query in 10 of 10 trials, from a table of 997 rows where a scan returned
  10. Of 20 other queries none came back empty, but the shortest answer had 5 rows. A committed
  insert (3 of 3 tables) or `FLUSH TABLES` (once) brought the 10 rows back, and the control
  without A's second search returned 10 rows in 5 of 5.


What to do when other connections write to the table: no index search in a REPEATABLE READ
transaction whose snapshot predates another connection's commit. Searching at READ COMMITTED or
in autocommit mode did not cause the state. PyMySQL opens connections with autocommit off, so a
connection that has run one `SELECT` is in a REPEATABLE READ transaction until it commits, rolls
back or closes. That is how WikiLense's own connections work. A web request runs one search and
closes; with "repeat" it rolls back between the runs, so each search has a transaction of its own. `wikilense query` runs a `COUNT(*)`, loads the model, runs one search and closes, which is
the shape of the `COUNT(*)` control. That shape stayed correct on this table but not in the
8-dimensional repro. `wikilense query` stays safe because no other connection commits a write to
the corpus while it runs. `wikilense eval` and the experiment scripts run many searches in one
transaction, so a committed write to the corpus by another connection during such a run would
start the sequence. Nothing here does that: only an ingest (`wikilense ingest`, or the one inside
an experiment group) commits writes to the corpus, and it recreates the tables; `sql/examples.sql`
updates one row and rolls back. The measurements are not affected. The experiment was run three
times with the same outcome in every row of the sequence table that the runs share
([results/isolation_run1.md](../results/isolation_run1.md) is the first run, made before three of
the controls were added). The behaviour is MariaDB issue
[MDEV-41321](https://jira.mariadb.org/browse/MDEV-41321). The tables above were measured on 11.8.9
only, on this one table.

## The joined statement against a brute-force ranking, and a copy of the vectors

"The server ranks exactly" above rests on the hit lists of the joined statement not changing with
`mhnsw_ef_search`. [scripts/compare_separate_index.py](../scripts/compare_separate_index.py) checks
it directly: it reads all vectors out of MariaDB, ranks them for every claim with NumPy in float64,
and compares that list with the statement's. The same copy, held in the application's memory, is
the simplest separate index there is, so the script also times it. On S1
([results/scaled/run1/s1/separate_index.md](../results/scaled/run1/s1/separate_index.md); 3,000
claims, k = 5, the methods taking turns query by query, one timed pass):

| Method | No filter: p50 ms | same list as exact | History (5,363 pass): p50 ms | same list | Four filters (3,107 pass): p50 ms | same list |
|---|---|---|---|---|---|---|
| MariaDB, bare index search (`ef_search` 100) | 2.87 | 2,917 | - | - | - | - |
| MariaDB, joined statement (`inline`) | 27.30 | 3,000 | 37.36 | 2,996 | 58.69 | 2,962 |
| MariaDB, `overfetch 10` | - | - | 3.27 (2,424 short) | 568 | 3.94 (2,801 short) | 198 |
| NumPy scan of the copy, ids only | 10.19 | 3,000 | 16.01 | 3,000 | 1.69 | 3,000 |
| the same, then the hit rows from MariaDB | 11.06 | 3,000 | 16.94 | 3,000 | 2.44 | 3,000 |
| hnswlib graph over the copy (`M=16`, `ef` 100), ids only | 0.36 | 2,544 | 19.01 | 2,981 | 11.73 | 2,990 |

- **The joined statement is the exact ranking without a filter**: the same 5 chunks in the same
  order as NumPy for 3,000 of 3,000 claims, on S1 as on S0
  ([results/scaled/run1/s0/separate_index.md](../results/scaled/run1/s0/separate_index.md)). The
  statement forms above give the same (3,000 of 3,000 joined to `page`).
- **Under a filter it is nearly exact.** These are the lists of `search()`, which sorts the rows it
  gets by the reported distance and the chunk id. They equal the exact ranking of the chunks that
  pass for 2,996 of 3,000 claims under the History heading and for 2,962 under the four metadata
  filters, where the mean overlap with the exact top 5 is 0.998. On S0 they are 2,998 and 2,993
  (1,588 and 824 chunks pass). On the demo corpus they were exact for 75 of 75 claims under both
  (`results/separate_index.md`). We did not look into the 38 lists.
- **The copy.** 132,359 vectors are 203,303,424 bytes of float32; reading them out of MariaDB with
  the columns the filters need took 2.42 s, and the client process grew by 701,804,544 bytes. The
  hnswlib graph is 222,955,496 bytes saved and took 5.87 s to build on all cores. A copy is stale
  after the next write and outside every transaction.
- **Speed.** Without a filter, MariaDB's index search (2.87 ms) is faster than an exact scan of the
  copy (10.19 ms); an in-process HNSW graph is faster still (0.36 ms) but finds less of the exact
  top 5 (0.961 against 0.993). Under the four filters the copy is far faster: 2.44 ms with the row
  fetch against 58.69 ms for MariaDB's joined statement (p95 189.01 ms). The copy's filtered times
  include evaluating the filter in Python inside every query (for the History heading, a match over
  the 132,359 headings), so they are an upper bound for a separate index with its own filter index.

## A search that runs away

**The setup.** The statements of the bullet above ran once more in throwaway containers of the
official images, each with a 2 GB memory cap
([results/scaled/version_probe/version_probe.md](../results/scaled/version_probe/version_probe.md),
with its script and logs):
- a table of the 132,359 S1 vectors;
- one claim's query vector;
- `LIMIT` 10.

**On 11.8.9.**
- **The first run after a server start was fast.** A distance threshold that 3 rows meet returned
  them in 1.51 and 1.71 s.
- **The next run was killed.** It grew `mariadbd` to the 2 GB cap in 20 to 22 s, and the kernel
  killed the server. The logs record this for that threshold, on the same connection and on a
  fresh one, and once for `WHERE w < x` with 5 rows passing.
- **With random vectors** the same statements took up to 16.55 s and 1,311 MiB, came back short,
  and gave the memory back.
- **With `max_statement_time` = 10** the threshold statement was stopped at 10.0 s (error 1969),
  at 1.3 and 1.5 GB, and the server stayed up.

**On 12.3.3 and 13.0.2.**
- Nothing ran away: memory peaked at 552 MiB or less.
- The threshold statement returned its 3 rows every time: 6.85 and 5.47 s on the first run, then
  0.63 to 1.69 s.
- `WHERE w < x` returned 0 or 1 of the 5 rows that pass. That is the short answer the
  documentation describes.

**Status.** We found no MariaDB ticket for this; we searched JIRA on 2026-11-20. Its nearest
relatives are [MDEV-33412](https://jira.mariadb.org/browse/MDEV-33412) and
[MDEV-35032](https://jira.mariadb.org/browse/MDEV-35032).

**What to do on 11.8.9.**
- Give vector searches a `max_statement_time`.
- Size the server's memory with room.
- Filter through a join: in our measurements the join stopped none of 120 executions, where the
  predicate on the indexed table itself stopped 36.

## MariaDB and a dedicated vector store, from their documentation

What the README's "Where this approach loses" item 4 summarises, with the sources:

- MariaDB documents that "there can be only one vector index in the table"
  ([CREATE TABLE with Vectors](https://mariadb.com/docs/server/reference/sql-structure/vectors/create-table-with-vectors)),
  and two distances with no dot product
  ([Vector Overview](https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-overview)).
- Its author writes that `ef_construction` is fixed at 10 and that the index always stores 16-bit
  integers ([How it works, part II](https://mariadb.org/mariadb-vector-how-it-works-part-ii/),
  [part III](https://mariadb.org/mariadb-vector-how-it-works-part-iii/)).
- Qdrant documents several named vectors per point
  ([Vectors](https://qdrant.tech/documentation/manage-data/vectors/)), a configurable `ef_construct`
  and an HNSW graph extended "with additional edges based on indexed payload values", which lets it
  "apply filters as you search in the graph"
  ([Indexing](https://qdrant.tech/documentation/manage-data/indexing/)), three kinds of optional
  quantization ([Quantization](https://qdrant.tech/documentation/manage-data/quantization/)) and
  sharding across nodes
  ([Distributed deployment](https://qdrant.tech/documentation/scaling/distributed_deployment/)).

We did not measure Qdrant or any other dedicated vector store.

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
| "The `LIMIT` is a hard cap on how many rows the index returns *before* the threshold is applied", and "(Adding the `WHERE` directly to an `ORDER BY ... LIMIT` query also uses the index, but with the same hard cap.)" (read on 2026-11-20) | holds for the subquery form, which reads exactly its inner `LIMIT` rows. Not on 11.8.9 for a `WHERE` inside the statement: the search goes on until `LIMIT` rows pass, and when fewer qualify it runs until it is stopped or out of memory ("A search that runs away" above). On 12.3.3 and 13.0.2 a very selective filter comes back short instead |

Not in the reference, and measured here: the second `ORDER BY` key and the `IN (subquery)` that
lose the index; the exact ranking of the joined statement without a filter, and its nearly exact
one under a filter; the approximate, continuing search under a predicate on the indexed table
itself; the optimizer's switch to a scan under selective range filters, and its choice of the
filter's own table over the vector index when the join order is left free; the hit lists that
change with the graph build (and how much, over twenty builds per `M` on S1) and, for `M=6`,
across a restart; the loss of
exactness and of repeatability when the graph does not fit in `mhnsw_max_cache_size`; the stale
index after a REPEATABLE READ reader; a search that fewer rows pass than `LIMIT`, which runs away on
11.8.9.

We searched MariaDB's JIRA for these on 2026-11-20. The stale index is
[MDEV-41321](https://jira.mariadb.org/browse/MDEV-41321), and the `NULL` distances are [MDEV-35192](https://jira.mariadb.org/browse/MDEV-35192).
[MDEV-35768](https://jira.mariadb.org/browse/MDEV-35768) fixed the lost index for views and derived tables in 11.7.2.
We found no ticket for these three:

- the semi-join of `IN (subquery)`;
- the tie-breaking `ORDER BY` key;
- the lower recall over uncommitted rows.

[MDEV-36869](https://jira.mariadb.org/browse/MDEV-36869), closed as not a bug, says the graph's cache cannot reproduce InnoDB's
transactional semantics exactly.

Two facts about the index come from its author's description, not from our measurements
([MariaDB Vector: How it works, part II](https://mariadb.org/mariadb-vector-how-it-works-part-ii/)
and [part III](https://mariadb.org/mariadb-vector-how-it-works-part-iii/)): the graph stores each
vector as 16-bit integers with a scale factor, and `ef_construction` is fixed at 10.
