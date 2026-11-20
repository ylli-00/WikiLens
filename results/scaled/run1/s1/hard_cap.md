# The LIMIT of an index search, under a distance threshold and a selective predicate

Generated 2026-11-20T16:53:18+00:00 by `scripts/measure_index_behaviour.py hard_cap` on MariaDB 11.8.9-MariaDB-ubu2404 at port 3308: 132,359 chunks, index M = 16, a random sample of 30 of the 3,000 corpus claims as queries (`random.Random(20261119)`), k = 10, `mhnsw_ef_search` 100, a time limit of 10 s on every statement (`max_statement_time`). The cases ran one after the other; for each claim the statement ran once untimed and then 1 time timed. MariaDB's documentation ("Vector Overview", "Using the Vector Index Efficiently") says: "The LIMIT is a hard cap on how many rows the index returns before the threshold is applied." A claim's threshold lies halfway between its n-th and (n+1)-th nearest chunk in NumPy's exact cosine ranking of all chunk vectors, so exactly n chunks are nearer than it; "Claims" counts the claims that have one (see the last bullet). "Expected" is the number of rows of an exact answer: min(k, n), min(L, n, k) with an inner `LIMIT` L, min(k, chunks passing) for a predicate. "Complete", "Short", "Stopped" and "Server lost" count the executions, 2 per claim: with the expected number of rows, with fewer, stopped at the time limit (error 1969), and with the connection to the server lost, after which the part waited for the server, reconnected and went on with the next claim. "Rows returned" lists each number of rows with the claims that got it in brackets, from each claim's first execution that returned rows; of those claims, "Equal to exact" counts the ones whose list is, in order, the start of NumPy's exact ranking of the chunks that qualify, "Same set" those with the same chunks in any order. "Rows read" is the median (and the largest) of the `chunk` rows read, `r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`, over the claims whose executions were neither stopped nor lost (`ANALYZE` runs the statement once more). "Chunk access" is the type and key of `chunk` in `EXPLAIN`, which does not run the statement, over every claim; `index embedding` is the vector index. "Timed" is the number of timed executions that came back complete or short, the ones p50 and p95 are over; the times are of the statement and the fetch of its rows.

## The threshold inside the statement

| n | Claims | Expected | Complete | Short | Stopped | Server lost | Rows returned | Equal to exact | Same set | Rows read (largest) | Chunk access | Timed | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 30 | 3 | 3 | 0 | 57 | 0 | 3 (3) | 3/3 | 3/3 | - | index embedding | 0 | - | - |
| 10 | 30 | 10 | 54 | 0 | 6 | 0 | 10 (27) | 27/27 | 27/27 | 10 (10) | index embedding | 27 | 1.18 | 2.19 |
| 30 | 30 | 10 | 60 | 0 | 0 | 0 | 10 (30) | 27/30 | 27/30 | 10 (10) | index embedding | 30 | 1.08 | 1.55 |

## The index search as a derived table, the threshold outside it

| Inner LIMIT | n | Claims | Expected | Complete | Short | Stopped | Server lost | Rows returned | Equal to exact | Same set | Rows read (largest) | Chunk access | Timed | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 3 | 30 | 3 | 58 | 2 | 0 | 0 | 3 (29), 2 (1) | 29/30 | 29/30 | 10 (10) | index embedding | 30 | 1.04 | 1.61 |
| 10 | 10 | 30 | 10 | 54 | 6 | 0 | 0 | 10 (27), 9 (2), 8 (1) | 27/30 | 27/30 | 10 (10) | index embedding | 30 | 0.93 | 1.55 |
| 10 | 30 | 30 | 10 | 60 | 0 | 0 | 0 | 10 (30) | 27/30 | 27/30 | 10 (10) | index embedding | 30 | 0.90 | 1.60 |
| 100 | 3 | 30 | 3 | 58 | 2 | 0 | 0 | 3 (29), 2 (1) | 29/30 | 29/30 | 100 (100) | index embedding | 30 | 1.32 | 2.15 |
| 100 | 10 | 30 | 10 | 54 | 6 | 0 | 0 | 10 (27), 9 (2), 8 (1) | 27/30 | 27/30 | 100 (100) | index embedding | 30 | 1.02 | 1.61 |
| 100 | 30 | 30 | 10 | 60 | 0 | 0 | 0 | 10 (30) | 27/30 | 27/30 | 100 (100) | index embedding | 30 | 1.03 | 1.73 |

## A selective predicate on `chunk`

| Statement | Chunks passing | Claims | Expected | Complete | Short | Stopped | Server lost | Rows returned | Equal to exact | Same set | Rows read (largest) | Chunk access | Timed | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WHERE chunk.n_words < 4 | 694 | 30 | 10 | 56 | 0 | 4 | 0 | 10 (28) | 17/28 | 17/28 | 3344 (7609) | index embedding | 28 | 1357.45 | 5260.05 |
| WHERE chunk.n_words < 4, joined to page | 694 | 30 | 10 | 60 | 0 | 0 | 0 | 10 (30) | 27/30 | 27/30 | 3354 (14451) | index embedding | 30 | 31.57 | 644.57 |
| WHERE chunk.n_words < 2 | 133 | 30 | 10 | 28 | 0 | 32 | 0 | 10 (14) | 9/14 | 9/14 | 5246 (12237) | index embedding | 14 | 2389.48 | 7500.17 |
| WHERE chunk.n_words < 2, joined to page | 133 | 30 | 10 | 60 | 0 | 0 | 0 | 10 (30) | 29/30 | 30/30 | 11811 (32688) | index embedding | 30 | 654.14 | 1929.98 |

How to read it:

- Threshold inside the statement: `EXPLAIN` gives `chunk` `index embedding` for every claim. Chunk rows read, median (largest): - at n = 3, 10 (10) at n = 10, 10 (10) at n = 30. Of the claims with their rows read measured, none read more than `LIMIT` = 10 chunk rows (n = 10, n = 30). At n = 3 no claim has its rows read measured: each had an execution stopped at the time limit or a lost server, and `ANALYZE` was not run for it. Short answers, of the executions: 0/60 at n = 3, 0/60 at n = 10, 0/60 at n = 30; equal to the exact list, of the claims with rows: 3/3 at n = 3, 27/27 at n = 10, 27/30 at n = 30.
- Derived table: the outer `WHERE` sees the inner statement's `LIMIT` L rows only, so an exact answer has min(L, n, k) rows. `EXPLAIN` gives `chunk` `index embedding` for every claim. Chunk rows read, median (largest): 10 (10) at L = 10, n = 3, 10 (10) at L = 10, n = 10, 10 (10) at L = 10, n = 30, 100 (100) at L = 100, n = 3, 100 (100) at L = 100, n = 10, 100 (100) at L = 100, n = 30. Short answers, of the executions: 2/60 at L = 10, n = 3, 6/60 at L = 10, n = 10, 0/60 at L = 10, n = 30, 2/60 at L = 100, n = 3, 6/60 at L = 100, n = 10, 0/60 at L = 100, n = 30; equal to the exact list, of the claims with rows: 29/30 at L = 10, n = 3, 27/30 at L = 10, n = 10, 27/30 at L = 10, n = 30, 29/30 at L = 100, n = 3, 27/30 at L = 100, n = 10, 27/30 at L = 100, n = 30.
- Predicate on `chunk`, the word limits nearest 0.5% and 0.05% of the chunks on a log scale (the rule of `scripts/scaled_filters.py`; 4 words: 694 chunks (0.52%, aimed at 0.50%), 2 words: 133 chunks (0.10%, aimed at 0.05%)): `EXPLAIN` gives `chunk` `index embedding` for every claim. 0 of the 240 executions came back short. Chunk rows read, median (largest): 3344 (7609) at WHERE chunk.n_words < 4, 3354 (14451) at WHERE chunk.n_words < 4, joined to page, 5246 (12237) at WHERE chunk.n_words < 2, 11811 (32688) at WHERE chunk.n_words < 2, joined to page. The server read more than `LIMIT` = 10 chunk rows for WHERE chunk.n_words < 4, WHERE chunk.n_words < 4, joined to page, WHERE chunk.n_words < 2, WHERE chunk.n_words < 2, joined to page. Equal to the exact list of the chunks that pass, of the claims with rows: 17/28 at WHERE chunk.n_words < 4, 27/30 at WHERE chunk.n_words < 4, joined to page, 9/14 at WHERE chunk.n_words < 2, 29/30 at WHERE chunk.n_words < 2, joined to page.
- Stopped at the time limit of 10 s: 99 of the 780 executions (57 of 60 for WHERE distance < threshold, n = 3, 6 of 60 for WHERE distance < threshold, n = 10, 4 of 60 for WHERE chunk.n_words < 4, 32 of 60 for WHERE chunk.n_words < 2), `ANALYZE` for 0 claims and the scan check for 0 claims. A stopped execution has neither rows nor a time, and its claim has no rows read.
- The connection to the server was never lost.
- The thresholds: the server's own scan without the index (`IGNORE INDEX (embedding)`) found exactly n chunks under the threshold for 30 of 30 claims at n = 3, 30 of 30 claims at n = 10, 30 of 30 claims at n = 30. Left out, because the n-th and (n+1)-th exact distances were at most 1e-06 apart: 0 at n = 3, 0 at n = 10, 0 at n = 30. The largest difference between a distance the server computed and NumPy's was 7.6e-08.

The statements (the vector, the threshold, the limits and the word limit are bound parameters), each after the session's time limit:

```sql
SET SESSION max_statement_time = 10;
SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance FROM chunk WHERE VEC_DISTANCE_COSINE(embedding, %s) < %s ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
SELECT t.chunk_id, t.distance FROM (SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s) AS t WHERE t.distance < %s ORDER BY t.distance LIMIT %s;
SELECT chunk_id FROM chunk WHERE n_words < %s ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id WHERE chunk.n_words < %s ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s;
SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance FROM chunk IGNORE INDEX (embedding) WHERE VEC_DISTANCE_COSINE(embedding, %s) < %s;
```
