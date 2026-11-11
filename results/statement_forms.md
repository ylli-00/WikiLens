# One nearest-neighbour statement in three shapes

Generated 2026-11-11T04:12:36+00:00 by `scripts/measure_index_behaviour.py forms` on MariaDB 11.8.9-MariaDB-ubu2404: 4,598 chunks, index M = 16, the 75 corpus claims as queries, k = 10, 3 timed passes after one untimed pass, the forms taking turns query by query. "Equal to exact" counts the claims whose hit list is, in order, NumPy's exact top 10 of the chunks that pass the predicate. "Rows read" is the median over the claims of `r_rows` for `chunk` in `ANALYZE`.

| Statement | Chunks passing | ef_search | p50 ms | p95 ms | Equal to exact | Short | Rows read |
|---|---|---|---|---|---|---|---|
| the bare statement on chunk | 4,598 | 20 | 0.27 | 0.50 | 75/75 | 0/75 | 10 |
| the same, joined to page | 4,598 | 20 | 2.01 | 2.29 | 75/75 | 0/75 | 10 |
| WHERE chunk.n_words < 60 | 766 | 20 | 0.98 | 3.08 | 61/75 | 0/75 | 88 |
| WHERE chunk.n_words < 60, joined to page | 766 | 20 | 2.08 | 2.32 | 75/75 | 0/75 | 84 |
| WHERE chunk.n_words < 30 | 304 | 20 | 3.95 | 12.85 | 40/75 | 0/75 | 276 |
| WHERE chunk.n_words < 30, joined to page | 304 | 20 | 2.25 | 2.70 | 75/75 | 0/75 | 270 |
| the bare statement on chunk | 4,598 | 100 | 0.65 | 0.94 | 75/75 | 0/75 | 10 |
| the same, joined to page | 4,598 | 100 | 2.27 | 2.94 | 75/75 | 0/75 | 10 |
| WHERE chunk.n_words < 60 | 766 | 100 | 2.02 | 4.97 | 68/75 | 0/75 | 86 |
| WHERE chunk.n_words < 60, joined to page | 766 | 100 | 2.32 | 3.02 | 75/75 | 0/75 | 84 |
| WHERE chunk.n_words < 30 | 304 | 100 | 6.46 | 19.18 | 50/75 | 0/75 | 277 |
| WHERE chunk.n_words < 30, joined to page | 304 | 100 | 2.56 | 3.23 | 75/75 | 0/75 | 270 |

How to read it:

- The bare statement is the HNSW search: it reads `LIMIT` rows, and `mhnsw_ef_search` decides how often it equals the exact ranking.
- A predicate on the indexed table itself does not stop that search at `LIMIT`: the server keeps asking the index for more rows until `k` of them pass. No query comes back short. The result stays approximate, more so the fewer rows pass, and a larger `mhnsw_ef_search` helps only in part. It also gets slower than the joined form.
- With a join in the statement the same predicate gives the exact answer for every claim, at both `mhnsw_ef_search` values, at a cost that does not depend on them.

The statements (the word limit, the query vector and `k` are bound parameters):

```sql
SELECT chunk_id FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s;
SELECT chunk_id FROM chunk WHERE n_words < %s ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id WHERE chunk.n_words < %s ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s;
```
