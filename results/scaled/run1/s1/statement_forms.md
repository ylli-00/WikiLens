# One nearest-neighbour statement in three shapes

Generated 2026-11-19T23:11:54+00:00 by `scripts/measure_index_behaviour.py forms` on MariaDB 11.8.9-MariaDB-ubu2404: 132,359 chunks, index M = 16, the 3000 corpus claims as queries, k = 5, 1 timed passes after one untimed pass, the forms taking turns query by query. "Equal to exact" counts the claims whose hit list is, in order, NumPy's exact top 5 of the chunks that pass the predicate. "Rows read" is the median over the claims of `r_rows` for `chunk` in `ANALYZE`. "Chunk access" is the type and key of `chunk` in `EXPLAIN`. The `mhnsw_ef_search` values were run one after the other, so compare times inside one value.

| Statement | Chunks passing | ef_search | p50 ms | p95 ms | Equal to exact | Short | Rows read | Chunk access |
|---|---|---|---|---|---|---|---|---|
| the bare statement on chunk | 132,359 | 20 | 1.13 | 2.94 | 2776/3000 | 0/3000 | 5 | index embedding |
| the same, joined to page | 132,359 | 20 | 26.97 | 31.89 | 3000/3000 | 0/3000 | 5 | index embedding |
| WHERE chunk.n_words < 60 | 26,935 | 20 | 4.79 | 19.23 | 2389/3000 | 0/3000 | 35 | index embedding |
| WHERE chunk.n_words < 60, joined to page | 26,935 | 20 | 27.14 | 33.50 | 2995/3000 | 0/3000 | 34 | index embedding |
| WHERE chunk.n_words < 30 | 12,341 | 20 | 20.82 | 179.15 | 2209/3000 | 0/3000 | 120 | index embedding |
| WHERE chunk.n_words < 30, joined to page | 12,341 | 20 | 27.59 | 43.25 | 2984/3000 | 0/3000 | 114 | index embedding |
| the bare statement on chunk | 132,359 | 100 | 2.16 | 4.39 | 2917/3000 | 0/3000 | 5 | index embedding |
| the same, joined to page | 132,359 | 100 | 26.68 | 31.65 | 3000/3000 | 0/3000 | 5 | index embedding |
| WHERE chunk.n_words < 60 | 26,935 | 100 | 9.40 | 33.03 | 2707/3000 | 0/3000 | 34 | index embedding |
| WHERE chunk.n_words < 60, joined to page | 26,935 | 100 | 26.75 | 33.45 | 2995/3000 | 0/3000 | 34 | index embedding |
| WHERE chunk.n_words < 30 | 12,341 | 100 | 33.30 | 248.78 | 2514/3000 | 0/3000 | 118 | index embedding |
| WHERE chunk.n_words < 30, joined to page | 12,341 | 100 | 27.30 | 43.50 | 2984/3000 | 0/3000 | 114 | index embedding |

How to read it:

- The bare statement is the HNSW search: it reads `LIMIT` rows. On the graph now in the database its list was the exact one for 2776 at `mhnsw_ef_search` 20, 2917 at `mhnsw_ef_search` 100 of 3000 claims. Other builds of the graph differ (the `builds` part measures by how much).
- A predicate on the indexed table itself does not stop that search at `LIMIT`: the server keeps taking rows from the index until `k` of them pass. 0 of the 12000 answers were short (with 26,935 and 12,341 chunks passing; nothing more selective was run). The list was the exact one for 2389 and 2209 at `mhnsw_ef_search` 20, 2707 and 2514 at `mhnsw_ef_search` 100 of 3000 claims, in the order of the table.
- With a join in the statement the list was the exact one for 3000 and 2995 and 2984 at `mhnsw_ef_search` 20, 3000 and 2995 and 2984 at `mhnsw_ef_search` 100 of 3000 claims, in the order of the table.

The statements (the word limit, the query vector and `k` are bound parameters):

```sql
SELECT chunk_id FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s;
SELECT chunk_id FROM chunk WHERE n_words < %s ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id WHERE chunk.n_words < %s ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s;
```
