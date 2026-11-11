# One nearest-neighbour statement in three shapes

Generated 2026-11-11T08:16:16+00:00 by `scripts/measure_index_behaviour.py forms` on MariaDB 11.8.9-MariaDB-ubu2404: 4,598 chunks, index M = 16, the 75 corpus claims as queries, k = 10, 3 timed passes after one untimed pass, the forms taking turns query by query. "Equal to exact" counts the claims whose hit list is, in order, NumPy's exact top 10 of the chunks that pass the predicate. "Rows read" is the median over the claims of `r_rows` for `chunk` in `ANALYZE`. "Chunk access" is the type and key of `chunk` in `EXPLAIN`. The `mhnsw_ef_search` values were run one after the other, so compare times inside one value.

| Statement | Chunks passing | ef_search | p50 ms | p95 ms | Equal to exact | Short | Rows read | Chunk access |
|---|---|---|---|---|---|---|---|---|
| the bare statement on chunk | 4,598 | 20 | 0.36 | 0.77 | 75/75 | 0/75 | 10 | index embedding |
| the same, joined to page | 4,598 | 20 | 2.16 | 3.26 | 75/75 | 0/75 | 10 | index embedding |
| WHERE chunk.n_words < 60 | 766 | 20 | 1.12 | 3.43 | 61/75 | 0/75 | 88 | index embedding |
| WHERE chunk.n_words < 60, joined to page | 766 | 20 | 2.25 | 3.48 | 75/75 | 0/75 | 84 | index embedding |
| WHERE chunk.n_words < 30 | 304 | 20 | 3.99 | 13.72 | 41/75 | 0/75 | 276 | index embedding |
| WHERE chunk.n_words < 30, joined to page | 304 | 20 | 2.45 | 3.77 | 75/75 | 0/75 | 270 | index embedding |
| the bare statement on chunk | 4,598 | 100 | 0.76 | 1.06 | 75/75 | 0/75 | 10 | index embedding |
| the same, joined to page | 4,598 | 100 | 2.42 | 3.38 | 75/75 | 0/75 | 10 | index embedding |
| WHERE chunk.n_words < 60 | 766 | 100 | 2.10 | 4.97 | 68/75 | 0/75 | 86 | index embedding |
| WHERE chunk.n_words < 60, joined to page | 766 | 100 | 2.46 | 3.23 | 75/75 | 0/75 | 84 | index embedding |
| WHERE chunk.n_words < 30 | 304 | 100 | 6.71 | 18.46 | 51/75 | 0/75 | 277 | index embedding |
| WHERE chunk.n_words < 30, joined to page | 304 | 100 | 2.68 | 3.39 | 75/75 | 0/75 | 270 | index embedding |

How to read it:

- The bare statement is the HNSW search: it reads `LIMIT` rows. On the graph now in the database its list was the exact one for 75 at `mhnsw_ef_search` 20, 75 at `mhnsw_ef_search` 100 of 75 claims. Other builds of the graph differ (`results/build_variance.md`).
- A predicate on the indexed table itself does not stop that search at `LIMIT`: the server keeps taking rows from the index until `k` of them pass. 0 of the 300 answers were short (with 766 and 304 chunks passing; nothing more selective was run). The list was the exact one for 61 and 41 at `mhnsw_ef_search` 20, 68 and 51 at `mhnsw_ef_search` 100 of 75 claims, in the order of the table.
- With a join in the statement the list was the exact one for 75 and 75 and 75 at `mhnsw_ef_search` 20, 75 and 75 and 75 at `mhnsw_ef_search` 100 of 75 claims, in the order of the table.

The statements (the word limit, the query vector and `k` are bound parameters):

```sql
SELECT chunk_id FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s;
SELECT chunk_id FROM chunk WHERE n_words < %s ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id WHERE chunk.n_words < %s ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s;
```
