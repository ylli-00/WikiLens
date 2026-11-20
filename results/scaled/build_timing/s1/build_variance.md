# The same vectors, built into the HNSW graph several times

Generated 2026-11-20T06:38:30+00:00 by `scripts/measure_index_behaviour.py builds` on MariaDB 11.8.9-MariaDB-ubu2404. The 132,359 chunk vectors of the corpus are copied, in `chunk_id` order and in one transaction, into a scratch table with the chunk table's vector column and index, in the test database. That is done 1 times for each `M`, and every build is searched with the 3000 corpus claims, k = 5, the bare statement. "Article recall" counts the claims with a gold page among the pages of the hits; NumPy's exact ranking of the same vectors gives 2943 of 3000. "Equal to exact" counts the claims whose hit list is NumPy's exact top 5 in order. "Two builds agree" counts, for each pair of builds, the claims with identical hit lists. Each cell is the smallest to the largest value over the builds (or pairs), one number when they are all the same.

| M | ef_search | Article recall@5, of 3000 | Equal to exact, of 3000 | Overlap with exact | Two builds agree, of 3000 | Build, s |
|---|---|---|---|---|---|---|
| 16 | 20 | 2866 | 2699 | 0.966 to 0.966 | - | 85.6 to 85.6 |
| 16 | 100 | 2928 | 2869 | 0.990 to 0.990 | - | 85.6 to 85.6 |

The statements (`M` is part of the table definition; the vector and `k` are bound):

```sql
CREATE TABLE build_probe (chunk_id INT UNSIGNED NOT NULL PRIMARY KEY, embedding VECTOR(384) NOT NULL, VECTOR INDEX (embedding) M=16 DISTANCE=cosine) ENGINE=InnoDB;
INSERT INTO build_probe (chunk_id, embedding) VALUES (%s, %s);
SELECT chunk_id FROM build_probe ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
```
