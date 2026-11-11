# The same vectors, built into the HNSW graph several times

Generated 2026-11-11T07:57:11+00:00 by `scripts/measure_index_behaviour.py builds` on MariaDB 11.8.9-MariaDB-ubu2404. The 4,598 chunk vectors of the corpus are copied, in `chunk_id` order and in one transaction, into a scratch table with the chunk table's vector column and index, in the test database. That is done 10 times for each `M`, and every build is searched with the 75 corpus claims, k = 10, the bare statement. "Article recall" counts the claims with a gold page among the pages of the hits; NumPy's exact ranking of the same vectors gives 74 of 75. "Equal to exact" counts the claims whose hit list is NumPy's exact top 10 in order. "Two builds agree" counts, for each pair of builds, the claims with identical hit lists. Each cell is the smallest to the largest value over the builds (or pairs), one number when they are all the same.

| M | ef_search | Article recall@10, of 75 | Equal to exact, of 75 | Overlap with exact | Two builds agree, of 75 | Build, s |
|---|---|---|---|---|---|---|
| 16 | 20 | 73 to 74 (median 74) | 70 to 74 (median 72) | 0.980 to 0.996 | 68 to 74 (median 71) | 1.4 to 2.3 |
| 16 | 100 | 74 | 74 to 75 (median 75) | 0.999 to 1.000 | 74 to 75 (median 75) | 1.4 to 2.3 |
| 6 | 20 | 68 to 73 (median 69) | 41 to 51 (median 43.5) | 0.805 to 0.887 | 33 to 43 (median 37) | 0.4 to 0.6 |
| 6 | 100 | 72 to 74 (median 74) | 55 to 63 (median 60) | 0.933 to 0.971 | 45 to 57 (median 52) | 0.4 to 0.6 |

The statements (`M` is part of the table definition; the vector and `k` are bound):

```sql
CREATE TABLE build_probe (chunk_id INT UNSIGNED NOT NULL PRIMARY KEY, embedding VECTOR(384) NOT NULL, VECTOR INDEX (embedding) M=16 DISTANCE=cosine) ENGINE=InnoDB;
CREATE TABLE build_probe (chunk_id INT UNSIGNED NOT NULL PRIMARY KEY, embedding VECTOR(384) NOT NULL, VECTOR INDEX (embedding) M=6 DISTANCE=cosine) ENGINE=InnoDB;
INSERT INTO build_probe (chunk_id, embedding) VALUES (%s, %s);
SELECT chunk_id FROM build_probe ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
```
