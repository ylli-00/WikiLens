# The same vectors, built into the HNSW graph several times

Generated 2026-11-20T07:31:29+00:00 by `scripts/measure_index_behaviour.py builds` on MariaDB 11.8.9-MariaDB-ubu2404. The 132,359 chunk vectors of the corpus are copied, in `chunk_id` order and in one transaction, into a scratch table with the chunk table's vector column and index, in the test database. That is done 20 times for each `M`, and every build is searched with the 3000 corpus claims, k = 5, the bare statement. "Article recall" counts the claims with a gold page among the pages of the hits; NumPy's exact ranking of the same vectors gives 2943 of 3000. "Equal to exact" counts the claims whose hit list is NumPy's exact top 5 in order. "Two builds agree" counts, for each pair of builds, the claims with identical hit lists. Each cell is the smallest to the largest value over the builds (or pairs), one number when they are all the same.

| M | ef_search | Article recall@5, of 3000 | Equal to exact, of 3000 | Overlap with exact | Two builds agree, of 3000 | Build, s |
|---|---|---|---|---|---|---|
| 16 | 20 | 2838 to 2887 (median 2870) | 2628 to 2743 (median 2688) | 0.955 to 0.972 | 2608 to 2752 (median 2674) | 86.5 to 102.3 |
| 16 | 100 | 2922 to 2937 (median 2930) | 2862 to 2918 (median 2883.5) | 0.988 to 0.993 | 2838 to 2907 (median 2866.5) | 86.5 to 102.3 |
| 6 | 20 | 2218 to 2433 (median 2331.5) | 1553 to 1794 (median 1625) | 0.738 to 0.807 | 1404 to 1676 (median 1493) | 24.5 to 29.4 |
| 6 | 100 | 2709 to 2786 (median 2744.5) | 2041 to 2281 (median 2118) | 0.888 to 0.921 | 1837 to 2112 (median 1946) | 24.5 to 29.4 |

The statements (`M` is part of the table definition; the vector and `k` are bound):

```sql
CREATE TABLE build_probe (chunk_id INT UNSIGNED NOT NULL PRIMARY KEY, embedding VECTOR(384) NOT NULL, VECTOR INDEX (embedding) M=16 DISTANCE=cosine) ENGINE=InnoDB;
CREATE TABLE build_probe (chunk_id INT UNSIGNED NOT NULL PRIMARY KEY, embedding VECTOR(384) NOT NULL, VECTOR INDEX (embedding) M=6 DISTANCE=cosine) ENGINE=InnoDB;
INSERT INTO build_probe (chunk_id, embedding) VALUES (%s, %s);
SELECT chunk_id FROM build_probe ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s;
```
