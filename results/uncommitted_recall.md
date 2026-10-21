# uncommitted_recall

Created 2026-10-21T23:33:10+00:00 by `scripts/run_experiments.py uncommitted` on MariaDB 11.8.9-MariaDB-ubu2404. A scratch table with the chunk table's vector column and index (VECTOR(384), M=16, cosine) is filled inside one transaction and searched with `ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT 10` before COMMIT; the transaction is rolled back, the table filled again, committed and searched again. Overlap is the share of the exact top 10 (NumPy, all rows) among the ids the index returns; mean over the queries, then mean (min..max) over 3 fresh builds. p50 is the statement time in ms, averaged over the builds.

- `chunks`: the 4598 FINAL chunk embeddings, queried with the 75 claim embeddings.
- `random`: 1000 random normal vectors (seed 1), as many random queries.

| data | state | ef | mean overlap@10 (builds) | worst query | p50 ms |
|---|---|---|---|---|---|
| chunks | uncommitted | 20 | 0.972 (0.963..0.980) | 0.3 | 0.44 |
| chunks | uncommitted | 100 | 0.973 (0.965..0.980) | 0.4 | 0.68 |
| chunks | uncommitted | 400 | 0.973 (0.965..0.980) | 0.4 | 0.86 |
| chunks | committed | 20 | 0.990 (0.987..0.997) | 0.5 | 0.69 |
| chunks | committed | 100 | 1.000 (1.000..1.000) | 1.0 | 0.97 |
| chunks | committed | 400 | 1.000 (1.000..1.000) | 1.0 | 1.03 |
| random | uncommitted | 20 | 0.583 (0.560..0.600) | 0.2 | 0.18 |
| random | uncommitted | 100 | 0.623 (0.605..0.632) | 0.3 | 0.23 |
| random | uncommitted | 400 | 0.623 (0.605..0.632) | 0.3 | 0.32 |
| random | committed | 20 | 0.787 (0.771..0.805) | 0.4 | 0.15 |
| random | committed | 100 | 0.992 (0.989..0.993) | 0.8 | 0.21 |
| random | committed | 400 | 0.996 (0.995..0.999) | 0.9 | 0.34 |

- `chunks`: rows left after ROLLBACK [0]; insert seconds uncommitted / committed 1.448 / 1.413, 1.414 / 1.342, 1.424 / 1.366.
- `random`: rows left after ROLLBACK [0]; insert seconds uncommitted / committed 0.51 / 0.434, 0.523 / 0.446, 0.472 / 0.437.
