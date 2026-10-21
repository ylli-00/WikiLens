# uncommitted_recall

Created 2026-10-21T22:26:41+00:00 by `scripts/run_experiments.py uncommitted` on MariaDB 11.8.9-MariaDB-ubu2404. A scratch table with the chunk table's vector column and index (VECTOR(384), M=16, cosine) is filled inside one transaction and searched with `ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT 10` before COMMIT; the transaction is rolled back, the table filled again, committed and searched again. Overlap is the share of the exact top 10 (NumPy, all rows) among the ids the index returns; mean over the queries, then mean (min..max) over 3 fresh builds. p50 is the statement time in ms, averaged over the builds.

- `chunks`: the 4598 FINAL chunk embeddings, queried with the 75 claim embeddings.
- `random`: 1000 random normal vectors (seed 1), as many random queries.

| data | state | ef | mean overlap@10 (builds) | worst query | p50 ms |
|---|---|---|---|---|---|
| chunks | uncommitted | 20 | 0.974 (0.971..0.976) | 0.5 | 0.17 |
| chunks | uncommitted | 100 | 0.975 (0.975..0.976) | 0.6 | 0.19 |
| chunks | uncommitted | 400 | 0.976 (0.975..0.979) | 0.6 | 0.25 |
| chunks | committed | 20 | 0.991 (0.991..0.991) | 0.5 | 0.49 |
| chunks | committed | 100 | 1.000 (1.000..1.000) | 1.0 | 0.52 |
| chunks | committed | 400 | 1.000 (1.000..1.000) | 1.0 | 0.71 |
| random | uncommitted | 20 | 0.628 (0.627..0.631) | 0.2 | 0.25 |
| random | uncommitted | 100 | 0.660 (0.655..0.663) | 0.3 | 0.23 |
| random | uncommitted | 400 | 0.660 (0.655..0.663) | 0.3 | 0.41 |
| random | committed | 20 | 0.779 (0.773..0.787) | 0.4 | 0.17 |
| random | committed | 100 | 0.991 (0.989..0.992) | 0.9 | 0.23 |
| random | committed | 400 | 0.997 (0.996..0.999) | 0.9 | 0.41 |

- `chunks`: rows left after ROLLBACK [0]; insert seconds uncommitted / committed 1.402 / 1.334, 1.446 / 1.325, 1.464 / 1.336.
- `random`: rows left after ROLLBACK [0]; insert seconds uncommitted / committed 0.506 / 0.429, 0.522 / 0.437, 0.526 / 0.441.
