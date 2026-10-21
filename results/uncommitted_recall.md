# uncommitted_recall

Created 2026-10-21T22:46:34+00:00 by `scripts/run_experiments.py uncommitted` on MariaDB 11.8.9-MariaDB-ubu2404. A scratch table with the chunk table's vector column and index (VECTOR(384), M=16, cosine) is filled inside one transaction and searched with `ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT 10` before COMMIT; the transaction is rolled back, the table filled again, committed and searched again. Overlap is the share of the exact top 10 (NumPy, all rows) among the ids the index returns; mean over the queries, then mean (min..max) over 3 fresh builds. p50 is the statement time in ms, averaged over the builds.

- `chunks`: the 4598 FINAL chunk embeddings, queried with the 75 claim embeddings.
- `random`: 1000 random normal vectors (seed 1), as many random queries.

| data | state | ef | mean overlap@10 (builds) | worst query | p50 ms |
|---|---|---|---|---|---|
| chunks | uncommitted | 20 | 0.968 (0.965..0.972) | 0.5 | 0.13 |
| chunks | uncommitted | 100 | 0.971 (0.969..0.972) | 0.5 | 0.17 |
| chunks | uncommitted | 400 | 0.971 (0.969..0.972) | 0.5 | 0.24 |
| chunks | committed | 20 | 0.987 (0.977..0.993) | 0.5 | 0.59 |
| chunks | committed | 100 | 1.000 (0.999..1.000) | 0.9 | 0.70 |
| chunks | committed | 400 | 1.000 (1.000..1.000) | 1.0 | 0.91 |
| random | uncommitted | 20 | 0.620 (0.616..0.624) | 0.2 | 0.15 |
| random | uncommitted | 100 | 0.651 (0.643..0.663) | 0.3 | 0.19 |
| random | uncommitted | 400 | 0.651 (0.643..0.663) | 0.3 | 0.25 |
| random | committed | 20 | 0.782 (0.765..0.797) | 0.4 | 0.16 |
| random | committed | 100 | 0.985 (0.983..0.988) | 0.8 | 0.22 |
| random | committed | 400 | 0.993 (0.989..0.996) | 0.9 | 0.29 |

- `chunks`: rows left after ROLLBACK [0]; insert seconds uncommitted / committed 1.454 / 1.346, 1.458 / 1.328, 1.4 / 1.349.
- `random`: rows left after ROLLBACK [0]; insert seconds uncommitted / committed 0.503 / 0.431, 0.497 / 0.439, 0.486 / 0.429.
