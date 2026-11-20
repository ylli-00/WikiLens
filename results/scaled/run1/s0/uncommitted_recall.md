# uncommitted_recall

Created 2026-11-19T21:47:55+00:00 by `scripts/scaled_uncommitted.py` on MariaDB 11.8.9-MariaDB-ubu2404. A scratch table with the chunk table's vector column and index (VECTOR(384), M=16, cosine) is filled inside one transaction and searched with `ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT 5` before COMMIT; the transaction is rolled back, the table filled again, committed and searched again. Overlap is the share of the exact top 5 (NumPy, all rows) among the ids the index returns; mean over the queries, then mean (min..max) over 3 fresh builds. p50 is the statement time in ms, averaged over the builds.

- `chunks`: the 43134 chunk embeddings of wikilense_s0, queried with its 3000 claim embeddings.
- `random`: 1000 random normal vectors (seed 1), queried with 3000 random vectors.

| data | state | ef | mean overlap@5 (builds) | worst query | p50 ms |
|---|---|---|---|---|---|
| chunks | uncommitted | 20 | 0.876 (0.872..0.881) | 0.0 | 0.34 |
| chunks | uncommitted | 100 | 0.877 (0.875..0.881) | 0.0 | 0.32 |
| chunks | uncommitted | 400 | 0.877 (0.875..0.881) | 0.0 | 0.45 |
| chunks | committed | 20 | 0.978 (0.973..0.983) | 0.0 | 0.39 |
| chunks | committed | 100 | 0.994 (0.993..0.996) | 0.4 | 0.83 |
| chunks | committed | 400 | 0.998 (0.998..0.999) | 0.6 | 2.12 |
| random | uncommitted | 20 | 0.618 (0.611..0.622) | 0.0 | 0.18 |
| random | uncommitted | 100 | 0.652 (0.649..0.654) | 0.0 | 0.25 |
| random | uncommitted | 400 | 0.652 (0.649..0.655) | 0.0 | 0.31 |
| random | committed | 20 | 0.809 (0.802..0.815) | 0.0 | 0.19 |
| random | committed | 100 | 0.992 (0.991..0.993) | 0.6 | 0.29 |
| random | committed | 400 | 0.996 (0.995..0.997) | 0.8 | 0.49 |

- `chunks`: rows left after ROLLBACK [0]; insert seconds uncommitted / committed 22.021 / 22.001, 22.198 / 22.008, 22.196 / 21.863.
- `random`: rows left after ROLLBACK [0]; insert seconds uncommitted / committed 0.705 / 0.603, 0.663 / 0.607, 0.667 / 0.609.
