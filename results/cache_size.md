# The index cache at, above and below the size of the graph

Generated 2026-11-11T09:00:24+00:00 by `scripts/measure_index_behaviour.py cache` on MariaDB 11.8.9-MariaDB-ubu2404: 4,598 chunks, index M = 16, the 75 corpus claims, k = 10, `mhnsw_ef_search` 100. `mhnsw_max_cache_size` was 536,870,912 bytes (512 MB) when the run began and is 536,870,912 bytes again after it. Each size is set with `SET GLOBAL` as root, and, except in the row that says otherwise, the table is reopened with `FLUSH TABLES chunk`. Then the claims are searched in 3 passes with the bare statement and with the joined one, through `wikilense.search.search`, which sorts the returned rows by distance and chunk id. "Equal to exact" counts, per pass, the claims whose hit list is NumPy's exact top 10 in order. "Two passes agree" counts, for each pair of passes, the claims with identical lists. The times are of the whole call.

| Cache | Statement | Equal to exact, of 75, per pass | Article recall@10, of 75 | Overlap with exact | Short | Two passes agree, of 75 | p50 ms |
|---|---|---|---|---|---|---|---|
| 512 MB (as configured) | bare | [75, 75, 75] | 74 | 1.000 to 1.000 | 0 | [75, 75, 75] | 0.65 |
| 512 MB (as configured) | joined | [75, 75, 75] | 74 | 1.000 to 1.000 | 0 | [75, 75, 75] | 2.25 |
| 1 MB, table not reopened | bare | [57, 58, 57] | 74 | 0.955 to 0.963 | 0 | [59, 53, 55] | 2.15 |
| 1 MB, table not reopened | joined | [57, 59, 57] | 74 | 0.955 to 0.965 | 0 | [55, 53, 52] | 2.26 |
| 16 MB | bare | [75, 75, 75] | 74 | 1.000 to 1.000 | 0 | [75, 75, 75] | 0.65 |
| 16 MB | joined | [75, 75, 75] | 74 | 1.000 to 1.000 | 0 | [75, 75, 75] | 2.34 |
| 8 MB | bare | [64, 65, 65] | 74 | 0.975 to 0.979 | 0 | [68, 60, 64] | 2.73 |
| 8 MB | joined | [65, 65, 65] | 74 | 0.972 to 0.972 | 0 | [75, 75, 75] | 3.80 |
| 4 MB | bare | [66, 66, 66] | 74 | 0.984 to 0.984 | 0 | [75, 75, 75] | 2.71 |
| 4 MB | joined | [67, 67, 67] | 74 | 0.983 to 0.983 | 0 | [75, 75, 75] | 4.14 |
| 2 MB | bare | [57, 58, 57] | 74 | 0.955 to 0.963 | 0 | [59, 53, 55] | 2.48 |
| 2 MB | joined | [57, 59, 57] | 74 | 0.955 to 0.965 | 0 | [55, 53, 52] | 2.59 |
| 1 MB | bare | [58, 58, 58] | 74 | 0.967 to 0.967 | 0 | [75, 75, 75] | 2.53 |
| 1 MB | joined | [58, 58, 59] | 74 | 0.961 to 0.967 | 0 | [75, 64, 64] | 2.61 |
| 512 MB (as configured, after the run) | bare | [75, 75, 75] | 74 | 1.000 to 1.000 | 0 | [75, 75, 75] | 1.02 |
| 512 MB (as configured, after the run) | joined | [75, 75, 75] | 74 | 1.000 to 1.000 | 0 | [75, 75, 75] | 2.58 |

NumPy's exact ranking finds a gold page for 74 of 75 claims. The statements are those of strategy `none` and strategy `inline` in `wikilense/search.py`; the settings statements are:

```sql
SET GLOBAL mhnsw_max_cache_size = 16777216;
SET GLOBAL mhnsw_max_cache_size = 8388608;
SET GLOBAL mhnsw_max_cache_size = 4194304;
SET GLOBAL mhnsw_max_cache_size = 2097152;
SET GLOBAL mhnsw_max_cache_size = 1048576;
FLUSH TABLES chunk;
```
