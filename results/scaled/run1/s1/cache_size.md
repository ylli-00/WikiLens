# The index cache at, above and below the size of the graph

Generated 2026-11-20T03:09:56+00:00 by `scripts/measure_index_behaviour.py cache` on MariaDB 11.8.9-MariaDB-ubu2404: 132,359 chunks, index M = 16, the 3000 corpus claims, k = 5, `mhnsw_ef_search` 100. `mhnsw_max_cache_size` was 536,870,912 bytes (512 MB) when the run began and is 536,870,912 bytes again after it. Each size is set with `SET GLOBAL` as root, and, except in the row that says otherwise, the table is reopened with `FLUSH TABLES chunk`. Then the claims are searched in 2 passes with the bare statement and with the joined one, through `wikilense.search.search`, which sorts the returned rows by distance and chunk id. "Equal to exact" counts, per pass, the claims whose hit list is NumPy's exact top 5 in order. "Two passes agree" counts, for each pair of passes, the claims with identical lists. The times are of the whole call.

| Cache | Statement | Equal to exact, of 3000, per pass | Article recall@5, of 3000 | Overlap with exact | Short | Two passes agree, of 3000 | p50 ms |
|---|---|---|---|---|---|---|---|
| 512 MB (as configured) | bare | [2917, 2917] | 2933 | 0.993 to 0.993 | 0 | [3000] | 2.56 |
| 512 MB (as configured) | joined | [3000, 3000] | 2943 | 1.000 to 1.000 | 0 | [3000] | 28.43 |
| 16 MB, table not reopened | bare | [2151, 2162] | 2657 to 2658 | 0.865 to 0.865 | 0 | [2487] | 17.71 |
| 16 MB, table not reopened | joined | [2314, 2311] | 2691 to 2694 | 0.880 to 0.881 | 0 | [2867] | 22.00 |
| 256 MB | bare | [2902, 2911] | 2931 to 2933 | 0.992 to 0.993 | 0 | [2990] | 2.37 |
| 256 MB | joined | [3000, 3000] | 2943 | 1.000 to 1.000 | 0 | [3000] | 27.20 |
| 192 MB | bare | [2902, 2911] | 2931 to 2933 | 0.992 to 0.993 | 0 | [2990] | 2.27 |
| 192 MB | joined | [3000, 3000] | 2943 | 1.000 to 1.000 | 0 | [3000] | 27.53 |
| 160 MB | bare | [2819, 2823] | 2910 to 2911 | 0.981 to 0.981 | 0 | [2932] | 25.21 |
| 160 MB | joined | [2734, 2725] | 2840 | 0.953 to 0.954 | 0 | [2729] | 351.45 |
| 128 MB | bare | [2722, 2724] | 2874 to 2875 | 0.965 to 0.965 | 0 | [2990] | 39.40 |
| 128 MB | joined | [2440, 2436] | 2743 to 2750 | 0.903 to 0.904 | 0 | [2971] | 493.34 |
| 64 MB | bare | [2561, 2548] | 2798 to 2807 | 0.933 to 0.933 | 0 | [2650] | 35.20 |
| 64 MB | joined | [2283, 2275] | 2685 to 2690 | 0.872 to 0.874 | 0 | [2291] | 166.27 |
| 16 MB | bare | [2163, 2162] | 2658 | 0.865 to 0.865 | 0 | [2999] | 16.54 |
| 16 MB | joined | [2314, 2312] | 2690 to 2695 | 0.879 to 0.881 | 0 | [2824] | 21.92 |
| 512 MB (as configured, after the run) | bare | [2902, 2911] | 2931 to 2933 | 0.992 to 0.993 | 0 | [2990] | 2.14 |
| 512 MB (as configured, after the run) | joined | [3000, 3000] | 2943 | 1.000 to 1.000 | 0 | [3000] | 27.31 |

NumPy's exact ranking finds a gold page for 2943 of 3000 claims. The statements are those of strategy `none` and strategy `inline` in `wikilense/search.py`; the settings statements are:

```sql
SET GLOBAL mhnsw_max_cache_size = 268435456;
SET GLOBAL mhnsw_max_cache_size = 201326592;
SET GLOBAL mhnsw_max_cache_size = 167772160;
SET GLOBAL mhnsw_max_cache_size = 134217728;
SET GLOBAL mhnsw_max_cache_size = 67108864;
SET GLOBAL mhnsw_max_cache_size = 16777216;
FLUSH TABLES chunk;
```
