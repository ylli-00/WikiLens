# Stability of the HNSW results (per-claim hit chunk ids, LIMIT 20)

Generated 2026-09-22T01:35:21+00:00. Configuration: 120 words, overlap 1, M=6, prefix on; strategy none, ef 20. A claim counts as identical when its 20 hit chunk ids are the same, in the same order; `identical (set)` ignores the order. The container was restarted between `stability_512mb_run3` and `stability_512mb_after_restart`; `SET GLOBAL mhnsw_max_cache_size = 16777216` was issued before `stability_16mb_run1` and 536870912 restored afterwards.

## Runs

| run | created_at | index ingested_at | cache bytes | ef_search | M | chunks | unstable claims | article hits @1/5/10/20 | evidence hits @5/10/20 |
|---|---|---|---|---|---|---|---|---|---|
| stability_512mb_run1 | 2026-09-22T01:34:54+00:00 | 2026-09-22T01:34:49+00:00 | 536870912 | 20 | 6 | 8658 | 0 | 62/68/68/68 | 46/53/58 |
| stability_512mb_run2 | 2026-09-22T01:34:58+00:00 | 2026-09-22T01:34:49+00:00 | 536870912 | 20 | 6 | 8658 | 0 | 62/68/68/68 | 46/53/58 |
| stability_512mb_run3 | 2026-09-22T01:35:02+00:00 | 2026-09-22T01:34:49+00:00 | 536870912 | 20 | 6 | 8658 | 0 | 62/68/68/68 | 46/53/58 |
| stability_512mb_after_restart | 2026-09-22T01:35:11+00:00 | 2026-09-22T01:34:49+00:00 | 536870912 | 20 | 6 | 8658 | 0 | 62/68/68/68 | 46/53/58 |
| stability_16mb_run1 | 2026-09-22T01:35:14+00:00 | 2026-09-22T01:34:49+00:00 | 16777216 | 20 | 6 | 8658 | 0 | 62/68/68/68 | 46/53/58 |
| stability_16mb_run2 | 2026-09-22T01:35:18+00:00 | 2026-09-22T01:34:49+00:00 | 16777216 | 20 | 6 | 8658 | 0 | 62/68/68/68 | 46/53/58 |
| stability_16mb_run3 | 2026-09-22T01:35:21+00:00 | 2026-09-22T01:34:49+00:00 | 16777216 | 20 | 6 | 8658 | 0 | 62/68/68/68 | 46/53/58 |

## Comparisons

### within_512mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| stability_512mb_run1 | stability_512mb_run2 | 75 | 75 | 75 | none |
| stability_512mb_run1 | stability_512mb_run3 | 75 | 75 | 75 | none |

### after_restart_vs_512mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| stability_512mb_run1 | stability_512mb_after_restart | 72 | 72 | 75 | 50355, 63422, 87976 |

### after_restart_vs_16mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| stability_512mb_after_restart | stability_16mb_run1 | 75 | 75 | 75 | none |

### within_16mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| stability_16mb_run1 | stability_16mb_run2 | 75 | 75 | 75 | none |
| stability_16mb_run1 | stability_16mb_run3 | 75 | 75 | 75 | none |

### 512mb_vs_16mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| stability_512mb_run1 | stability_16mb_run1 | 72 | 72 | 75 | 50355, 63422, 87976 |
| stability_512mb_run1 | stability_16mb_run2 | 72 | 72 | 75 | 50355, 63422, 87976 |
| stability_512mb_run1 | stability_16mb_run3 | 72 | 72 | 75 | 50355, 63422, 87976 |
