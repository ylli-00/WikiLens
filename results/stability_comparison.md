# Stability of the HNSW results (per-claim hit chunk ids, LIMIT 20)

Generated 2026-09-17T12:12:54+00:00. A claim counts as identical when its 20 hit chunk ids are the same, in the same order; `identical_set` ignores the order.

## Runs

| run | created_at | index ingested_at | cache bytes | ef_search | M | unstable claims | article hits @1/5/10/20 | evidence hits @5/10/20 |
|---|---|---|---|---|---|---|---|---|
| baseline | 2026-09-16T23:55:22+00:00 | 2026-09-16T23:30:33+00:00 | - | 20 | - | 0 | 56/64/68/68 | 33/39/48 |
| stability_512mb_run1 | 2026-09-17T11:59:31+00:00 | 2026-09-16T23:30:33+00:00 | 536870912 | 20 | 6 | 0 | 63/69/73/74 | 39/45/53 |
| stability_512mb_run2 | 2026-09-17T11:59:35+00:00 | 2026-09-16T23:30:33+00:00 | 536870912 | 20 | 6 | 0 | 63/69/73/74 | 39/45/53 |
| stability_512mb_run3 | 2026-09-17T11:59:39+00:00 | 2026-09-16T23:30:33+00:00 | 536870912 | 20 | 6 | 0 | 63/69/73/74 | 39/45/53 |
| stability_512mb_after_restart | 2026-09-17T11:59:49+00:00 | 2026-09-16T23:30:33+00:00 | 536870912 | 20 | 6 | 0 | 49/58/61/61 | 30/35/42 |
| stability_16mb_run1 | 2026-09-17T11:59:53+00:00 | 2026-09-16T23:30:33+00:00 | 16777216 | 20 | 6 | 0 | 49/58/61/61 | 30/35/42 |
| stability_16mb_run2 | 2026-09-17T11:59:58+00:00 | 2026-09-16T23:30:33+00:00 | 16777216 | 20 | 6 | 0 | 49/58/61/61 | 30/35/42 |
| stability_16mb_run3 | 2026-09-17T12:00:03+00:00 | 2026-09-16T23:30:33+00:00 | 16777216 | 20 | 6 | 0 | 49/58/61/61 | 30/35/42 |

## Comparisons

### within_512mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| stability_512mb_run1 | stability_512mb_run2 | 75 | 75 | 75 | none |
| stability_512mb_run1 | stability_512mb_run3 | 75 | 75 | 75 | none |

### committed_baseline_vs_512mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| baseline | stability_512mb_run1 | 58 | 58 | 75 | 7971, 9154, 13383, 14909, 15823, 22563, 26799, 31383, 32256, 32645, 34024, 43675, 50355, 64595, 85631, 90340, 95085 |
| baseline | stability_512mb_run2 | 58 | 58 | 75 | 7971, 9154, 13383, 14909, 15823, 22563, 26799, 31383, 32256, 32645, 34024, 43675, 50355, 64595, 85631, 90340, 95085 |
| baseline | stability_512mb_run3 | 58 | 58 | 75 | 7971, 9154, 13383, 14909, 15823, 22563, 26799, 31383, 32256, 32645, 34024, 43675, 50355, 64595, 85631, 90340, 95085 |

### after_restart_vs_512mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| stability_512mb_run1 | stability_512mb_after_restart | 46 | 46 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 22563, 26799, 31383, 32256, 34024, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85881, 89819, 90340, 95085 |

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
| stability_512mb_run1 | stability_16mb_run1 | 46 | 46 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 22563, 26799, 31383, 32256, 34024, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85881, 89819, 90340, 95085 |
| stability_512mb_run1 | stability_16mb_run2 | 46 | 46 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 22563, 26799, 31383, 32256, 34024, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85881, 89819, 90340, 95085 |
| stability_512mb_run1 | stability_16mb_run3 | 46 | 46 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 22563, 26799, 31383, 32256, 34024, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85881, 89819, 90340, 95085 |

### committed_baseline_vs_16mb

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| baseline | stability_16mb_run1 | 48 | 48 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 26799, 31383, 32645, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85631, 85881, 89819, 90340 |
| baseline | stability_16mb_run2 | 48 | 48 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 26799, 31383, 32645, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85631, 85881, 89819, 90340 |
| baseline | stability_16mb_run3 | 48 | 48 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 26799, 31383, 32645, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85631, 85881, 89819, 90340 |
