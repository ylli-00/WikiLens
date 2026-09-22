# Final configuration: hit lists before and after a container restart (M=16)

Generated 2026-09-22T01:42:40+00:00. Configuration: 240 words, overlap 1, M=16, prefix on. `final_ef_100` / `final_ef_20` ran before `docker compose restart`, `*_after_restart` after it, on the same on-disk index; `exact_vs_approximate` compares each with the exact ranking `final_inline`. A claim counts as identical when its 20 hit chunk ids are the same, in the same order.

## Runs

| run | created_at | index ingested_at | cache bytes | ef_search | M | chunks | unstable claims | article hits @1/5/10/20 | evidence hits @5/10/20 |
|---|---|---|---|---|---|---|---|---|---|
| final_ef_100 | 2026-09-22T01:41:43+00:00 | 2026-09-22T01:41:38+00:00 | 536870912 | 100 | 16 | 4598 | 0 | 63/74/74/74 | 55/60/65 |
| final_ef_20 | 2026-09-22T01:41:46+00:00 | 2026-09-22T01:41:38+00:00 | 536870912 | 20 | 16 | 4598 | 0 | 63/74/74/74 | 55/60/65 |
| final_inline | 2026-09-22T01:41:54+00:00 | 2026-09-22T01:41:38+00:00 | 536870912 | 100 | 16 | 4598 | 0 | 63/74/74/74 | 55/60/65 |
| final_rrf | 2026-09-22T01:42:17+00:00 | 2026-09-22T01:41:38+00:00 | 536870912 | 100 | 16 | 4598 | 0 | 63/74/74/74 | 57/60/64 |
| final_oracle_titles | 2026-09-22T01:42:27+00:00 | 2026-09-22T01:41:38+00:00 | 536870912 | 100 | 16 | 4598 | 0 | 75/75/75/75 | 59/64/66 |
| final_ef_100_after_restart | 2026-09-22T01:42:37+00:00 | 2026-09-22T01:41:38+00:00 | 536870912 | 100 | 16 | 4598 | 0 | 63/74/74/74 | 55/60/65 |
| final_ef_20_after_restart | 2026-09-22T01:42:40+00:00 | 2026-09-22T01:41:38+00:00 | 536870912 | 20 | 16 | 4598 | 0 | 63/74/74/74 | 55/60/65 |

## Comparisons

### final_ef_100_vs_final_ef_100_after_restart

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| final_ef_100 | final_ef_100_after_restart | 74 | 74 | 75 | 2835 |

### final_ef_20_vs_final_ef_20_after_restart

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| final_ef_20 | final_ef_20_after_restart | 73 | 73 | 75 | 2835, 46127 |

### exact_vs_approximate

| reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|
| final_inline | final_ef_100 | 74 | 74 | 75 | 32256 |
| final_inline | final_ef_100_after_restart | 73 | 73 | 75 | 2835, 32256 |
| final_inline | final_ef_20 | 64 | 64 | 75 | 7971, 13383, 25324, 32256, 36109, 50355, 64595, 74482, 82504, 87976, 94092 |
| final_inline | final_ef_20_after_restart | 62 | 62 | 75 | 2835, 7971, 13383, 25324, 32256, 36109, 46127, 50355, 64595, 74482, 82504, 87976, 94092 |
