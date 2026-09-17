# WikiLense experiments: summary

Generated 2026-09-17T12:12:54+00:00 by `scripts/run_experiments.py summary`. Every number below is read from the JSON files in `results/` written by the same script (`write_results` of the evaluation harness); the per-run Markdown files hold the full tables. 75 FEVEROUS claims over a 100-page corpus (8,868 chunks of at most 120 words unless stated); article recall counts the claims with a gold page among the top-k chunks (of 75), evidence recall the claims whose gold sentences are all inside the top-k chunks (of the 66 claims with a sentence-only evidence set). Latencies are the SQL time of one `search()` call in milliseconds, p50 / p95 over 5 repeats x 75 claims, warm, query embedding excluded (about 4.6 ms on the GPU, see the per-run files). Unless stated, the vector index is the schema's HNSW index with M=6 and cosine distance, and `ef` is `mhnsw_ef_search`.

## Machine and versions

| Item | Value |
|---|---|
| platform | Linux-7.0.0-31-generic-x86_64-with-glibc2.39 |
| architecture | x86_64 |
| cpu | 13th Gen Intel(R) Core(TM) i9-13900H |
| cpu_count | 20 |
| memory_gb | 15.2 |
| embedding_device | cuda |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU |
| python | 3.12.3 |
| wikilense | 0.1.0 |
| mariadb | 11.8.9-MariaDB-ubu2404 |
| PyMySQL | 1.2.0 |
| numpy | 2.5.3 |
| torch | 2.14.0 |
| sentence-transformers | 6.0.1 |

## 1. Stability: the same configuration repeated (strategy none, ef 20, no filters)

`baseline` is the committed run of 2026-09-16 (16 MB cache, minutes after the ingest). The other runs use the same on-disk index; the container was restarted between `stability_512mb_run3` and `stability_512mb_after_restart`, and `SET GLOBAL mhnsw_max_cache_size = 16777216` was issued before `stability_16mb_run1` (restored to 536870912 afterwards). "claims unstable within run" counts the claims whose LIMIT-20 hits changed between the five repeats of one run.

| run | cache | index built | article@1 | article@10 | evidence@10 | evidence@20 | SQL p50@10 ms | claims unstable within run |
|---|---|---|---|---|---|---|---|---|
| baseline | not recorded (16 MB server default at the time) | 2026-09-16T23:30:33+00:00 | 56/75 | 68/75 | 39/66 | 48/66 | 0.67 | 0 |
| stability_512mb_run1 | 512.0 MB | 2026-09-16T23:30:33+00:00 | 63/75 | 73/75 | 45/66 | 53/66 | 0.65 | 0 |
| stability_512mb_run2 | 512.0 MB | 2026-09-16T23:30:33+00:00 | 63/75 | 73/75 | 45/66 | 53/66 | 0.74 | 0 |
| stability_512mb_run3 | 512.0 MB | 2026-09-16T23:30:33+00:00 | 63/75 | 73/75 | 45/66 | 53/66 | 0.86 | 0 |
| stability_512mb_after_restart | 512.0 MB | 2026-09-16T23:30:33+00:00 | 49/75 | 61/75 | 35/66 | 42/66 | 0.70 | 0 |
| stability_16mb_run1 | 16.0 MB | 2026-09-16T23:30:33+00:00 | 49/75 | 61/75 | 35/66 | 42/66 | 0.80 | 0 |
| stability_16mb_run2 | 16.0 MB | 2026-09-16T23:30:33+00:00 | 49/75 | 61/75 | 35/66 | 42/66 | 0.98 | 0 |
| stability_16mb_run3 | 16.0 MB | 2026-09-16T23:30:33+00:00 | 49/75 | 61/75 | 35/66 | 42/66 | 1.15 | 0 |

Per-claim comparison of the 20 hit chunk ids (order-sensitive; `identical (set)` ignores the order):

| comparison | reference | run | identical (sequence) | identical (set) | of | claims that differ |
|---|---|---|---|---|---|---|
| within_512mb | stability_512mb_run1 | stability_512mb_run2 | 75 | 75 | 75 | none |
| within_512mb | stability_512mb_run1 | stability_512mb_run3 | 75 | 75 | 75 | none |
| after_restart_vs_512mb | stability_512mb_run1 | stability_512mb_after_restart | 46 | 46 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 22563, 26799, 31383, 32256, 34024, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85881, 89819, 90340, 95085 |
| after_restart_vs_16mb | stability_512mb_after_restart | stability_16mb_run1 | 75 | 75 | 75 | none |
| within_16mb | stability_16mb_run1 | stability_16mb_run2 | 75 | 75 | 75 | none |
| within_16mb | stability_16mb_run1 | stability_16mb_run3 | 75 | 75 | 75 | none |
| 512mb_vs_16mb | stability_512mb_run1 | stability_16mb_run1 | 46 | 46 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 22563, 26799, 31383, 32256, 34024, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85881, 89819, 90340, 95085 |
| 512mb_vs_16mb | stability_512mb_run1 | stability_16mb_run2 | 46 | 46 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 22563, 26799, 31383, 32256, 34024, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85881, 89819, 90340, 95085 |
| 512mb_vs_16mb | stability_512mb_run1 | stability_16mb_run3 | 46 | 46 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 22563, 26799, 31383, 32256, 34024, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85881, 89819, 90340, 95085 |
| committed_baseline_vs_512mb | baseline | stability_512mb_run1 | 58 | 58 | 75 | 7971, 9154, 13383, 14909, 15823, 22563, 26799, 31383, 32256, 32645, 34024, 43675, 50355, 64595, 85631, 90340, 95085 |
| committed_baseline_vs_512mb | baseline | stability_512mb_run2 | 58 | 58 | 75 | 7971, 9154, 13383, 14909, 15823, 22563, 26799, 31383, 32256, 32645, 34024, 43675, 50355, 64595, 85631, 90340, 95085 |
| committed_baseline_vs_512mb | baseline | stability_512mb_run3 | 58 | 58 | 75 | 7971, 9154, 13383, 14909, 15823, 22563, 26799, 31383, 32256, 32645, 34024, 43675, 50355, 64595, 85631, 90340, 95085 |
| committed_baseline_vs_16mb | baseline | stability_16mb_run1 | 48 | 48 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 26799, 31383, 32645, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85631, 85881, 89819, 90340 |
| committed_baseline_vs_16mb | baseline | stability_16mb_run2 | 48 | 48 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 26799, 31383, 32645, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85631, 85881, 89819, 90340 |
| committed_baseline_vs_16mb | baseline | stability_16mb_run3 | 48 | 48 | 75 | 5848, 7971, 9154, 13383, 14253, 14727, 14909, 15823, 26799, 31383, 32645, 40707, 43675, 45340, 45793, 50355, 54047, 56899, 64595, 65842, 67426, 69572, 76962, 85631, 85881, 89819, 90340 |

Reading: within one server process the index search is deterministic. The three 512 MB runs return the same 20 chunks in the same order for 75 of 75 claims, the three 16 MB runs for 75 of 75, and lowering the cache from 512 MB to 16 MB with SET GLOBAL changed nothing (75/75 identical between the run just before and just after it). The container restart is what changed the results: the run after it matches the runs before it for only 46 of 75 claims, 13 claims lost their gold page from the top 20 (5848, 7971, 9154, 13383, 15823, 31383, 34024, 43675, 45793, 54047, 76962, 85881, 90340) and 0 gained one, and article recall@10 went from 73/75 to 61/75. The committed baseline is a third state of the same on-disk index: identical hit lists for 58 of 75 claims with the pre-restart runs and 48 with the post-restart runs. So the phase-2 suspicion, a thrashing 16 MB cache, is not supported by these runs; what differs is the in-memory graph the server has after (re)loading the index, and at ef 20 the effect on recall is large. Section 2 shows that a higher ef_search narrows it.

## 2. mhnsw_ef_search sweep (strategy none; `inline_*` = the joined statement without filters, all on the post-restart index state)

Article recall (of 75):

| run | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| ef_20 | 49/75 | 55/75 | 58/75 | 61/75 | 61/75 |
| ef_50 | 55/75 | 62/75 | 65/75 | 68/75 | 69/75 |
| ef_100 | 61/75 | 67/75 | 70/75 | 73/75 | 73/75 |
| ef_200 | 65/75 | 71/75 | 73/75 | 74/75 | 74/75 |
| ef_400 | 66/75 | 71/75 | 73/75 | 74/75 | 74/75 |
| inline_ef_20 | 66/75 | 71/75 | 73/75 | 74/75 | 74/75 |
| inline_ef_100 | 66/75 | 71/75 | 73/75 | 74/75 | 74/75 |

Evidence recall (of 66):

| run | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| ef_20 | 20/66 | 27/66 | 30/66 | 35/66 | 42/66 |
| ef_50 | 24/66 | 31/66 | 34/66 | 40/66 | 49/66 |
| ef_100 | 30/66 | 37/66 | 40/66 | 46/66 | 56/66 |
| ef_200 | 34/66 | 43/66 | 46/66 | 52/66 | 62/66 |
| ef_400 | 34/66 | 44/66 | 47/66 | 54/66 | 63/66 |
| inline_ef_20 | 34/66 | 44/66 | 47/66 | 54/66 | 63/66 |
| inline_ef_100 | 34/66 | 44/66 | 47/66 | 54/66 | 63/66 |

SQL latency p50 / p95 in ms:

| run | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| ef_20 | 0.62 / 0.93 | 0.62 / 0.93 | 0.63 / 0.95 | 0.69 / 1.04 | 0.82 / 1.15 |
| ef_50 | 0.63 / 0.73 | 0.63 / 0.72 | 0.64 / 0.73 | 0.70 / 0.81 | 0.83 / 0.94 |
| ef_100 | 0.76 / 1.67 | 0.74 / 1.70 | 0.74 / 1.71 | 0.82 / 1.83 | 0.94 / 2.10 |
| ef_200 | 0.90 / 1.63 | 0.89 / 1.68 | 0.89 / 1.70 | 0.94 / 1.88 | 1.05 / 2.07 |
| ef_400 | 0.91 / 1.66 | 0.86 / 1.62 | 0.87 / 1.58 | 0.93 / 1.68 | 1.06 / 1.88 |
| inline_ef_20 | 4.26 / 4.80 | 4.26 / 4.78 | 4.27 / 4.76 | 4.34 / 4.85 | 4.47 / 4.96 |
| inline_ef_100 | 4.34 / 5.22 | 4.36 / 5.27 | 4.36 / 5.16 | 4.42 / 5.11 | 4.54 / 5.31 |

Reading: raising ef_search from 20 to 100 lifts article recall@10 from 61/75 to 73/75 and evidence recall@20 from 42/66 to 56/66 for p50@10 0.69 vs 0.82 ms; ef 200 reaches 74/75 / 62/66 and ef 400 74/75 / 63/66 at 0.94 / 0.93 ms. The inline statement walks the index until k joined rows have passed and so returns the exact top-k: 74/75 / 63/66 at both ef values (hit lists identical for 75/75 claims) for p50@10 4.34 ms. ef 400 matches the exact counts at every k; its hit lists are identical to the exact ones for 54/75 claims (ef 200: 44/75, ef 100: 33/75, ef 20: 17/75). Across the two index states of section 1 the LIMIT-20 hit lists at ef 100 (`ef_search_100`, committed, vs `ef_100`) agree for 61/75 claims, against 48/75 at ef 20: a higher ef_search also makes the results less dependent on the index state.

## 3. Index M sweep (ALTER TABLE chunk DROP INDEX / ADD VECTOR INDEX ... M=n DISTANCE=cosine; ef 20 and 100)

`chunk index_length` is `information_schema.tables` for `chunk` (its B-tree secondary indexes only, recomputed by the ALTER); `vector tablespace` is the file size of the hidden InnoDB table `chunk#i#NN` that holds the HNSW graph (`information_schema.innodb_sys_tablespaces`).

| M before | M | DROP INDEX s | ADD VECTOR INDEX s | chunk index_length | vector tablespace |
|---|---|---|---|---|---|
| 6 | 16 | 0.13 | 2.56 | 49,152 B | 17.0 MB |
| 16 | 32 | 0.11 | 8.72 | 49,152 B | 20.0 MB |
| 32 | 6 | 0.13 | 0.69 | 49,152 B | 16.0 MB |

`ef_20` / `ef_100` are the M=6 index built at ingest (section 2); `inline_ef_20` is the exact ranking for reference; the last column counts the claims whose 20 hits are identical to that exact ranking.

| run | M | ef | vector tablespace | article@1 | article@5 | article@10 | evidence@5 | evidence@10 | evidence@20 | SQL p50 / p95 @10 ms | identical to exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ef_20 | 6 | 20 | 16.0 MB | 49/75 | 58/75 | 61/75 | 30/66 | 35/66 | 42/66 | 0.69 / 1.04 | 17/75 |
| ef_100 | 6 | 100 | 16.0 MB | 61/75 | 70/75 | 73/75 | 40/66 | 46/66 | 56/66 | 0.82 / 1.83 | 33/75 |
| m16_ef_20 | 16 | 20 | 17.0 MB | 66/75 | 73/75 | 74/75 | 47/66 | 54/66 | 63/66 | 0.71 / 1.33 | 64/75 |
| m16_ef_100 | 16 | 100 | 17.0 MB | 66/75 | 73/75 | 74/75 | 47/66 | 54/66 | 63/66 | 0.87 / 1.15 | 71/75 |
| m32_ef_20 | 32 | 20 | 20.0 MB | 66/75 | 73/75 | 74/75 | 47/66 | 54/66 | 63/66 | 0.82 / 1.92 | 65/75 |
| m32_ef_100 | 32 | 100 | 20.0 MB | 66/75 | 73/75 | 74/75 | 47/66 | 54/66 | 63/66 | 1.00 / 1.80 | 74/75 |
| m6_rebuilt_ef_20 | 6 | 20 | 16.0 MB | 61/75 | 68/75 | 68/75 | 43/66 | 49/66 | 56/66 | 0.57 / 1.31 | 27/75 |
| inline_ef_20 | 6 | 20 | 16.0 MB | 66/75 | 73/75 | 74/75 | 47/66 | 54/66 | 63/66 | 4.34 / 4.85 | 75/75 |

`m6_rebuilt_ef_20` vs `stability_512mb_run1`: identical hit lists for 13 of 75 claims (same set: 13).

`m6_rebuilt_ef_20` vs `ef_20`: identical hit lists for 11 of 75 claims (same set: 11).

Reading: at ef 20 the M=16 and M=32 graphs already return the exact counts of the inline statement (article@10 74/75 and 74/75, evidence@20 63/66 and 63/66), where the M=6 graph gives 61/75 / 42/66 at ef 20 and 73/75 / 56/66 at ef 100. Hit lists identical to the exact ranking: M=16 64/75 at ef 20 and 71/75 at ef 100, M=32 65/75 and 74/75. p50@10 at ef 20 is 0.69 (M=6) / 0.71 (M=16) / 0.82 (M=32) ms; ADD VECTOR INDEX took 0.69 / 2.56 / 8.72 s and the graph tablespace is 16.0 MB / 17.0 MB / 20.0 MB. The M=6 index rebuilt by ALTER TABLE does not reproduce `stability_512mb_run1`: 13/75 identical hit lists, article recall@10 68/75 against 73/75; every build of an M=6 graph is a different approximation of the same vectors (`final_state_ef_20` in section 7 is a third one, identical to the rebuilt one for 27/75 claims).

## 4. Chunk size at equal retrieved text (strategy none, ef 100, overlap 1, prefix on)

ef 100 for all three chunkings so that the comparison measures the chunking rather than the HNSW approximation at the default ef 20. Each chunking has its own ingest and its own freshly built M=6 index; the 120-word case is `ef_100` of section 2 (the committed ingest), its ingest time is that of the identical re-ingest `restore_defaults` in `ingest_runs.json`.

| chunk_max_words | chunks | words mean | median | p95 | max | ingest total s | embed s | vector tablespace |
|---|---|---|---|---|---|---|---|---|
| 60 | 19467 | 47.9 | 50.0 | 60.0 | 174 | 29.2 | 23.0 | 27.0 MB |
| 120 | 8868 | 93.0 | 103.0 | 119.0 | 174 | 23.52 | 19.27 | 16.0 MB |
| 240 | 4751 | 153.9 | 181.0 | 238.0 | 240 | 20.79 | 17.43 | 12.0 MB |

Aligned by nominal retrieved words (chunk_max_words x k); the words in brackets are k x the mean chunk length actually stored. Each cell: article recall / evidence recall / unit coverage / SQL p50.

| nominal words | 60-word chunks | 120-word chunks | 240-word chunks |
|---|---|---|---|
| 1200 | k=20 (958 words): 73/75 / 49/66 / 0.771 / 0.88 ms | k=10 (930 words): 73/75 / 46/66 / 0.727 / 0.82 ms | k=5 (770 words): 73/75 / 49/66 / 0.761 / 0.69 ms |
| 720 | - | - | k=3 (462 words): 73/75 / 48/66 / 0.747 / 0.67 ms |
| 600 | k=10 (479 words): 73/75 / 43/66 / 0.696 / 0.75 ms | k=5 (465 words): 70/75 / 40/66 / 0.633 / 0.74 ms | - |
| 480 | - | - | k=2 (308 words): 70/75 / 43/66 / 0.672 / 0.67 ms |
| 360 | k=6 (287 words): 73/75 / 39/66 / 0.635 / 0.73 ms | k=3 (279 words): 67/75 / 37/66 / 0.586 / 0.74 ms | - |
| 240 | - | - | k=1 (154 words): 62/75 / 35/66 / 0.543 / 0.69 ms |
| 120 | k=2 (96 words): 66/75 / 30/66 / 0.500 / 0.74 ms | k=1 (93 words): 61/75 / 30/66 / 0.467 / 0.76 ms | - |

Reading: at 1,200 nominal words the three chunkings give article recall 73/75 / 73/75 / 73/75 and evidence recall 49/66 / 46/66 / 49/66 (unit coverage 0.771 / 0.727 / 0.761) for 60 / 120 / 240 words; at about 600 words, 60x10 gives 73/75 / 43/66, 120x5 70/75 / 40/66, 240x2 (480 words) 70/75 / 43/66 and 240x3 (720 words) 73/75 / 48/66. The 240-word chunks are never below the others at equal nominal text although they actually retrieve fewer words (mean chunk 153.9 words, because sections are short), and the differences of two to three claims are the size of the index-state noise at ef 100 (section 2: `ef_search_100` vs `ef_100` differ by one claim in article recall@10 and three in evidence recall@20). So the chunk size does not move recall at equal retrieved text on this corpus; it moves the cost: 19467 / 8868 / 4751 vectors, a graph of 27.0 MB / 16.0 MB / 12.0 MB, and p50 0.88 / 0.82 / 0.69 ms for the 1,200-word row.

## 5. Embedding prefix "title > section path: " on / off (120-word chunks, strategy none)

The no-prefix runs use their own re-ingest and freshly built M=6 index; `inline_ef_20` is the exact ranking of the prefixed vectors, for reference.

| run | prefix | ef | article@1 | article@3 | article@5 | article@10 | article@20 | evidence@1 | evidence@3 | evidence@5 | evidence@10 | evidence@20 | coverage@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ef_20 | on | 20 | 49/75 | 55/75 | 58/75 | 61/75 | 61/75 | 20/66 | 27/66 | 30/66 | 35/66 | 42/66 | 0.553 |
| noprefix_ef_20 | off | 20 | 60/75 | 67/75 | 68/75 | 69/75 | 70/75 | 32/66 | 39/66 | 43/66 | 50/66 | 53/66 | 0.788 |
| ef_100 | on | 100 | 61/75 | 67/75 | 70/75 | 73/75 | 73/75 | 30/66 | 37/66 | 40/66 | 46/66 | 56/66 | 0.727 |
| noprefix_ef_100 | off | 100 | 65/75 | 72/75 | 74/75 | 74/75 | 74/75 | 35/66 | 42/66 | 47/66 | 54/66 | 58/66 | 0.848 |
| inline_ef_20 | on (exact ranking) | 20 | 66/75 | 71/75 | 73/75 | 74/75 | 74/75 | 34/66 | 44/66 | 47/66 | 54/66 | 63/66 | 0.841 |

Reading: without the prefix the counts are equal or higher at every k. At ef 100, article recall@1 is 65/75 against 61/75 with the prefix, @10 74/75 against 73/75, evidence recall@10 54/66 against 46/66, @20 58/66 against 56/66, unit coverage@10 0.848 against 0.727; at ef 20 the gap is wider (69/75 vs 61/75 article@10) but that comparison is confounded by the index state of section 1. Against the exact ranking of the prefixed vectors (`inline_ef_20`: article@10 74/75, evidence@10 54/66, @20 63/66) the approximate no-prefix ranking at ef 100 stands at 74/75, 54/66 and 58/66. The title-and-section prefix therefore buys nothing on these claims, which name their subject in the claim text itself.

## 6. Filter strategies (ef 20, k 5 and 10, 120-word chunks, prefix on)

`queries < k rows` counts, over the 75 claims, the queries that returned fewer than k rows (a separate untimed pass of `search()` per claim and k); `no rows` those that returned nothing.

| run | SQL p50 / p95 @5 ms | SQL p50 / p95 @10 ms | queries < 5 rows | queries < 10 rows | no rows @10 | rows mean @10 | article@5 | article@10 | evidence@5 | evidence@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| strategy_inline_minwords1000 | 4.37 / 5.50 | 4.44 / 5.96 | 0 | 0 | 0 | 10.0 | 63/75 | 64/75 | 37/66 | 45/66 |
| strategy_overfetch10_minwords1000 | 1.03 / 1.46 | 1.20 / 1.77 | 0 | 0 | 0 | 10.0 | 63/75 | 64/75 | 35/66 | 43/66 |
| strategy_overfetch50_minwords1000 | 1.38 / 1.74 | 1.95 / 2.31 | 0 | 0 | 0 | 10.0 | 63/75 | 64/75 | 36/66 | 44/66 |
| strategy_inline_history | 4.97 / 6.43 | 5.30 / 6.93 | 0 | 0 | 0 | 10.0 | 31/75 | 32/75 | 0/66 | 0/66 |
| strategy_overfetch10_history | 1.16 / 1.82 | 1.40 / 2.26 | 65 | 66 | 25 | 2.99 | 27/75 | 27/75 | 0/66 | 0/66 |
| strategy_overfetch50_history | 1.90 / 3.55 | 2.79 / 5.03 | 31 | 23 | 0 | 8.73 | 30/75 | 31/75 | 0/66 | 0/66 |

Filter `minwords1000` ({'min_words': 1000}): 8787 of 8868 chunks pass, on 89 of 100 pages. The filter itself excludes every gold page of 10 of 75 claims, so article recall can reach at most 65/75; evidence recall at most 56/66 (a claim's gold sentences must all lie in chunks that pass the filter).

Filter `history` ({'heading_like': '%History%'}): 268 of 8868 chunks pass, on 66 of 100 pages. The filter itself excludes every gold page of 37 of 75 claims, so article recall can reach at most 38/75; evidence recall at most 0/66 (a claim's gold sentences must all lie in chunks that pass the filter).

Reading: with the non-selective filter (min_words 1000, 8787 of 8868 chunks pass) every strategy returns k rows for every query and the same article recall (64/75 at k=10, of a ceiling of 65); evidence recall@10 is 45/66 inline against 43/66 / 44/66 for overfetch 10 / 50, whose inner index query is approximate at ef 20 while the inline walk is exact. Inline costs 4.44 ms p50 at k=10 against 1.20 ms (overfetch 10) and 1.95 ms (overfetch 50). With the selective filter (heading LIKE '%History%', 268 chunks pass) overfetch 10 returns fewer than 10 rows for 66 of 75 queries (25 with no row at all) and overfetch 50 for 23, while inline always fills k at 5.30 ms p50 / 6.93 ms p95 (overfetch 10: 1.40 ms, overfetch 50: 2.79 ms). Article recall@10 under the History filter is 32/75 inline, 27/75 overfetch 10 and 31/75 overfetch 50, of a ceiling of 38; evidence recall is 0 for all because no claim's gold sentences lie in a History section.

## 7. Restored default state (120 words, overlap 1, prefix on, M=6, ef 20)

| run | index built | cache | article@1 | article@3 | article@5 | article@10 | article@20 | evidence@1 | evidence@3 | evidence@5 | evidence@10 | evidence@20 | SQL p50@10 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2026-09-16T23:30:33+00:00 | not recorded (16 MB server default at the time) | 56/75 | 62/75 | 64/75 | 68/75 | 68/75 | 22/66 | 31/66 | 33/66 | 39/66 | 48/66 | 0.67 |
| stability_512mb_run1 | 2026-09-16T23:30:33+00:00 | 512.0 MB | 63/75 | 66/75 | 69/75 | 73/75 | 74/75 | 28/66 | 36/66 | 39/66 | 45/66 | 53/66 | 0.65 |
| ef_20 | 2026-09-16T23:30:33+00:00 | 512.0 MB | 49/75 | 55/75 | 58/75 | 61/75 | 61/75 | 20/66 | 27/66 | 30/66 | 35/66 | 42/66 | 0.69 |
| m6_rebuilt_ef_20 | 2026-09-16T23:30:33+00:00 | 512.0 MB | 61/75 | 67/75 | 68/75 | 68/75 | 68/75 | 32/66 | 40/66 | 43/66 | 49/66 | 56/66 | 0.57 |
| final_state_ef_20 | 2026-09-17T12:04:08+00:00 | 512.0 MB | 62/75 | 68/75 | 69/75 | 70/75 | 70/75 | 30/66 | 38/66 | 42/66 | 48/66 | 55/66 | 0.57 |

Reading: the final re-ingest (8868 chunks, 23.52 s) left `ingest_meta` identical to the committed baseline's parameters (ingested_at excepted), the vector index at M=6 and mhnsw_max_cache_size at 536870912 bytes. The freshly built index gives article recall@10 70/75 and evidence recall@20 55/66 at ef 20, against 68/75 / 48/66 for the committed baseline's index and 61/75 / 42/66 for the same index after the restart; its hit lists are identical to the committed baseline's for 12/75 claims. Same parameters, another HNSW graph (section 3): at ef 20 and M=6 the numbers a reader reproduces will differ from these by a few claims.

## Recommended defaults

The current defaults are the provisional ones of docs/DESIGN.md (M=6, ef_search 20, 120-word chunks, prefix on, strategy inline, 512 MB cache). What the numbers say:

- **Index M = 16** (sql/schema.sql, currently M=6). At ef 20 the M=16 graph returns the exact ranking's counts, article recall@10 74/75 and evidence recall@20 63/66, at p50@10 0.71 ms, where M=6 gives 61/75 / 42/66 at 0.69 ms and needs ef 200 to 400 (74/75 / 62/66 at 0.94 ms, 74/75 / 63/66 at 0.93 ms) to get there. M=32 adds nothing (74/75 / 63/66 at 0.82 ms) and costs 8.72 s to build against 2.56 s for M=16 and 0.69 s for M=6; the graph tablespace is 16.0 MB / 17.0 MB / 20.0 MB for M 6 / 16 / 32 (section 3).
- **mhnsw_ef_search = 100** per query (`search(ef_search=100)`), with M=16. At M=16 ef 100 keeps the exact counts (74/75 / 63/66) and brings the hit lists closer to the exact ranking (71/75 identical claims against 64/75 at ef 20) for 0.87 instead of 0.71 ms p50@10; on the M=6 index ef 100 is also where recall stops depending strongly on the index state (section 1 and 2: 61/75 identical hit lists between two states at ef 100 against 48/75 at ef 20). If M stays 6, use ef 200 (74/75 / 62/66, 0.94 ms).
- **chunk_max_words = 240, overlap 1** (currently 120). At equal retrieved text the 240-word chunks are never below the others (1,200 nominal words: article 73/75 / 73/75 / 73/75, evidence 49/66 / 46/66 / 49/66 for 60 / 120 / 240; 480 to 600 words: 43/66 / 40/66 / 43/66), the differences are within the index-state noise, and 240 halves the vectors (4751 against 8868), the graph (12.0 MB against 16.0 MB) and the rows per query (section 4). 120 remains defensible when a shorter passage is wanted for display: the recall difference is not measurable on this corpus.
- **Prefix off** (currently on). The prefix gives no measurable gain: at ef 100 the no-prefix vectors reach article recall@1 65/75, @10 74/75, evidence recall@10 54/66, @20 58/66 against 61/75, 73/75, 46/66, 56/66 with the prefix, and they match the exact ranking of the prefixed vectors (74/75, 54/66, 63/66) although they are themselves approximate (section 5). One re-ingest each; confirm with an inline (exact) run on a no-prefix index before the README states it.
- **Filtered queries: strategy inline** (the current default of `search()`). It always fills k: under the selective History filter overfetch 10 returned fewer than 10 rows for 66 of 75 queries and overfetch 50 for 23, inline for 0, at p50@10 5.30 ms (p95 6.93 ms) against 1.40 / 2.79 ms. For a non-selective filter overfetch 10 gives the same article recall for 1.20 ms instead of 4.44 ms, so it is the option when latency matters more than a full result (section 6). The inline walk grows with the table (docs/DESIGN.md: 24 ms on 12,000 random chunks), which is the number to watch when the corpus is scaled.
- **mhnsw_max_cache_size = 512 MB** (docker-compose.yml, kept). The cache size did not change a single hit list (75 of 75 hit lists identical between 512 MB and 16 MB within one server process); the container restart did (46 of 75 identical, section 1). 512 MB is kept for capacity: the 8,868-chunk graph is already 16.0 MB on disk, the 16 MB default would not hold a larger corpus. The restart effect is an open point for the README: results at ef 20 / M=6 are reproducible only within one server process.

