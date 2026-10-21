# WikiLense experiments: summary

Generated 2026-10-21T23:46:16+00:00 by `scripts/run_experiments.py summary`. Every number below is read from the JSON files in `results/` written by the same script (`write_results` of the evaluation harness); the per-run Markdown files hold the full tables. 75 FEVEROUS claims over the 100-page corpus; article recall counts the claims with a gold page among the pages of the top-k chunks (of 75), evidence recall the claims whose gold sentences are all inside the top-k chunks (of the 66 claims with a sentence-only evidence set), unit coverage the mean share of a claim's gold units inside the top-k chunks. Latencies are the SQL time of one `search()` call in milliseconds, p50 / p95 over 5 repeats x the claims, warm, query embedding excluded (reported separately). `ef` is `mhnsw_ef_search`, set per session by the application; the vector index is the HNSW index of `sql/schema.sql` with cosine distance, its M as stated per section. Two ingest configurations are used: OLD = 120 words, overlap 1, M=6, prefix on (phases 1 and 2) and FINAL = 240 words, overlap 1, M=16, prefix on (the chosen defaults); hatnote units ("Main article: ...") are in no chunk in either. "Identical to exact" counts the claims whose 20 hit chunk ids equal, in order, those of the exact ranking (the inline statement without filters, which does not depend on the HNSW graph).

## Machine and versions

| Item | Value |
|---|---|
| platform | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 |
| architecture | x86_64 |
| cpu | 13th Gen Intel(R) Core(TM) i9-13900H |
| cpu_count | 20 |
| memory_gb | 15.2 |
| embedding_device | cuda |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU |
| python | 3.12.3 |
| wikilense | 0.1.0 |
| mariadb | 11.8.9-MariaDB-ubu2404 |
| PyMySQL | 1.2.3 |
| numpy | 2.5.3 |
| torch | 2.14.0 |
| sentence-transformers | 6.0.1 |

## 1. Stability: the same run repeated (OLD: 120 words, overlap 1, M=6, prefix on; strategy none, ef 20)

One fresh ingest, then seven identical evaluations on the same on-disk index: three with the 512 MB HNSW cache, one after a container restart, three after `SET GLOBAL mhnsw_max_cache_size = 16777216` (536870912 restored afterwards). "unstable" counts the claims whose LIMIT-20 hits changed between the five repeats of one run; the last two columns count the claims whose 20 hit chunk ids equal, in order, those of `stability_512mb_run1` and of `stability_512mb_after_restart` (`stability_comparison.md` lists the claims).

| run | cache | ef | article@1 | article@10 | article@20 | evidence@10 | evidence@20 | SQL p50 / p95 @10 ms | unstable | identical to 512mb_run1 | identical to after_restart |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stability_512mb_run1 | 512.0 MB | 20 | 61/75 | 68/75 | 68/75 | 46/66 | 51/66 | 0.60 / 1.06 | 0 | 75/75 | 74/75 |
| stability_512mb_run2 | 512.0 MB | 20 | 61/75 | 68/75 | 68/75 | 46/66 | 51/66 | 0.44 / 1.06 | 0 | 75/75 | 74/75 |
| stability_512mb_run3 | 512.0 MB | 20 | 61/75 | 68/75 | 68/75 | 46/66 | 51/66 | 0.47 / 1.46 | 0 | 75/75 | 74/75 |
| stability_512mb_after_restart | 512.0 MB | 20 | 61/75 | 68/75 | 68/75 | 46/66 | 51/66 | 0.47 / 1.06 | 0 | 74/75 | 75/75 |
| stability_16mb_run1 | 16.0 MB | 20 | 61/75 | 68/75 | 68/75 | 46/66 | 51/66 | 0.45 / 1.31 | 0 | 74/75 | 75/75 |
| stability_16mb_run2 | 16.0 MB | 20 | 61/75 | 68/75 | 68/75 | 46/66 | 51/66 | 0.31 / 0.88 | 0 | 74/75 | 75/75 |
| stability_16mb_run3 | 16.0 MB | 20 | 61/75 | 68/75 | 68/75 | 46/66 | 51/66 | 0.46 / 0.93 | 0 | 74/75 | 75/75 |

Reading: within one server process the search over this M=6 graph is deterministic: the three 512 MB runs return the same 20 chunks in the same order for 75 of 75 claims, the three 16 MB runs for 75 of 75, and lowering the cache from 512 MB to 16 MB with SET GLOBAL left 75 of 75 hit lists unchanged. The container restart is what changed the results: the run after it matches the run before it for 74 of 75 claims, 0 claims lost their gold page from the top 20 and 0 gained one, and article recall@10 went from 68/75 to 68/75 (evidence recall@20 51/66 to 51/66). SQL p50@10 was 0.60 ms before and 0.47 ms after the restart, 0.45 ms with the 16 MB cache.

## 2. mhnsw_ef_search sweep (OLD: 120 words, overlap 1, M=6, prefix on; the index state after the restart of section 1)

`ef_<n>` is strategy none (the bare `ORDER BY VEC_DISTANCE_COSINE ... LIMIT k`, the HNSW search proper); `inline_ef_<n>` the joined statement without filters, which ranks exactly over the index and so does not depend on `ef`; `rrf_120_m6` the hybrid (vector top-100 at ef 100 fused with the full-text top-100 of the claim words by reciprocal rank fusion). The reference of the last column is `inline_ef_20`.

| run | strategy | ef | article@1 | article@5 | article@10 | article@20 | evidence@5 | evidence@10 | evidence@20 | coverage@10 | SQL p50 / p95 @10 ms | identical to exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ef_20 | none | 20 | 61/75 | 67/75 | 68/75 | 68/75 | 39/66 | 46/66 | 51/66 | 0.712 | 0.46 / 1.34 | 21/75 |
| ef_50 | none | 50 | 63/75 | 70/75 | 71/75 | 71/75 | 43/66 | 50/66 | 55/66 | 0.773 | 0.48 / 0.78 | 25/75 |
| ef_100 | none | 100 | 64/75 | 71/75 | 72/75 | 72/75 | 44/66 | 52/66 | 58/66 | 0.803 | 0.50 / 0.62 | 29/75 |
| ef_200 | none | 200 | 66/75 | 73/75 | 74/75 | 74/75 | 47/66 | 55/66 | 60/66 | 0.848 | 0.58 / 0.96 | 36/75 |
| ef_400 | none | 400 | 66/75 | 73/75 | 74/75 | 74/75 | 47/66 | 55/66 | 61/66 | 0.848 | 0.68 / 0.89 | 51/75 |
| inline_ef_20 | inline (exact) | 20 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 3.97 / 4.80 | 75/75 |
| inline_ef_100 | inline (exact) | 100 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 3.89 / 4.54 | 75/75 |
| rrf_120_m6 | rrf (overfetch 10) | 100 | 62/75 | 73/75 | 74/75 | 74/75 | 50/66 | 56/66 | 58/66 | 0.879 | 15.64 / 20.47 | - (other order by construction) |

Reading: raising ef from 20 to 400 moves article recall@10 from 68/75 to 74/75 and evidence recall@20 from 51/66 to 61/66 for SQL p50@10 0.46 against 0.68 ms; ef 100 gives 72/75 / 58/66 at 0.50 ms. The exact ranking (inline) gives 74/75 / 63/66 at p50@10 3.97 ms, with hit lists identical at ef 20 and ef 100 for 75/75 claims; the approximate hit lists are identical to it for 21/75 / 25/75 / 29/75 / 36/75 / 51/75 claims at ef 20 / 50 / 100 / 200 / 400. The rrf hybrid at ef 100 gives article recall@10 74/75 and evidence recall@20 58/66 against 72/75 / 58/66 for the vector ranking alone at the same ef, at p50@10 15.64 ms against 0.50 ms.

## 3. Index M sweep (OLD vectors: 120 words, overlap 1, prefix on; strategy none)

The vector index rebuilt with `ALTER TABLE chunk DROP INDEX` + `ADD VECTOR INDEX ... M=n DISTANCE=cosine` for M 6, 16 and 32 (the drop and add timed separately), each evaluated at ef 20 and 100. `graph tablespace` is the file size of the hidden InnoDB table `chunk#i#NN` that holds the HNSW graph; the exact ranking `inline_ef_20` is the last row for reference.

| run | M | ef | DROP INDEX | ADD VECTOR INDEX | graph tablespace | article@1 | article@10 | article@20 | evidence@10 | evidence@20 | SQL p50 / p95 @10 ms | identical to exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m6_ef_20 | 6 | 20 | 0.84 s | 1.11 s | 16.0 MB | 59/75 | 67/75 | 67/75 | 49/66 | 54/66 | 0.31 / 0.59 | 29/75 |
| m6_ef_100 | 6 | 100 | 0.84 s | 1.11 s | 16.0 MB | 65/75 | 73/75 | 73/75 | 55/66 | 60/66 | 0.53 / 0.93 | 38/75 |
| m16_ef_20 | 16 | 20 | 0.87 s | 2.94 s | 17.0 MB | 66/75 | 74/75 | 74/75 | 57/66 | 64/66 | 0.35 / 0.70 | 60/75 |
| m16_ef_100 | 16 | 100 | 0.87 s | 2.94 s | 17.0 MB | 66/75 | 74/75 | 74/75 | 57/66 | 63/66 | 0.48 / 1.18 | 69/75 |
| m32_ef_20 | 32 | 20 | 0.85 s | 9.08 s | 19.0 MB | 65/75 | 73/75 | 73/75 | 56/66 | 62/66 | 0.37 / 0.56 | 63/75 |
| m32_ef_100 | 32 | 100 | 0.85 s | 9.08 s | 19.0 MB | 66/75 | 74/75 | 74/75 | 57/66 | 63/66 | 0.63 / 0.84 | 73/75 |
| inline_ef_20 | 6 | - | - | - | 16.0 MB | 66/75 | 74/75 | 74/75 | 57/66 | 63/66 | 3.97 / 4.80 | 75/75 |

Reading: ADD VECTOR INDEX took 1.11 / 2.94 / 9.08 s for M 6 / 16 / 32 (DROP INDEX 0.84 / 0.87 / 0.85 s) and the graph tablespace is 16.0 MB / 17.0 MB / 19.0 MB. At ef 20 article recall@10 is 67/75 / 74/75 / 73/75 and evidence recall@20 54/66 / 64/66 / 62/66 against 74/75 / 63/66 for the exact ranking; at ef 100 73/75 / 74/75 / 74/75 and 60/66 / 63/66 / 63/66. Hit lists identical to the exact ranking: 29/75 / 60/75 / 63/75 at ef 20 and 38/75 / 69/75 / 73/75 at ef 100, for SQL p50@10 0.31 / 0.35 / 0.37 ms at ef 20 and 0.53 / 0.48 / 0.63 ms at ef 100. The M=6 graph built here and the M=6 graph of section 2 (the same vectors, `ef_20`) give identical hit lists for 12/75 claims: every build of an M=6 graph is another approximation.

## 4. Chunk size at equal retrieved text (60 / 120 / 240 words, overlap 1, M=16, prefix on; strategy none, ef 100)

One fresh ingest per chunk size, each with its own M=16 graph built by the schema. The lower rows align the ks by nominal retrieved words (chunk_max_words x k); the words in brackets are k x the mean chunk length actually stored. Each cell: article recall / evidence recall / unit coverage / SQL p50.

|  | 60-word chunks | 120-word chunks | 240-word chunks |
|---|---|---|---|
| chunks | 19065 | 8658 | 4598 |
| hatnote units excluded | 1316 | 1316 | 1316 |
| words mean / median / p95 / max | 48.1 / 50.0 / 60.0 / 174 | 93.8 / 104.0 / 119.0 / 174 | 156.6 / 183.5 / 238.0 / 240 |
| ingest total s (embed s) | 32.37 (22.81) | 24.51 (18.86) | 21.16 (16.91) |
| graph tablespace | 29.0 MB | 17.0 MB | 13.0 MB |
| 2,400 nominal words | k=40 (1924 words): 74/75 / 60/66 / 0.932 / 0.93 ms | k=20 (1876 words): 74/75 / 63/66 / 0.955 / 0.66 ms | k=10 (1566 words): 74/75 / 60/66 / 0.917 / 0.43 ms |
| 1,200 nominal words | k=20 (962 words): 74/75 / 54/66 / 0.847 / 0.71 ms | k=10 (938 words): 74/75 / 57/66 / 0.879 / 0.53 ms | k=5 (783 words): 74/75 / 55/66 / 0.852 / 0.36 ms |
| 720 nominal words | - | - | k=3 (470 words): 74/75 / 52/66 / 0.808 / 0.34 ms |
| 600 nominal words | k=10 (481 words): 74/75 / 48/66 / 0.771 / 0.58 ms | k=5 (469 words): 74/75 / 49/66 / 0.769 / 0.47 ms | - |
| 480 nominal words | - | - | k=2 (313 words): 70/75 / 49/66 / 0.763 / 0.35 ms |
| 360 nominal words | k=6 (289 words): 74/75 / 44/66 / 0.711 / 0.52 ms | k=3 (281 words): 72/75 / 44/66 / 0.692 / 0.48 ms | - |
| 240 nominal words | - | - | k=1 (157 words): 63/75 / 39/66 / 0.596 / 0.39 ms |
| 120 nominal words | k=2 (96 words): 67/75 / 34/66 / 0.561 / 0.58 ms | k=1 (94 words): 66/75 / 35/66 / 0.543 / 0.52 ms | - |

Reading: at 1,200 nominal words (k = 20 / 10 / 5) the 60 / 120 / 240-word chunkings give article recall 74/75 / 74/75 / 74/75, evidence recall 54/66 / 57/66 / 55/66 and unit coverage 0.847 / 0.879 / 0.852, so the chunkings differ by at most 0 claims in article recall and 3 in evidence recall there; at 2,400 words (k = 40 / 20 / 10) 74/75 / 74/75 / 74/75 and 60/66 / 63/66 / 60/66; at 600 words 60x10 gives 74/75 / 48/66 and 120x5 74/75 / 49/66, with 240x2 (480 words) at 70/75 / 49/66 and 240x3 (720 words) at 74/75 / 52/66. The cost side: 19065 / 8658 / 4598 chunks, a graph of 29.0 MB / 17.0 MB / 13.0 MB, ingest 32.37 / 24.51 / 21.16 s, and SQL p50 0.71 / 0.53 / 0.36 ms for the 1,200-word row. The 240-word chunks retrieve fewer actual words at equal nominal text (mean chunk 156.6 words against 93.8 and 48.1) because a chunk never crosses a section boundary and many sections are short.

## 5. Embedding prefix "title > section path: " on / off (120 words, overlap 1, M=16; ef 100)

One fresh ingest with the prefix and one without, each evaluated approximately (strategy none, ef 100) and exactly (inline, no filters). The last column compares each run with the exact ranking of its own vectors.

| run | prefix | strategy | ef | article@1 | article@5 | article@10 | article@20 | evidence@5 | evidence@10 | evidence@20 | coverage@10 | SQL p50 / p95 @10 ms | identical to own exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prefix_on_ef_100 | on | none | 100 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 0.49 / 0.65 | 74/75 |
| prefix_on_inline | on | inline (exact) | 100 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 4.28 / 5.57 | 75/75 |
| prefix_off_ef_100 | off | none | 100 | 65/75 | 74/75 | 74/75 | 74/75 | 48/66 | 56/66 | 61/66 | 0.875 | 0.51 / 0.94 | 75/75 |
| prefix_off_inline | off | inline (exact) | 100 | 65/75 | 74/75 | 74/75 | 74/75 | 48/66 | 56/66 | 61/66 | 0.875 | 4.23 / 4.65 | 75/75 |

Reading: the exact rankings give, with the prefix, article recall@1 / @10 66/75 / 74/75 and evidence recall@10 / @20 57/66 / 63/66 (unit coverage@10 0.879); without it 65/75 / 74/75 and 56/66 / 61/66 (0.875). So the exact evidence recall@20 with the prefix is above the one without, by 2 claims, and article recall@10 is equal to it. The approximate runs (M=16, ef 100) give 74/75 / 63/66 with and 74/75 / 61/66 without the prefix, with hit lists identical to their exact ranking for 74/75 and 75/75 claims.

## 6. Filter strategies (FINAL: 240 words, overlap 1, M=16, prefix on; ef 100)

`min_words >= 1000` (page length) and `heading LIKE '%History%'` (section heading), each with the inline statement (predicates joined into the index-driven statement), overfetch 10 and overfetch 50 (inner `LIMIT k x factor` by the index, filtered outside). `ceiling` is the best value a strategy can reach because the filter itself excludes the gold pages or gold sentences of some claims; `queries < 10 rows` / `no rows` count, over the claims, the k=10 queries that returned fewer than 10 rows or none (a separate untimed pass of `search()`).

| run | filter | strategy | chunks passing | article ceiling | evidence ceiling | article@5 | article@10 | article@20 | evidence@10 | evidence@20 | queries < 10 rows | no rows @10 | rows mean @10 | SQL p50 / p95 @10 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strategy_inline_minwords1000 | minwords1000 | inline | 4544 of 4598 | 65/75 | 56/66 | 64/75 | 64/75 | 64/75 | 50/66 | 55/66 | 0 | 0 | 10.0 | 2.42 / 2.55 |
| strategy_overfetch10_minwords1000 | minwords1000 | overfetch10 | 4544 of 4598 | 65/75 | 56/66 | 64/75 | 64/75 | 64/75 | 50/66 | 55/66 | 0 | 0 | 10.0 | 0.71 / 1.28 |
| strategy_overfetch50_minwords1000 | minwords1000 | overfetch50 | 4544 of 4598 | 65/75 | 56/66 | 64/75 | 64/75 | 64/75 | 50/66 | 55/66 | 0 | 0 | 10.0 | 1.81 / 2.24 |
| strategy_inline_history | history | inline | 117 of 4598 | 38/75 | 0/66 | 25/75 | 25/75 | 25/75 | 0/66 | 0/66 | 0 | 0 | 10.0 | 3.34 / 4.74 |
| strategy_overfetch10_history | history | overfetch10 | 117 of 4598 | 38/75 | 0/66 | 19/75 | 23/75 | 24/75 | 0/66 | 0/66 | 69 | 18 | 2.97 | 1.02 / 2.00 |
| strategy_overfetch50_history | history | overfetch50 | 117 of 4598 | 38/75 | 0/66 | 25/75 | 25/75 | 25/75 | 0/66 | 0/66 | 20 | 0 | 9.15 | 2.37 / 2.81 |

Reading: under min_words >= 1000 (4544 of 4598 chunks pass; ceilings 65/75 and 56/66) inline / overfetch 10 / overfetch 50 give article recall@10 64/75 / 64/75 / 64/75 and evidence recall@20 55/66 / 55/66 / 55/66 at SQL p50@10 2.42 / 0.71 / 1.81 ms, with 0 / 0 / 0 queries short of 10 rows. Under heading LIKE '%History%' (117 chunks pass; ceilings 38/75 and 0/66) they give 25/75 / 23/75 / 25/75 and 0/66 / 0/66 / 0/66 at 3.34 / 1.02 / 2.37 ms, and overfetch 10 returned fewer than 10 rows for 69 of 75 queries (18 with no row), overfetch 50 for 20, inline for 0. The inline statement's cost is the exact ranking over the index plus the walk until k rows pass the predicates (p95@10 4.74 ms under the History filter against 2.55 ms under min_words), so it grows with the table and with the selectivity of the filter.

## 7. Final configuration (240 words, overlap 1, M=16, prefix on; the fresh ingest of the defaults)

Headline run `final_ef_100`: strategy none (the bare index query joined back to `page` and `section`), ef 100, 4598 chunks (1316 hatnote units excluded), M=16, 512 MB cache. Query embedding p50 is the time of one `embed_queries([claim])` call on the GPU and does not depend on k.

| k | article recall | evidence recall | unit coverage | SQL p50 ms | SQL p95 ms | query embedding p50 ms |
|---|---|---|---|---|---|---|
| 1 | 63/75 (0.840) | 39/66 (0.591) | 0.596 | 0.50 | 0.86 | 4.58 |
| 3 | 74/75 (0.987) | 52/66 (0.788) | 0.808 | 0.45 | 0.78 | 4.58 |
| 5 | 74/75 (0.987) | 55/66 (0.833) | 0.852 | 0.45 | 0.79 | 4.58 |
| 10 | 74/75 (0.987) | 60/66 (0.909) | 0.917 | 0.51 | 0.94 | 4.58 |
| 20 | 74/75 (0.987) | 65/66 (0.985) | 0.985 | 0.64 | 1.10 | 4.58 |

The other runs of the group on the same ingest (`final_inline` is the exact ranking and the reference of the last column; `final_oracle_titles` restricts the inline statement to the claim's gold page with the `titles` filter, so its evidence recall says at which rank the gold sentences surface once the page is known; the `*_after_restart` runs repeat the two approximate runs after a container restart):

| run | strategy | article@1 | article@5 | article@10 | article@20 | evidence@1 | evidence@5 | evidence@10 | evidence@20 | coverage@10 | SQL p50 / p95 @10 ms | identical to exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final_ef_100 | none, ef 100 | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 0.51 / 0.94 | 74/75 |
| final_ef_20 | none, ef 20 | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 0.45 / 0.70 | 65/75 |
| final_inline | inline (exact), ef 100 | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 2.24 / 2.57 | 75/75 |
| final_rrf | rrf (overfetch 10), ef 100 | 63/75 | 74/75 | 74/75 | 74/75 | 34/66 | 57/66 | 60/66 | 64/66 | 0.936 | 9.11 / 12.84 | - (other order by construction) |
| final_oracle_titles | inline + gold page as titles filter, ef 100 | 75/75 | 75/75 | 75/75 | 75/75 | 42/66 | 59/66 | 64/66 | 66/66 | 0.977 | 2.45 / 8.59 | - (other order by construction) |
| final_ef_100_after_restart | none, ef 100, after restart | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 0.52 / 0.93 | 74/75 |
| final_ef_20_after_restart | none, ef 20, after restart | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 0.32 / 0.36 | 65/75 |

Restart comparison (`final_restart_comparison.md`), 20 hit chunk ids per claim:

| before | after | identical (sequence) | identical (set) | article@10 | evidence@20 | gold page lost | gold page gained | claims that differ |
|---|---|---|---|---|---|---|---|---|
| final_ef_100 | final_ef_100_after_restart | 75/75 | 75/75 | 74/75 -> 74/75 | 65/66 -> 65/66 | 0 | 0 | none |
| final_ef_20 | final_ef_20_after_restart | 75/75 | 75/75 | 74/75 -> 74/75 | 65/66 -> 65/66 | 0 | 0 | none |

Reading: the headline run reaches article recall 63/75 / 74/75 / 74/75 / 74/75 at k 1 / 5 / 10 / 20 and evidence recall 39/66 / 55/66 / 60/66 / 65/66, at SQL p50 / p95 0.51 / 0.94 ms for k=10 and query embedding p50 4.58 ms; its article recall@10 is equal to the exact ranking's (74/75) and its evidence recall@20 equal to it (65/66), with hit lists identical to the exact ones for 74/75 claims (65/75 at ef 20) where the exact statement costs 2.24 ms p50. The rrf hybrid gives article recall@10 74/75 and evidence recall@20 64/66 at 9.11 ms p50, equal to the vector ranking alone in article recall@10 and below it in evidence recall@20; with the gold page known (oracle) the gold sentences are inside the first chunk for 42/66 claims, within 5 for 59/66 and within 20 for 66/66, at 2.45 / 8.59 ms. After the container restart the hit lists at ef 100 are identical to the pre-restart ones for 75 of 75 claims (article recall@10 74/75 -> 74/75, evidence recall@20 65/66 -> 65/66) and at ef 20 for 75 of 75 (74/75 -> 74/75, 65/66 -> 65/66), against 74/75 identical hit lists across the restart of the M=6 index in section 1.

## 8. Restored default state (240 words, overlap 1, M=16, prefix on; a fresh `run_ingest` with the defaults)

| check | expected | found | ok |
|---|---|---|---|
| ingest_meta.chunk_max_words | 240 | 240 | yes |
| ingest_meta.chunk_overlap_units | 1 | 1 | yes |
| ingest_meta.embedding_prefix | true | true | yes |
| ingest_meta.index_m | 16 | 16 | yes |
| ingest_meta.index_distance | cosine | cosine | yes |
| ingest_meta.hatnote_pattern | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\b\|instance\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\b\|instance\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) | yes |
| ingest_meta.embedding_model | BAAI/bge-small-en-v1.5 | BAAI/bge-small-en-v1.5 | yes |
| ingest_meta.embedding_dim | 384 | 384 | yes |
| vector index M (SHOW CREATE TABLE chunk) | 16 | 16 | yes |
| vector index DISTANCE | cosine | cosine | yes |
| @@GLOBAL.mhnsw_max_cache_size | 536870912 | 536870912 | yes |
| ingest_meta equals final_ef_100's (ingested_at, analyze_seconds excepted) | identical | identical | yes |

Reading: the last ingest (4598 chunks, 1316 hatnote units excluded, 21.14 s of which 16.83 s embedding) passed every check: `ingest_meta` holds the defaults, the vector index is M=16 DISTANCE=cosine and mhnsw_max_cache_size is 536870912 bytes. This is the state `wikilense ingest` produces, so `wikilense eval` on it reproduces section 7 up to the HNSW build (a fresh graph).

## 9. Searching rows the transaction has not committed yet

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

Reading: the same vectors, the same index definition and the same statement; only the transaction state differs. Compare the overlap of each `uncommitted` row with its `committed` row (`uncommitted_recall.md` has the per-build insert times and the row count after ROLLBACK).

## 10. When the optimizer keeps the vector index under a filter (inline statement)

| filter | chunks passing | plans using the vector index | chunk access (type / key) | chunk rows read, p50 | SQL ms p50 / p95 | short results |
|---|---|---|---|---|---|---|
| none | 4598 | 75/75 | index / embedding | 10 (max 10) | 2.11 / 2.91 | 0 |
| min_words_1000 | 4544 | 75/75 | index / embedding | 10 (max 19) | 2.16 / 2.50 | 0 |
| min_words_5000 | 3891 | 75/75 | index / embedding | 17 (max 75) | 2.05 / 2.33 | 0 |
| max_words_2000 | 89 | 75/75 | index / embedding | 601 (max 1663) | 2.84 / 3.93 | 0 |
| max_words_500 | 25 | 0/75 | ALL / None | 4598 (max 4598) | 5.17 / 6.30 | 0 |
| heading_history | 117 | 75/75 | index / embedding | 315 (max 1014) | 2.99 / 4.02 | 0 |
| linked_from | 661 | 75/75 | index / embedding | 121 (max 492) | 2.90 / 3.71 | 0 |
| category | 695 | 75/75 | index / embedding | 108 (max 420) | 2.93 / 3.85 | 0 |
| edited_since_2020_12_01 | 4525 | 75/75 | index / embedding | 10 (max 25) | 2.24 / 2.50 | 0 |
| edited_before_2020_12_15 | 778 | 75/75 | index / embedding | 67 (max 491) | 2.89 / 4.09 | 0 |
| edited_before_2020_06_01 | 4 | 0/75 | ALL / None | 4598 (max 4598) | 3.08 / 4.02 | 0 |
| four_filters | 377 | 75/75 | index / embedding | 184 (max 888) | 3.22 / 4.29 | 0 |

Details and the exact filter values: `planner_choices.md`.

## 11. Scale: the same claims with more distractor pages

| pages | chunks | graph | HNSW article@10 | exact article@10 | identical hit lists | HNSW ms p50 / p95 | exact ms p50 / p95 | History inline ms p50 / p95 | History overfetch 10: ms p50, short |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 4,598 | 14 MB | 74/75 | 74/75 | 75/75 | 0.83 / 1.90 | 2.52 / 5.16 | 3.30 / 5.04 | 1.03, 69/75 |
| 5,100 | 26,700 | 42 MB | 74/75 | 74/75 | 73/75 | 1.05 / 1.55 | 10.27 / 14.18 | 11.54 / 17.67 | 1.44, 68/75 |
| 20,100 | 89,650 | 113 MB | 71/75 | 71/75 | 72/75 | 1.23 / 3.00 | 19.17 / 27.94 | 21.30 / 32.84 | 1.70, 65/75 |

Method and per-size ingest times: `scale.md`.

## Chosen defaults and evidence

The defaults in the code are `sql/schema.sql` M=16, `Settings.ef_search` 100, `chunk_max_words` 240 with `chunk_overlap_units` 1, the embedding prefix on, `search()` strategy `inline` for filtered queries and `mhnsw_max_cache_size` 512 MB in docker-compose.yml. The evidence, section by section:

- **Index M = 16.** On the same vectors (section 3) the M=6 / 16 / 32 graphs give article recall@10 67/75 / 74/75 / 73/75 and evidence recall@20 54/66 / 64/66 / 62/66 at ef 20, against 74/75 / 63/66 for the exact ranking; at ef 100, the default it is paired with, 73/75 / 74/75 / 74/75 and 60/66 / 63/66 / 63/66, with 38/75 / 69/75 / 73/75 hit lists identical to the exact ranking; the builds took 1.11 / 2.94 / 9.08 s and the graphs are 16.0 MB / 17.0 MB / 19.0 MB. The M values whose article recall@10 and evidence recall@20 at ef 20 equal the exact ranking's: none; M=32 builds in 9.08 s against 2.94 s for M=16 and moves article recall@10 by 0 claims and evidence recall@20 by 0 at ef 100.
- **mhnsw_ef_search = 100 per query.** On the M=6 graph (section 2) ef 20 / 100 / 400 give article recall@10 68/75 / 72/75 / 74/75 and evidence recall@20 51/66 / 58/66 / 61/66 for p50@10 0.46 / 0.50 / 0.68 ms. On the M=16 graph ef 100 makes 69/75 hit lists identical to the exact ranking against 60/75 at ef 20 (section 3), and on the final ingest (section 7) 74/75 against 65/75, for p50@10 0.51 against 0.45 ms; across the restart 75 hit lists stayed identical at ef 100 against 75 at ef 20.
- **chunk_max_words = 240, overlap 1.** At 1,200 nominal words (section 4) the 60 / 120 / 240-word chunkings give article recall 74/75 / 74/75 / 74/75 and evidence recall 54/66 / 57/66 / 55/66, at 2,400 words 74/75 / 74/75 / 74/75 and 60/66 / 63/66 / 60/66; 240 words store 4598 vectors against 8658 and 19065, a graph of 13.0 MB against 17.0 MB and 29.0 MB, and SQL p50 0.36 against 0.53 and 0.71 ms for the 1,200-word row.
- **Embedding prefix on.** The exact rankings (section 5) give evidence recall@20 63/66 with the prefix and 61/66 without, article recall@10 74/75 against 74/75, unit coverage@10 0.879 against 0.875: with the prefix, evidence recall@20 is above the value without it.
- **Filtered queries: strategy inline; overfetch when latency matters more than a full result.** Under the selective History filter (section 6) overfetch 10 returned fewer than 10 rows for 69 of 75 queries and overfetch 50 for 20, inline for 0, at p50@10 3.34 ms (p95 4.74 ms) against 1.02 / 2.37 ms; under the non-selective min_words filter the three give article recall@10 64/75 / 64/75 / 64/75 at 2.42 / 0.71 / 1.81 ms.
- **rrf hybrid: available, not the default.** On the final ingest (section 7) rrf gives article recall@10 74/75 and evidence recall@20 64/66 against 74/75 / 65/66 for the vector ranking alone, at p50@10 9.11 against 0.51 ms; on the OLD ingest (section 2) 74/75 / 58/66 against 72/75 / 58/66 at 15.64 against 0.50 ms.
- **mhnsw_max_cache_size = 512 MB, for capacity.** Within one server process the cache size changed nothing (section 1: 75 of 75 hit lists identical between 512 MB and 16 MB), while the container restart changed 1 of 75 hit lists on the M=6 graph and 0 of 75 on the M=16 graph at ef 100. The graph of the final ingest is 13.0 MB on disk and the 120-word ingest's 16.0 MB, already the size of the 16 MB server default.

