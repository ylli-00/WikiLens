# WikiLense experiments: summary

Generated 2026-09-22T01:44:18+00:00 by `scripts/run_experiments.py summary`. Every number below is read from the JSON files in `results/` written by the same script (`write_results` of the evaluation harness); the per-run Markdown files hold the full tables. 75 FEVEROUS claims over the 100-page corpus; article recall counts the claims with a gold page among the pages of the top-k chunks (of 75), evidence recall the claims whose gold sentences are all inside the top-k chunks (of the 66 claims with a sentence-only evidence set), unit coverage the mean share of a claim's gold units inside the top-k chunks. Latencies are the SQL time of one `search()` call in milliseconds, p50 / p95 over 5 repeats x the claims, warm, query embedding excluded (reported separately). `ef` is `mhnsw_ef_search`, set per session by the application; the vector index is the HNSW index of `sql/schema.sql` with cosine distance, its M as stated per section. Two ingest configurations are used: OLD = 120 words, overlap 1, M=6, prefix on (phases 1 and 2) and FINAL = 240 words, overlap 1, M=16, prefix on (the chosen defaults); hatnote units ("Main article: ...") are in no chunk in either. "Identical to exact" counts the claims whose 20 hit chunk ids equal, in order, those of the exact ranking (the inline statement without filters, which does not depend on the HNSW graph).

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

## 1. Stability: the same run repeated (OLD: 120 words, overlap 1, M=6, prefix on; strategy none, ef 20)

One fresh ingest, then seven identical evaluations on the same on-disk index: three with the 512 MB HNSW cache, one after `docker compose restart`, three after `SET GLOBAL mhnsw_max_cache_size = 16777216` (536870912 restored afterwards). "unstable" counts the claims whose LIMIT-20 hits changed between the five repeats of one run; the last two columns count the claims whose 20 hit chunk ids equal, in order, those of `stability_512mb_run1` and of `stability_512mb_after_restart` (`stability_comparison.md` lists the claims).

| run | cache | ef | article@1 | article@10 | article@20 | evidence@10 | evidence@20 | SQL p50 / p95 @10 ms | unstable | identical to 512mb_run1 | identical to after_restart |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stability_512mb_run1 | 512.0 MB | 20 | 62/75 | 68/75 | 68/75 | 53/66 | 58/66 | 0.54 / 0.95 | 0 | 75/75 | 72/75 |
| stability_512mb_run2 | 512.0 MB | 20 | 62/75 | 68/75 | 68/75 | 53/66 | 58/66 | 0.56 / 0.98 | 0 | 75/75 | 72/75 |
| stability_512mb_run3 | 512.0 MB | 20 | 62/75 | 68/75 | 68/75 | 53/66 | 58/66 | 0.70 / 0.80 | 0 | 75/75 | 72/75 |
| stability_512mb_after_restart | 512.0 MB | 20 | 62/75 | 68/75 | 68/75 | 53/66 | 58/66 | 0.56 / 1.08 | 0 | 72/75 | 75/75 |
| stability_16mb_run1 | 16.0 MB | 20 | 62/75 | 68/75 | 68/75 | 53/66 | 58/66 | 0.56 / 1.11 | 0 | 72/75 | 75/75 |
| stability_16mb_run2 | 16.0 MB | 20 | 62/75 | 68/75 | 68/75 | 53/66 | 58/66 | 0.55 / 0.96 | 0 | 72/75 | 75/75 |
| stability_16mb_run3 | 16.0 MB | 20 | 62/75 | 68/75 | 68/75 | 53/66 | 58/66 | 0.67 / 0.84 | 0 | 72/75 | 75/75 |

Reading: within one server process the search over this M=6 graph is deterministic: the three 512 MB runs return the same 20 chunks in the same order for 75 of 75 claims, the three 16 MB runs for 75 of 75, and lowering the cache from 512 MB to 16 MB with SET GLOBAL left 75 of 75 hit lists unchanged. The container restart is what changed the results: the run after it matches the run before it for 72 of 75 claims, 0 claims lost their gold page from the top 20 and 0 gained one, and article recall@10 went from 68/75 to 68/75 (evidence recall@20 58/66 to 58/66). SQL p50@10 was 0.54 ms before and 0.56 ms after the restart, 0.56 ms with the 16 MB cache.

## 2. mhnsw_ef_search sweep (OLD: 120 words, overlap 1, M=6, prefix on; the index state after the restart of section 1)

`ef_<n>` is strategy none (the bare `ORDER BY VEC_DISTANCE_COSINE ... LIMIT k`, the HNSW search proper); `inline_ef_<n>` the joined statement without filters, which ranks exactly over the index and so does not depend on `ef`; `rrf_120_m6` the hybrid (vector top-100 at ef 100 fused with the full-text top-100 of the claim words by reciprocal rank fusion). The reference of the last column is `inline_ef_20`.

| run | strategy | ef | article@1 | article@5 | article@10 | article@20 | evidence@5 | evidence@10 | evidence@20 | coverage@10 | SQL p50 / p95 @10 ms | identical to exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ef_20 | none | 20 | 62/75 | 68/75 | 68/75 | 68/75 | 46/66 | 53/66 | 58/66 | 0.811 | 0.56 / 0.74 | 24/75 |
| ef_50 | none | 50 | 67/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 64/66 | 0.879 | 0.59 / 0.81 | 40/75 |
| ef_100 | none | 100 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 64/66 | 0.879 | 0.64 / 0.82 | 49/75 |
| ef_200 | none | 200 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 0.77 / 0.93 | 52/75 |
| ef_400 | none | 400 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 0.95 / 1.44 | 60/75 |
| inline_ef_20 | inline (exact) | 20 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 4.11 / 4.64 | 75/75 |
| inline_ef_100 | inline (exact) | 100 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 4.12 / 4.43 | 75/75 |
| rrf_120_m6 | rrf (overfetch 10) | 100 | 61/75 | 74/75 | 74/75 | 74/75 | 54/66 | 58/66 | 60/66 | 0.909 | 15.87 / 20.34 | - (other order by construction) |

Reading: raising ef from 20 to 400 moves article recall@10 from 68/75 to 74/75 and evidence recall@20 from 58/66 to 63/66 for SQL p50@10 0.56 against 0.95 ms; ef 100 gives 74/75 / 64/66 at 0.64 ms. The exact ranking (inline) gives 74/75 / 63/66 at p50@10 4.11 ms, with hit lists identical at ef 20 and ef 100 for 75/75 claims; the approximate hit lists are identical to it for 24/75 / 40/75 / 49/75 / 52/75 / 60/75 claims at ef 20 / 50 / 100 / 200 / 400. The rrf hybrid at ef 100 gives article recall@10 74/75 and evidence recall@20 60/66 against 74/75 / 64/66 for the vector ranking alone at the same ef, at p50@10 15.87 ms against 0.64 ms.

## 3. Index M sweep (OLD vectors: 120 words, overlap 1, prefix on; strategy none)

The vector index rebuilt with `ALTER TABLE chunk DROP INDEX` + `ADD VECTOR INDEX ... M=n DISTANCE=cosine` for M 6, 16 and 32 (the drop and add timed separately), each evaluated at ef 20 and 100. `graph tablespace` is the file size of the hidden InnoDB table `chunk#i#NN` that holds the HNSW graph; the exact ranking `inline_ef_20` is the last row for reference.

| run | M | ef | DROP INDEX | ADD VECTOR INDEX | graph tablespace | article@1 | article@10 | article@20 | evidence@10 | evidence@20 | SQL p50 / p95 @10 ms | identical to exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m6_ef_20 | 6 | 20 | 0.83 s | 1.13 s | 16.0 MB | 62/75 | 69/75 | 69/75 | 50/66 | 55/66 | 0.67 / 0.74 | 24/75 |
| m6_ef_100 | 6 | 100 | 0.83 s | 1.13 s | 16.0 MB | 66/75 | 74/75 | 74/75 | 55/66 | 61/66 | 0.65 / 1.14 | 44/75 |
| m16_ef_20 | 16 | 20 | 0.81 s | 2.94 s | 17.0 MB | 64/75 | 72/75 | 72/75 | 55/66 | 61/66 | 0.57 / 0.80 | 62/75 |
| m16_ef_100 | 16 | 100 | 0.81 s | 2.94 s | 17.0 MB | 66/75 | 74/75 | 74/75 | 57/66 | 63/66 | 0.68 / 0.83 | 71/75 |
| m32_ef_20 | 32 | 20 | 0.84 s | 8.98 s | 19.0 MB | 66/75 | 74/75 | 74/75 | 57/66 | 63/66 | 0.76 / 0.86 | 66/75 |
| m32_ef_100 | 32 | 100 | 0.84 s | 8.98 s | 19.0 MB | 66/75 | 74/75 | 74/75 | 57/66 | 63/66 | 0.77 / 0.95 | 75/75 |
| inline_ef_20 | 6 | - | - | - | 16.0 MB | 66/75 | 74/75 | 74/75 | 57/66 | 63/66 | 4.11 / 4.64 | 75/75 |

Reading: ADD VECTOR INDEX took 1.13 / 2.94 / 8.98 s for M 6 / 16 / 32 (DROP INDEX 0.83 / 0.81 / 0.84 s) and the graph tablespace is 16.0 MB / 17.0 MB / 19.0 MB. At ef 20 article recall@10 is 69/75 / 72/75 / 74/75 and evidence recall@20 55/66 / 61/66 / 63/66 against 74/75 / 63/66 for the exact ranking; at ef 100 74/75 / 74/75 / 74/75 and 61/66 / 63/66 / 63/66. Hit lists identical to the exact ranking: 24/75 / 62/75 / 66/75 at ef 20 and 44/75 / 71/75 / 75/75 at ef 100, for SQL p50@10 0.67 / 0.57 / 0.76 ms at ef 20 and 0.65 / 0.68 / 0.77 ms at ef 100. The M=6 graph built here and the M=6 graph of section 2 (the same vectors, `ef_20`) give identical hit lists for 16/75 claims: every build of an M=6 graph is another approximation.

## 4. Chunk size at equal retrieved text (60 / 120 / 240 words, overlap 1, M=16, prefix on; strategy none, ef 100)

One fresh ingest per chunk size, each with its own M=16 graph built by the schema. The lower rows align the ks by nominal retrieved words (chunk_max_words x k); the words in brackets are k x the mean chunk length actually stored. Each cell: article recall / evidence recall / unit coverage / SQL p50.

|  | 60-word chunks | 120-word chunks | 240-word chunks |
|---|---|---|---|
| chunks | 19065 | 8658 | 4598 |
| hatnote units excluded | 1316 | 1316 | 1316 |
| words mean / median / p95 / max | 48.1 / 50.0 / 60.0 / 174 | 93.8 / 104.0 / 119.0 / 174 | 156.6 / 183.5 / 238.0 / 240 |
| ingest total s (embed s) | 33.92 (22.51) | 24.8 (18.58) | 21.3 (16.65) |
| graph tablespace | 29.0 MB | 17.0 MB | 13.0 MB |
| 2,400 nominal words | k=40 (1924 words): 74/75 / 60/66 / 0.932 / 1.06 ms | k=20 (1876 words): 74/75 / 63/66 / 0.955 / 0.93 ms | k=10 (1566 words): 74/75 / 60/66 / 0.917 / 0.81 ms |
| 1,200 nominal words | k=20 (962 words): 74/75 / 54/66 / 0.847 / 0.83 ms | k=10 (938 words): 74/75 / 57/66 / 0.879 / 0.81 ms | k=5 (783 words): 74/75 / 55/66 / 0.852 / 0.73 ms |
| 720 nominal words | - | - | k=3 (470 words): 74/75 / 52/66 / 0.808 / 0.70 ms |
| 600 nominal words | k=10 (481 words): 74/75 / 48/66 / 0.771 / 0.72 ms | k=5 (469 words): 74/75 / 49/66 / 0.769 / 0.74 ms | - |
| 480 nominal words | - | - | k=2 (313 words): 70/75 / 49/66 / 0.763 / 0.69 ms |
| 360 nominal words | k=6 (289 words): 74/75 / 44/66 / 0.711 / 0.66 ms | k=3 (281 words): 72/75 / 44/66 / 0.692 / 0.72 ms | - |
| 240 nominal words | - | - | k=1 (157 words): 63/75 / 39/66 / 0.596 / 0.72 ms |
| 120 nominal words | k=2 (96 words): 67/75 / 34/66 / 0.561 / 0.65 ms | k=1 (94 words): 66/75 / 35/66 / 0.543 / 0.73 ms | - |

Reading: at 1,200 nominal words (k = 20 / 10 / 5) the 60 / 120 / 240-word chunkings give article recall 74/75 / 74/75 / 74/75, evidence recall 54/66 / 57/66 / 55/66 and unit coverage 0.847 / 0.879 / 0.852, so the chunkings differ by at most 0 claims in article recall and 3 in evidence recall there; at 2,400 words (k = 40 / 20 / 10) 74/75 / 74/75 / 74/75 and 60/66 / 63/66 / 60/66; at 600 words 60x10 gives 74/75 / 48/66 and 120x5 74/75 / 49/66, with 240x2 (480 words) at 70/75 / 49/66 and 240x3 (720 words) at 74/75 / 52/66. The cost side: 19065 / 8658 / 4598 chunks, a graph of 29.0 MB / 17.0 MB / 13.0 MB, ingest 33.92 / 24.8 / 21.3 s, and SQL p50 0.83 / 0.81 / 0.73 ms for the 1,200-word row. The 240-word chunks retrieve fewer actual words at equal nominal text (mean chunk 156.6 words against 93.8 and 48.1) because a chunk never crosses a section boundary and many sections are short.

## 5. Embedding prefix "title > section path: " on / off (120 words, overlap 1, M=16; ef 100)

One fresh ingest with the prefix and one without, each evaluated approximately (strategy none, ef 100) and exactly (inline, no filters). The last column compares each run with the exact ranking of its own vectors.

| run | prefix | strategy | ef | article@1 | article@5 | article@10 | article@20 | evidence@5 | evidence@10 | evidence@20 | coverage@10 | SQL p50 / p95 @10 ms | identical to own exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prefix_on_ef_100 | on | none | 100 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 0.77 / 0.90 | 71/75 |
| prefix_on_inline | on | inline (exact) | 100 | 66/75 | 74/75 | 74/75 | 74/75 | 49/66 | 57/66 | 63/66 | 0.879 | 4.38 / 4.54 | 75/75 |
| prefix_off_ef_100 | off | none | 100 | 65/75 | 74/75 | 74/75 | 74/75 | 48/66 | 56/66 | 61/66 | 0.875 | 0.90 / 1.32 | 75/75 |
| prefix_off_inline | off | inline (exact) | 100 | 65/75 | 74/75 | 74/75 | 74/75 | 48/66 | 56/66 | 61/66 | 0.875 | 4.31 / 4.68 | 75/75 |

Reading: the exact rankings give, with the prefix, article recall@1 / @10 66/75 / 74/75 and evidence recall@10 / @20 57/66 / 63/66 (unit coverage@10 0.879); without it 65/75 / 74/75 and 56/66 / 61/66 (0.875). So the exact evidence recall@20 with the prefix is above the one without, by 2 claims, and article recall@10 is equal to it. The approximate runs (M=16, ef 100) give 74/75 / 63/66 with and 74/75 / 61/66 without the prefix, with hit lists identical to their exact ranking for 71/75 and 75/75 claims.

## 6. Filter strategies (FINAL: 240 words, overlap 1, M=16, prefix on; ef 100)

`min_words >= 1000` (page length) and `heading LIKE '%History%'` (section heading), each with the inline statement (predicates joined into the index-driven statement), overfetch 10 and overfetch 50 (inner `LIMIT k x factor` by the index, filtered outside). `ceiling` is the best value a strategy can reach because the filter itself excludes the gold pages or gold sentences of some claims; `queries < 10 rows` / `no rows` count, over the claims, the k=10 queries that returned fewer than 10 rows or none (a separate untimed pass of `search()`).

| run | filter | strategy | chunks passing | article ceiling | evidence ceiling | article@5 | article@10 | article@20 | evidence@10 | evidence@20 | queries < 10 rows | no rows @10 | rows mean @10 | SQL p50 / p95 @10 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strategy_inline_minwords1000 | minwords1000 | inline | 4544 of 4598 | 65/75 | 56/66 | 64/75 | 64/75 | 64/75 | 50/66 | 55/66 | 0 | 0 | 10.0 | 2.70 / 2.90 |
| strategy_overfetch10_minwords1000 | minwords1000 | overfetch10 | 4544 of 4598 | 65/75 | 56/66 | 64/75 | 64/75 | 64/75 | 50/66 | 55/66 | 0 | 0 | 10.0 | 1.00 / 1.56 |
| strategy_overfetch50_minwords1000 | minwords1000 | overfetch50 | 4544 of 4598 | 65/75 | 56/66 | 64/75 | 64/75 | 64/75 | 50/66 | 55/66 | 0 | 0 | 10.0 | 2.10 / 2.48 |
| strategy_inline_history | history | inline | 117 of 4598 | 38/75 | 0/66 | 25/75 | 25/75 | 25/75 | 0/66 | 0/66 | 0 | 0 | 10.0 | 3.56 / 4.93 |
| strategy_overfetch10_history | history | overfetch10 | 117 of 4598 | 38/75 | 0/66 | 24/75 | 24/75 | 24/75 | 0/66 | 0/66 | 69 | 18 | 2.97 | 1.05 / 1.58 |
| strategy_overfetch50_history | history | overfetch50 | 117 of 4598 | 38/75 | 0/66 | 25/75 | 25/75 | 25/75 | 0/66 | 0/66 | 20 | 0 | 9.13 | 2.76 / 3.28 |

Reading: under min_words >= 1000 (4544 of 4598 chunks pass; ceilings 65/75 and 56/66) inline / overfetch 10 / overfetch 50 give article recall@10 64/75 / 64/75 / 64/75 and evidence recall@20 55/66 / 55/66 / 55/66 at SQL p50@10 2.70 / 1.00 / 2.10 ms, with 0 / 0 / 0 queries short of 10 rows. Under heading LIKE '%History%' (117 chunks pass; ceilings 38/75 and 0/66) they give 25/75 / 24/75 / 25/75 and 0/66 / 0/66 / 0/66 at 3.56 / 1.05 / 2.76 ms, and overfetch 10 returned fewer than 10 rows for 69 of 75 queries (18 with no row), overfetch 50 for 20, inline for 0. The inline statement's cost is the exact ranking over the index plus the walk until k rows pass the predicates (p95@10 4.93 ms under the History filter against 2.90 ms under min_words), so it grows with the table and with the selectivity of the filter.

## 7. Final configuration (240 words, overlap 1, M=16, prefix on; the fresh ingest of the defaults)

Headline run `final_ef_100`: strategy none (the bare index query joined back to `page` and `section`), ef 100, 4598 chunks (1316 hatnote units excluded), M=16, 512 MB cache. Query embedding p50 is the time of one `embed_queries([claim])` call on the GPU and does not depend on k.

| k | article recall | evidence recall | unit coverage | SQL p50 ms | SQL p95 ms | query embedding p50 ms |
|---|---|---|---|---|---|---|
| 1 | 63/75 (0.840) | 39/66 (0.591) | 0.596 | 0.83 | 1.27 | 4.67 |
| 3 | 74/75 (0.987) | 52/66 (0.788) | 0.808 | 0.81 | 1.23 | 4.67 |
| 5 | 74/75 (0.987) | 55/66 (0.833) | 0.852 | 0.82 | 1.27 | 4.67 |
| 10 | 74/75 (0.987) | 60/66 (0.909) | 0.917 | 0.90 | 1.41 | 4.67 |
| 20 | 74/75 (0.987) | 65/66 (0.985) | 0.985 | 1.05 | 1.59 | 4.67 |

The other runs of the group on the same ingest (`final_inline` is the exact ranking and the reference of the last column; `final_oracle_titles` restricts the inline statement to the claim's gold page with the `titles` filter, so its evidence recall says at which rank the gold sentences surface once the page is known; the `*_after_restart` runs repeat the two approximate runs after `docker compose restart`):

| run | strategy | article@1 | article@5 | article@10 | article@20 | evidence@1 | evidence@5 | evidence@10 | evidence@20 | coverage@10 | SQL p50 / p95 @10 ms | identical to exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final_ef_100 | none, ef 100 | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 0.90 / 1.41 | 74/75 |
| final_ef_20 | none, ef 20 | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 0.69 / 0.87 | 64/75 |
| final_inline | inline (exact), ef 100 | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 2.63 / 2.87 | 75/75 |
| final_rrf | rrf (overfetch 10), ef 100 | 63/75 | 74/75 | 74/75 | 74/75 | 35/66 | 57/66 | 60/66 | 64/66 | 0.936 | 10.84 / 13.94 | - (other order by construction) |
| final_oracle_titles | inline + gold page as titles filter, ef 100 | 75/75 | 75/75 | 75/75 | 75/75 | 42/66 | 59/66 | 64/66 | 66/66 | 0.977 | 2.65 / 9.87 | - (other order by construction) |
| final_ef_100_after_restart | none, ef 100, after restart | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 1.00 / 1.43 | 73/75 |
| final_ef_20_after_restart | none, ef 20, after restart | 63/75 | 74/75 | 74/75 | 74/75 | 39/66 | 55/66 | 60/66 | 65/66 | 0.917 | 0.58 / 0.73 | 62/75 |

Restart comparison (`final_restart_comparison.md`), 20 hit chunk ids per claim:

| before | after | identical (sequence) | identical (set) | article@10 | evidence@20 | gold page lost | gold page gained | claims that differ |
|---|---|---|---|---|---|---|---|---|
| final_ef_100 | final_ef_100_after_restart | 74/75 | 74/75 | 74/75 -> 74/75 | 65/66 -> 65/66 | 0 | 0 | 2835 |
| final_ef_20 | final_ef_20_after_restart | 73/75 | 73/75 | 74/75 -> 74/75 | 65/66 -> 65/66 | 0 | 0 | 2835, 46127 |

Reading: the headline run reaches article recall 63/75 / 74/75 / 74/75 / 74/75 at k 1 / 5 / 10 / 20 and evidence recall 39/66 / 55/66 / 60/66 / 65/66, at SQL p50 / p95 0.90 / 1.41 ms for k=10 and query embedding p50 4.67 ms; its article recall@10 is equal to the exact ranking's (74/75) and its evidence recall@20 equal to it (65/66), with hit lists identical to the exact ones for 74/75 claims (64/75 at ef 20) where the exact statement costs 2.63 ms p50. The rrf hybrid gives article recall@10 74/75 and evidence recall@20 64/66 at 10.84 ms p50, equal to the vector ranking alone in article recall@10 and below it in evidence recall@20; with the gold page known (oracle) the gold sentences are inside the first chunk for 42/66 claims, within 5 for 59/66 and within 20 for 66/66, at 2.65 / 9.87 ms. After the container restart the hit lists at ef 100 are identical to the pre-restart ones for 74 of 75 claims (article recall@10 74/75 -> 74/75, evidence recall@20 65/66 -> 65/66) and at ef 20 for 73 of 75 (74/75 -> 74/75, 65/66 -> 65/66), against 72/75 identical hit lists across the restart of the M=6 index in section 1.

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

Reading: the last ingest (4598 chunks, 1316 hatnote units excluded, 21.0 s of which 16.44 s embedding) passed every check: `ingest_meta` holds the defaults, the vector index is M=16 DISTANCE=cosine and mhnsw_max_cache_size is 536870912 bytes. This is the state `wikilense ingest` produces, so `wikilense eval` on it reproduces section 7 up to the HNSW build (a fresh graph).

## Chosen defaults and evidence

The defaults in the code are `sql/schema.sql` M=16, `Settings.ef_search` 100, `chunk_max_words` 240 with `chunk_overlap_units` 1, the embedding prefix on, `search()` strategy `inline` for filtered queries and `mhnsw_max_cache_size` 512 MB in docker-compose.yml. The evidence, section by section:

- **Index M = 16.** On the same vectors (section 3) the M=6 / 16 / 32 graphs give article recall@10 69/75 / 72/75 / 74/75 and evidence recall@20 55/66 / 61/66 / 63/66 at ef 20, against 74/75 / 63/66 for the exact ranking; at ef 100, the default it is paired with, 74/75 / 74/75 / 74/75 and 61/66 / 63/66 / 63/66, with 44/75 / 71/75 / 75/75 hit lists identical to the exact ranking; the builds took 1.13 / 2.94 / 8.98 s and the graphs are 16.0 MB / 17.0 MB / 19.0 MB. The M values whose article recall@10 and evidence recall@20 at ef 20 equal the exact ranking's: 32; M=32 builds in 8.98 s against 2.94 s for M=16 and moves article recall@10 by 0 claims and evidence recall@20 by 0 at ef 100.
- **mhnsw_ef_search = 100 per query.** On the M=6 graph (section 2) ef 20 / 100 / 400 give article recall@10 68/75 / 74/75 / 74/75 and evidence recall@20 58/66 / 64/66 / 63/66 for p50@10 0.56 / 0.64 / 0.95 ms. On the M=16 graph ef 100 makes 71/75 hit lists identical to the exact ranking against 62/75 at ef 20 (section 3), and on the final ingest (section 7) 74/75 against 64/75, for p50@10 0.90 against 0.69 ms; across the restart 74 hit lists stayed identical at ef 100 against 73 at ef 20.
- **chunk_max_words = 240, overlap 1.** At 1,200 nominal words (section 4) the 60 / 120 / 240-word chunkings give article recall 74/75 / 74/75 / 74/75 and evidence recall 54/66 / 57/66 / 55/66, at 2,400 words 74/75 / 74/75 / 74/75 and 60/66 / 63/66 / 60/66; 240 words store 4598 vectors against 8658 and 19065, a graph of 13.0 MB against 17.0 MB and 29.0 MB, and SQL p50 0.73 against 0.81 and 0.83 ms for the 1,200-word row.
- **Embedding prefix on.** The exact rankings (section 5) give evidence recall@20 63/66 with the prefix and 61/66 without, article recall@10 74/75 against 74/75, unit coverage@10 0.879 against 0.875: with the prefix, evidence recall@20 is above the value without it.
- **Filtered queries: strategy inline; overfetch when latency matters more than a full result.** Under the selective History filter (section 6) overfetch 10 returned fewer than 10 rows for 69 of 75 queries and overfetch 50 for 20, inline for 0, at p50@10 3.56 ms (p95 4.93 ms) against 1.05 / 2.76 ms; under the non-selective min_words filter the three give article recall@10 64/75 / 64/75 / 64/75 at 2.70 / 1.00 / 2.10 ms.
- **rrf hybrid: available, not the default.** On the final ingest (section 7) rrf gives article recall@10 74/75 and evidence recall@20 64/66 against 74/75 / 65/66 for the vector ranking alone, at p50@10 10.84 against 0.90 ms; on the OLD ingest (section 2) 74/75 / 60/66 against 74/75 / 64/66 at 15.87 against 0.64 ms.
- **mhnsw_max_cache_size = 512 MB, for capacity.** Within one server process the cache size changed nothing (section 1: 75 of 75 hit lists identical between 512 MB and 16 MB), while the container restart changed 3 of 75 hit lists on the M=6 graph and 1 of 75 on the M=16 graph at ef 100. The graph of the final ingest is 13.0 MB on disk and the 120-word ingest's 16.0 MB, already the size of the 16 MB server default.

