# WikiLense evaluation: s0_full_run2

Generated 2026-11-15T02:28:12+00:00. 3000 claims; 3000 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [1, 3, 5, 10, 20] |
| max_k | 20 |
| repeats | 5 |
| strategy | none |
| overfetch | - |
| filters | - |
| claim_filters | - |
| ef_search | 100 |
| ef_search_source | argument |
| ef_search_effective | 100 |
| mhnsw_max_cache_size | 536870912 |
| index_m | 16 |
| index_distance | cosine |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_device | cuda |
| n_claims | 3000 |
| n_evidence_claims | 3000 |
| n_pages | 3851 |
| n_chunks | 43134 |
| n_sentences | 284302 |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| analyze_seconds | 0.042 |
| analyze_tables | chunk,page,section,link,category,page_category |
| chunk_max_words | 240 |
| chunk_overlap_units | 1 |
| corpus_claims_sha256 | d29154f4cfe95ab5c2330f20ea05e2836fd2f23a9f4894b25fcc2c89e946297c |
| corpus_dir | (outside the repository) scaled_data/S0 |
| corpus_page_meta_sha256 | none |
| corpus_pages_sha256 | 18f1acff74165674074ad75f7f03ddd525a12db2500577b3368d40013f6e5ea7 |
| embedding_dim | 384 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_prefix | true |
| embedding_revision | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a |
| hatnote_pattern | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\\b\|instance\\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) |
| index_distance | cosine |
| index_m | 16 |
| ingested_at | 2026-11-12T02:13:51+00:00 |
| mariadb_version | 11.8.9-MariaDB-ubu2404 |
| n_units_hatnote | 6117 |
| wikilense_version | 0.1.0 |

## Machine and versions

| Item | Value |
|---|---|
| platform | Linux-7.0.0-43-generic-x86_64-with-glibc2.43 |
| architecture | x86_64 |
| cpu | 13th Gen Intel(R) Core(TM) i9-13900H |
| cpu_count | 20 |
| memory_gb | 14.7 |
| embedding_device | cuda |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU |
| python | 3.12.16 |
| wikilense | 0.1.0 |
| mariadb | 11.8.9-MariaDB-ubu2404 |
| PyMySQL | 1.2.3 |
| numpy | 2.5.3 |
| torch | 2.14.0 |
| sentence-transformers | 6.0.1 |

## Article recall@k

A claim counts when a gold page is among the pages of its first k chunks.

| k | claims recalled | of | article recall |
|---|---|---|---|
| 1 | 2895 | 3000 | 0.965 |
| 3 | 2937 | 3000 | 0.979 |
| 5 | 2951 | 3000 | 0.984 |
| 10 | 2959 | 3000 | 0.986 |
| 20 | 2971 | 3000 | 0.990 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 1337 | 3000 | 0.446 |
| 3 | 2108 | 3000 | 0.703 |
| 5 | 2341 | 3000 | 0.780 |
| 10 | 2558 | 3000 | 0.853 |
| 20 | 2678 | 3000 | 0.893 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 3000 | 0.628 |
| 3 | 3000 | 0.834 |
| 5 | 3000 | 0.884 |
| 10 | 3000 | 0.926 |
| 20 | 3000 | 0.948 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 15000 | 1.27 | 1.73 | 1.28 | 0.61 | 4.22 |
| 3 | 15000 | 1.13 | 1.56 | 1.14 | 0.50 | 4.53 |
| 5 | 15000 | 1.11 | 1.52 | 1.11 | 0.52 | 3.66 |
| 10 | 15000 | 1.22 | 1.70 | 1.24 | 0.60 | 143.55 |
| 20 | 15000 | 1.41 | 2.06 | 1.45 | 0.73 | 4.57 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 15000 | 6.15 | 6.33 | 6.17 | 5.90 | 47.10 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 893 | NOT ENOUGH INFO | Kelly Holcomb | not in top k | not in top k | In 2005, the running game was the main highlight of an otherwise stagnant team l |
| 1028 | NOT ENOUGH INFO | Spring Cup (disambiguation) | not in top k | not in top k | In the Spring Cup four teams would play against each other to qualify for the ne |
| 3274 | NOT ENOUGH INFO | Suszki | not in top k | not in top k | Dürrkunzendorf united with the County of Glatz in 1763 before the reorganization |
| 5589 | SUPPORTS | Ditrigona obliquilinea | not in top k | not in top k | The wingspan of the Corycia pnocaria is thirteen to twenty millimeters for males |
| 7492 | NOT ENOUGH INFO | Vector | not in top k | not in top k | The weather on day 0 (today) is known to be sunny, which is represented by a vec |
| 7733 | NOT ENOUGH INFO | Dwight Henry | not in top k | not in top k | Democrat Newd McWherter beat Republican Dwight Henry, an unknown one term member |
| 9447 | NOT ENOUGH INFO | NCAA tournament | not in top k | not in top k | Kentucky Wildcats has the most number of NCAA Tourney wins with 121 wins and 55 |
| 15206 | NOT ENOUGH INFO | M6 | not in top k | not in top k | Most of the area west of the M6 is residential, while area to the south contains |
| 18123 | NOT ENOUGH INFO | FOX proteins | not in top k | not in top k | The presence of FoxO in the human body is a hoax. |
| 22196 | NOT ENOUGH INFO | Speaker of the United States House of Representatives | not in top k | not in top k | Democrat Eni Faleomavaega was the most senior member of the United States House |

Claims whose LIMIT k hits were not the first k of LIMIT max_k, per k: {"3": 1, "10": 2}.

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the metrics at k use the hits of the LIMIT k search of each claim (first timed pass), in the order search() returned them; the per-claim ranks use the LIMIT max_k search.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
