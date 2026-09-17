# WikiLense evaluation: strategy_overfetch10_minwords1000

Generated 2026-09-17T12:03:24+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [5, 10] |
| max_k | 10 |
| repeats | 5 |
| strategy | overfetch |
| overfetch | 10 |
| filters | {"min_words": 1000, "max_words": null, "heading_like": null, "path_like": null, "linked_from": null, "links_to": null, "titles": null} |
| claim_filters | - |
| ef_search | 20 |
| ef_search_effective | 20 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_device | cuda |
| n_claims | 75 |
| n_evidence_claims | 66 |
| n_pages | 100 |
| n_chunks | 8868 |
| n_sentences | 33637 |
| experiment_group | filters |
| eval_seconds | 2.9 |
| mhnsw_max_cache_size | 536870912 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 6 |
| vector_index_distance | cosine |
| chunk_data_length | 23642112 |
| chunk_index_length | 9273344 |
| vector_index_tablespace_bytes | 16777216 |
| chunk_words_mean | 93.0 |
| chunk_words_median | 103.0 |
| chunk_words_p95 | 119.0 |
| chunk_words_max | 174 |
| filter_ground_truth | {"filter": {"min_words": 1000}, "n_chunks": 8868, "n_chunks_passing": 8787, "n_pages_passing": 89, "claims_gold_page_excluded": 10, "claims_gold_page_excluded_ids": [5753, 5848, 14253, 26799, 32256, 54047, 63422, 74482, 89819, 93212], "article_recall_ceiling": 65, "n_claims": 75, "evidence_recall_ceiling": 56, "n_evidence_claims": 66} |
| short_results | {"5": {"queries_short_of_k": 0, "queries_with_no_rows": 0, "rows_min": 5, "rows_mean": 5.0, "short_claim_ids": []}, "10": {"queries_short_of_k": 0, "queries_with_no_rows": 0, "rows_min": 10, "rows_mean": 10.0, "short_claim_ids": []}} |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| chunk_max_words | 120 |
| chunk_overlap_units | 1 |
| corpus_claims_sha256 | a9a36c5c14f9a3baea97df6b42361f866249a4c636e4504cd259d7e54725b8ac |
| corpus_dir | /home/ylli/Desktop/Projects/WikiLense/wikilense-linux-20260916T222239Z-1-001/wikilense-linux/data/corpus |
| corpus_pages_sha256 | edf0955892c37fe54bb840cda33a6df5a06c44394289e7176a922556d09621dc |
| embedding_dim | 384 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_prefix | true |
| index_distance | cosine |
| index_m | 6 |
| ingested_at | 2026-09-17T12:03:13+00:00 |
| mariadb_version | 11.8.9-MariaDB-ubu2404 |
| wikilense_version | 0.1.0 |

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

## Article recall@k

A claim counts when a gold page is among the pages of its first k chunks.

| k | claims recalled | of | article recall |
|---|---|---|---|
| 5 | 63 | 75 | 0.840 |
| 10 | 64 | 75 | 0.853 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 5 | 35 | 66 | 0.530 |
| 10 | 43 | 66 | 0.652 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 5 | 66 | 0.549 |
| 10 | 66 | 0.674 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 5 | 375 | 1.03 | 1.46 | 1.06 | 0.72 | 2.15 |
| 10 | 375 | 1.20 | 1.77 | 1.25 | 0.86 | 2.32 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.67 | 4.87 | 4.70 | 4.49 | 6.13 |

## Claims with the worst rank of their gold page

Rank within the first 10 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 5753 | REFUTES | Asterales | not in top k | not in top k | Asterales is a species in the tribe Anthemideae, in the kingdom Plantae. |
| 5848 | SUPPORTS | Biennial plant | not in top k | not in top k | Some biennial plants have leaves that develop as basal rosettes. |
| 14253 | REFUTES | Asterales | not in top k | not in top k | Russowia belongs to the Asteraceae family of the Asterales order, an order of mo |
| 26799 | REFUTES | Monreith House | not in top k | not in top k | In the 1870s Monreith Estate covered about 16,000 acres (65 km), and the Monreit |
| 32256 | REFUTES | The Apache Software Foundation | not in top k | not in top k | The Apache Directory (an American nonprofit corporation (classified as a 501(c)( |
| 54047 | NOT ENOUGH INFO | Lamiales | not in top k | not in top k | Including 23,810 species, Lamiales holds erythranthe moschata. |
| 63422 | REFUTES | Kalmar Union | not in top k | not in top k | Iceland remained under Norwegian kingship until 1380; then Norway became part of |
| 74482 | REFUTES | Smithfield, Utah | not in top k | not in top k | Smithfield, Utah is the largest city in the area. |
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 89819 | SUPPORTS | Toshiki Kaifu | not in top k | not in top k | In 1989, Toshiki Kaifu (born January 2, 1931) is an LDP member who became the Pr |

Claims whose LIMIT k hits were not the first k of LIMIT max_k, per k: {"5": 3}.

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
