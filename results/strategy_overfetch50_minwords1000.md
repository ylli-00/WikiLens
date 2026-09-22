# WikiLense evaluation: strategy_overfetch50_minwords1000

Generated 2026-09-22T01:40:52+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [1, 3, 5, 10, 20] |
| max_k | 20 |
| repeats | 5 |
| strategy | overfetch |
| overfetch | 50 |
| filters | {"min_words": 1000, "max_words": null, "heading_like": null, "path_like": null, "linked_from": null, "links_to": null, "titles": null} |
| claim_filters | - |
| ef_search | 100 |
| ef_search_source | argument |
| ef_search_effective | 100 |
| mhnsw_max_cache_size | 536870912 |
| index_m | 16 |
| index_distance | cosine |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_device | cuda |
| n_claims | 75 |
| n_evidence_claims | 66 |
| n_pages | 100 |
| n_chunks | 4598 |
| n_sentences | 33637 |
| experiment_group | filters |
| configuration | {"chunk_max_words": 240, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": true, "label": "240 words, overlap 1, M=16, prefix on"} |
| eval_seconds | 6.1 |
| n_units_hatnote | 1316 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 16 |
| vector_index_distance | cosine |
| chunk_data_length | 14172160 |
| chunk_index_length | 6111232 |
| vector_index_tablespace_bytes | 13631488 |
| chunk_words_mean | 156.6 |
| chunk_words_median | 183.5 |
| chunk_words_p95 | 238.0 |
| chunk_words_max | 240 |
| filter_ground_truth | {"filter": {"min_words": 1000}, "n_chunks": 4598, "n_chunks_passing": 4544, "n_pages_passing": 89, "claims_gold_page_excluded": 10, "claims_gold_page_excluded_ids": [5753, 5848, 14253, 26799, 32256, 54047, 63422, 74482, 89819, 93212], "article_recall_ceiling": 65, "n_claims": 75, "evidence_recall_ceiling": 56, "n_evidence_claims": 66} |
| short_results | {"1": {"queries_short_of_k": 0, "queries_with_no_rows": 0, "rows_min": 1, "rows_mean": 1.0, "short_claim_ids": []}, "3": {"queries_short_of_k": 0, "queries_with_no_rows": 0, "rows_min": 3, "rows_mean": 3.0, "short_claim_ids": []}, "5": {"queries_short_of_k": 0, "queries_with_no_rows": 0, "rows_min": 5, "rows_mean": 5.0, "short_claim_ids": []}, "10": {"queries_short_of_k": 0, "queries_with_no_rows": 0, "rows_min": 10, "rows_mean": 10.0, "short_claim_ids": []}, "20": {"queries_short_of_k": 0, "queries_with_no_rows": 0, "rows_min": 20, "rows_mean": 20.0, "short_claim_ids": []}} |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| analyze_seconds | 0.009 |
| analyze_tables | chunk,page,section,link |
| chunk_max_words | 240 |
| chunk_overlap_units | 1 |
| corpus_claims_sha256 | a9a36c5c14f9a3baea97df6b42361f866249a4c636e4504cd259d7e54725b8ac |
| corpus_dir | data/corpus |
| corpus_pages_sha256 | edf0955892c37fe54bb840cda33a6df5a06c44394289e7176a922556d09621dc |
| embedding_dim | 384 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_prefix | true |
| hatnote_pattern | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\\b\|instance\\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) |
| index_distance | cosine |
| index_m | 16 |
| ingested_at | 2026-09-22T01:40:31+00:00 |
| mariadb_version | 11.8.9-MariaDB-ubu2404 |
| n_units_hatnote | 1316 |
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
| 1 | 59 | 75 | 0.787 |
| 3 | 64 | 75 | 0.853 |
| 5 | 64 | 75 | 0.853 |
| 10 | 64 | 75 | 0.853 |
| 20 | 64 | 75 | 0.853 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 33 | 66 | 0.500 |
| 3 | 43 | 66 | 0.652 |
| 5 | 46 | 66 | 0.697 |
| 10 | 50 | 66 | 0.758 |
| 20 | 55 | 66 | 0.833 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.500 |
| 3 | 66 | 0.672 |
| 5 | 66 | 0.716 |
| 10 | 66 | 0.765 |
| 20 | 66 | 0.833 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 1.13 | 1.42 | 1.16 | 0.72 | 2.12 |
| 3 | 375 | 1.34 | 1.65 | 1.36 | 0.93 | 2.42 |
| 5 | 375 | 1.56 | 1.88 | 1.60 | 1.09 | 2.88 |
| 10 | 375 | 2.10 | 2.48 | 2.16 | 1.56 | 4.09 |
| 20 | 375 | 3.06 | 3.63 | 3.15 | 2.57 | 5.92 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.69 | 4.79 | 4.69 | 4.57 | 5.05 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

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

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
