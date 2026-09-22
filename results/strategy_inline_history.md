# WikiLense evaluation: strategy_inline_history

Generated 2026-09-22T01:41:04+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [1, 3, 5, 10, 20] |
| max_k | 20 |
| repeats | 5 |
| strategy | inline |
| overfetch | - |
| filters | {"min_words": null, "max_words": null, "heading_like": "%History%", "path_like": null, "linked_from": null, "links_to": null, "titles": null} |
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
| eval_seconds | 9.7 |
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
| filter_ground_truth | {"filter": {"heading_like": "%History%"}, "n_chunks": 4598, "n_chunks_passing": 117, "n_pages_passing": 66, "claims_gold_page_excluded": 37, "claims_gold_page_excluded_ids": [3482, 5753, 5848, 6419, 11910, 13207, 14253, 14727, 14909, 17676, 22563, 25324, 28863, 30562, 31383, 40707, 42937, 45793, 46127, 48999, 54047, 56899, 63422, 64595, 65842, 66646, 69572, 76756, 82504, 82761, 85631, 85881, 89819, 90340, 93212, 94092, 95085], "article_recall_ceiling": 38, "n_claims": 75, "evidence_recall_ceiling": 0, "n_evidence_claims": 66} |
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
| 1 | 21 | 75 | 0.280 |
| 3 | 24 | 75 | 0.320 |
| 5 | 25 | 75 | 0.333 |
| 10 | 25 | 75 | 0.333 |
| 20 | 25 | 75 | 0.333 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 0 | 66 | 0.000 |
| 3 | 0 | 66 | 0.000 |
| 5 | 0 | 66 | 0.000 |
| 10 | 0 | 66 | 0.000 |
| 20 | 0 | 66 | 0.000 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.000 |
| 3 | 66 | 0.000 |
| 5 | 66 | 0.000 |
| 10 | 66 | 0.000 |
| 20 | 66 | 0.000 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 2.81 | 3.36 | 2.87 | 2.28 | 10.01 |
| 3 | 375 | 2.96 | 3.65 | 3.03 | 2.37 | 9.14 |
| 5 | 375 | 3.14 | 3.98 | 3.20 | 2.33 | 6.36 |
| 10 | 375 | 3.56 | 4.93 | 3.70 | 2.56 | 7.77 |
| 20 | 375 | 4.36 | 6.08 | 4.53 | 2.87 | 13.31 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.66 | 4.75 | 4.67 | 4.56 | 5.00 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 3482 | REFUTES | Jack Brabham | not in top k | not in top k | Jack Brabham, the winner of the 1966 United States Grand Prix at Watkins Glen, b |
| 4212 | REFUTES | Arsenal F.C. | not in top k | not in top k | In Chuks Aneke's career from 2011 to 2021, he was part of the Arsenal (football |
| 5753 | REFUTES | Asterales | not in top k | not in top k | Asterales is a species in the tribe Anthemideae, in the kingdom Plantae. |
| 5848 | SUPPORTS | Biennial plant | not in top k | not in top k | Some biennial plants have leaves that develop as basal rosettes. |
| 6419 | REFUTES | Kim Philby | not in top k | not in top k | Kim Philby, part of the Cambridge Five, was an effective American spy while he w |
| 7971 | REFUTES | Kabul | not in top k | not in top k | Urban decay is a process by which a city falls into a state of disrepair and neg |
| 11235 | REFUTES | London Underground | not in top k | not in top k | London's Underground is one of the few Railway electrification systems that util |
| 11435 | NOT ENOUGH INFO | Economy of Azerbaijan | not in top k | not in top k | Azerbaijan's primary economic sectors were oil, gas, chemical, light industries, |
| 11910 | SUPPORTS | Jimmy Carter | not in top k | not in top k | Jimmy Carter won the 1980 New Hampshire Primary vote, but lost the Presidential |
| 13207 | REFUTES | The Hunger Games (film) | not in top k | not in top k | Leigha Hancock appeared in The Hunger Games (2012), which Gary Valenciano direct |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
