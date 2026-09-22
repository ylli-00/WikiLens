# WikiLense evaluation: stability_512mb_run1

Generated 2026-09-22T01:34:54+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| ef_search | 20 |
| ef_search_source | argument |
| ef_search_effective | 20 |
| mhnsw_max_cache_size | 536870912 |
| index_m | 6 |
| index_distance | cosine |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_device | cuda |
| n_claims | 75 |
| n_evidence_claims | 66 |
| n_pages | 100 |
| n_chunks | 8658 |
| n_sentences | 33637 |
| experiment_group | stability |
| configuration | {"chunk_max_words": 120, "chunk_overlap_units": 1, "index_m": 6, "use_prefix": true, "label": "120 words, overlap 1, M=6, prefix on"} |
| eval_seconds | 3.2 |
| n_units_hatnote | 1316 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 6 |
| vector_index_distance | cosine |
| chunk_data_length | 22593536 |
| chunk_index_length | 524288 |
| vector_index_tablespace_bytes | 16777216 |
| chunk_words_mean | 93.8 |
| chunk_words_median | 104.0 |
| chunk_words_p95 | 119.0 |
| chunk_words_max | 174 |
| ingest_run | stability_old |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| analyze_seconds | 0.008 |
| analyze_tables | chunk,page,section,link |
| chunk_max_words | 120 |
| chunk_overlap_units | 1 |
| corpus_claims_sha256 | a9a36c5c14f9a3baea97df6b42361f866249a4c636e4504cd259d7e54725b8ac |
| corpus_dir | data/corpus |
| corpus_pages_sha256 | edf0955892c37fe54bb840cda33a6df5a06c44394289e7176a922556d09621dc |
| embedding_dim | 384 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_prefix | true |
| hatnote_pattern | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\\b\|instance\\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) |
| index_distance | cosine |
| index_m | 6 |
| ingested_at | 2026-09-22T01:34:49+00:00 |
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
| 1 | 62 | 75 | 0.827 |
| 3 | 67 | 75 | 0.893 |
| 5 | 68 | 75 | 0.907 |
| 10 | 68 | 75 | 0.907 |
| 20 | 68 | 75 | 0.907 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 35 | 66 | 0.530 |
| 3 | 43 | 66 | 0.652 |
| 5 | 46 | 66 | 0.697 |
| 10 | 53 | 66 | 0.803 |
| 20 | 58 | 66 | 0.879 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.543 |
| 3 | 66 | 0.669 |
| 5 | 66 | 0.716 |
| 10 | 66 | 0.811 |
| 20 | 66 | 0.879 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.53 | 0.88 | 0.57 | 0.43 | 1.19 |
| 3 | 375 | 0.46 | 0.86 | 0.51 | 0.40 | 1.27 |
| 5 | 375 | 0.48 | 0.87 | 0.53 | 0.42 | 1.26 |
| 10 | 375 | 0.54 | 0.95 | 0.59 | 0.47 | 1.36 |
| 20 | 375 | 0.70 | 1.18 | 0.75 | 0.57 | 1.54 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.66 | 4.77 | 4.67 | 4.54 | 5.55 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 9154 | REFUTES | Asia | not in top k | not in top k | A total of 22 foreign NBA players came from Asia, coming from seven of the 48 As |
| 11910 | SUPPORTS | Jimmy Carter | not in top k | not in top k | Jimmy Carter won the 1980 New Hampshire Primary vote, but lost the Presidential |
| 13383 | REFUTES | London | not in top k | not in top k | Jackie Tyler is introduced in "Rose", she is attacked by shop window dummies and |
| 32256 | REFUTES | The Apache Software Foundation | not in top k | not in top k | The Apache Directory (an American nonprofit corporation (classified as a 501(c)( |
| 69200 | REFUTES | Bangladesh | not in top k | n/a | Uttar Badepasha ( Sylhet District, Bangladesh (capital New York)) had ten names |
| 79299 | REFUTES | Antlia | not in top k | not in top k | Zeta Antliae is in the Antlia Constellation (originally Antlia Pneumatica, estab |
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 76756 | SUPPORTS | Asteraceae | 4 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 22563 | REFUTES | Asteraceae | 3 | 16 | Genus Hyoseris is classified under tribe Cichorieae,  a tribe in the plant famil |
| 85631 | REFUTES | Asteraceae | 3 | 3 | Plantae kingdom Diplazoptilon are classified in the Asteraceae family, a very la |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
