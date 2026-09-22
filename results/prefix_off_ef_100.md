# WikiLense evaluation: prefix_off_ef_100

Generated 2026-09-22T01:39:58+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| n_claims | 75 |
| n_evidence_claims | 66 |
| n_pages | 100 |
| n_chunks | 8658 |
| n_sentences | 33637 |
| experiment_group | prefix |
| configuration | {"chunk_max_words": 120, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": false, "label": "120 words, overlap 1, M=16, prefix off"} |
| eval_seconds | 4.0 |
| n_units_hatnote | 1316 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 16 |
| vector_index_distance | cosine |
| chunk_data_length | 22593536 |
| chunk_index_length | 524288 |
| vector_index_tablespace_bytes | 17825792 |
| chunk_words_mean | 93.8 |
| chunk_words_median | 104.0 |
| chunk_words_p95 | 119.0 |
| chunk_words_max | 174 |
| ingest_run | {"label": "prefix_off", "kind": "ingest", "configuration": {"chunk_max_words": 120, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": false, "label": "120 words, overlap 1, M=16, prefix off"}, "ingested_at": "2026-09-22T01:39:53+00:00", "counts": {"n_pages": 100, "n_sections": 3173, "n_sentences": 33637, "n_units_empty": 372, "n_units_hatnote": 1316, "n_chunks": 8658, "n_links": 40542, "n_links_resolved": 373, "n_links_skipped": 0, "n_claims": 75, "n_evidence": 114, "n_evidence_page_resolved": 114, "n_evidence_sentence_resolved": 87}, "seconds": {"schema": 0.1, "parse": 0.86, "embed": 17.19, "load": 5.82, "resolve": 0.0, "analyze": 0.01, "total": 24.03}, "n_units_hatnote": 1316, "chunk_words_mean": 93.8, "chunk_words_median": 104.0, "chunk_words_p95": 119.0, "chunk_words_max": 174, "vector_index_name": "embedding", "vector_index_m": 16, "vector_index_distance": "cosine", "chunk_data_length": 22593536, "chunk_index_length": 524288, "vector_index_tablespace_bytes": 17825792, "index_rebuild": null} |

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
| embedding_prefix | false |
| hatnote_pattern | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\\b\|instance\\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) |
| index_distance | cosine |
| index_m | 16 |
| ingested_at | 2026-09-22T01:39:53+00:00 |
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
| 1 | 65 | 75 | 0.867 |
| 3 | 73 | 75 | 0.973 |
| 5 | 74 | 75 | 0.987 |
| 10 | 74 | 75 | 0.987 |
| 20 | 74 | 75 | 0.987 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 36 | 66 | 0.545 |
| 3 | 46 | 66 | 0.697 |
| 5 | 48 | 66 | 0.727 |
| 10 | 56 | 66 | 0.848 |
| 20 | 61 | 66 | 0.924 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.558 |
| 3 | 66 | 0.721 |
| 5 | 66 | 0.759 |
| 10 | 66 | 0.875 |
| 20 | 66 | 0.939 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.82 | 1.20 | 0.89 | 0.63 | 1.84 |
| 3 | 375 | 0.82 | 1.24 | 0.89 | 0.56 | 1.77 |
| 5 | 375 | 0.84 | 1.27 | 0.91 | 0.58 | 1.93 |
| 10 | 375 | 0.90 | 1.32 | 0.97 | 0.65 | 1.99 |
| 20 | 375 | 1.03 | 1.48 | 1.11 | 0.77 | 2.39 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.71 | 4.89 | 4.73 | 4.56 | 5.70 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 95085 | SUPPORTS | Asteraceae | 4 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 9154 | REFUTES | Asia | 3 | 9 | A total of 22 foreign NBA players came from Asia, coming from seven of the 48 As |
| 13383 | REFUTES | London | 3 | 3 | Jackie Tyler is introduced in "Rose", she is attacked by shop window dummies and |
| 40707 | NOT ENOUGH INFO | American Civil War | 3 | 9 | Following the American Civil War, the Army and Navy were unsupportive of each ot |
| 11235 | REFUTES | London Underground | 2 | 14 | London's Underground is one of the few Railway electrification systems that util |
| 14253 | REFUTES | Asterales | 2 | 5 | Russowia belongs to the Asteraceae family of the Asterales order, an order of mo |
| 22563 | REFUTES | Asteraceae | 2 | 12 | Genus Hyoseris is classified under tribe Cichorieae,  a tribe in the plant famil |
| 76756 | SUPPORTS | Asteraceae | 2 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 85631 | REFUTES | Asteraceae | 2 | 2 | Plantae kingdom Diplazoptilon are classified in the Asteraceae family, a very la |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
