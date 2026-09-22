# WikiLense evaluation: chunk60_ef_100

Generated 2026-09-22T01:37:54+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [1, 2, 3, 5, 6, 10, 20, 40] |
| max_k | 40 |
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
| n_chunks | 19065 |
| n_sentences | 33637 |
| experiment_group | chunk_size |
| configuration | {"chunk_max_words": 60, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": true, "label": "60 words, overlap 1, M=16, prefix on"} |
| eval_seconds | 4.9 |
| n_units_hatnote | 1316 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 16 |
| vector_index_distance | cosine |
| chunk_data_length | 41484288 |
| chunk_index_length | 1064960 |
| vector_index_tablespace_bytes | 30408704 |
| chunk_words_mean | 48.1 |
| chunk_words_median | 50.0 |
| chunk_words_p95 | 60.0 |
| chunk_words_max | 174 |
| ingest_run | {"label": "chunk60", "kind": "ingest", "configuration": {"chunk_max_words": 60, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": true, "label": "60 words, overlap 1, M=16, prefix on"}, "ingested_at": "2026-09-22T01:37:48+00:00", "counts": {"n_pages": 100, "n_sections": 3173, "n_sentences": 33637, "n_units_empty": 372, "n_units_hatnote": 1316, "n_chunks": 19065, "n_links": 40542, "n_links_resolved": 373, "n_links_skipped": 0, "n_claims": 75, "n_evidence": 114, "n_evidence_page_resolved": 114, "n_evidence_sentence_resolved": 87}, "seconds": {"schema": 0.1, "parse": 0.87, "embed": 22.51, "load": 10.34, "resolve": 0.02, "analyze": 0.02, "total": 33.92}, "n_units_hatnote": 1316, "chunk_words_mean": 48.1, "chunk_words_median": 50.0, "chunk_words_p95": 60.0, "chunk_words_max": 174, "vector_index_name": "embedding", "vector_index_m": 16, "vector_index_distance": "cosine", "chunk_data_length": 41484288, "chunk_index_length": 1064960, "vector_index_tablespace_bytes": 30408704, "index_rebuild": null} |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| analyze_seconds | 0.025 |
| analyze_tables | chunk,page,section,link |
| chunk_max_words | 60 |
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
| ingested_at | 2026-09-22T01:37:48+00:00 |
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
| 1 | 66 | 75 | 0.880 |
| 2 | 67 | 75 | 0.893 |
| 3 | 71 | 75 | 0.947 |
| 5 | 74 | 75 | 0.987 |
| 6 | 74 | 75 | 0.987 |
| 10 | 74 | 75 | 0.987 |
| 20 | 74 | 75 | 0.987 |
| 40 | 74 | 75 | 0.987 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 29 | 66 | 0.439 |
| 2 | 34 | 66 | 0.515 |
| 3 | 39 | 66 | 0.591 |
| 5 | 43 | 66 | 0.652 |
| 6 | 44 | 66 | 0.667 |
| 10 | 48 | 66 | 0.727 |
| 20 | 54 | 66 | 0.818 |
| 40 | 60 | 66 | 0.909 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.480 |
| 2 | 66 | 0.561 |
| 3 | 66 | 0.631 |
| 5 | 66 | 0.696 |
| 6 | 66 | 0.711 |
| 10 | 66 | 0.771 |
| 20 | 66 | 0.847 |
| 40 | 66 | 0.932 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.75 | 0.97 | 0.75 | 0.53 | 1.28 |
| 2 | 375 | 0.65 | 0.85 | 0.66 | 0.46 | 1.19 |
| 3 | 375 | 0.63 | 0.82 | 0.65 | 0.46 | 1.22 |
| 5 | 375 | 0.64 | 0.85 | 0.66 | 0.48 | 1.33 |
| 6 | 375 | 0.66 | 0.85 | 0.67 | 0.49 | 1.22 |
| 10 | 375 | 0.72 | 0.90 | 0.73 | 0.53 | 1.24 |
| 20 | 375 | 0.83 | 1.06 | 0.85 | 0.65 | 1.49 |
| 40 | 375 | 1.06 | 1.27 | 1.08 | 0.87 | 2.08 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.76 | 5.03 | 4.79 | 4.58 | 7.16 |

## Claims with the worst rank of their gold page

Rank within the first 40 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 22563 | REFUTES | Asteraceae | 4 | 26 | Genus Hyoseris is classified under tribe Cichorieae,  a tribe in the plant famil |
| 76756 | SUPPORTS | Asteraceae | 4 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 95085 | SUPPORTS | Asteraceae | 4 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 36109 | REFUTES | Lebanese Armed Forces | 3 | n/a | The May 17 Agreement called for the Lebanese Army (founded neither on August 1, |
| 40707 | NOT ENOUGH INFO | American Civil War | 3 | 12 | Following the American Civil War, the Army and Navy were unsupportive of each ot |
| 43675 | REFUTES | Los Angeles | 3 | 4 | Annie Duke born September 13, 1965 lives in Los Angeles, California, U.S. (the s |
| 85631 | REFUTES | Asteraceae | 3 | 3 | Plantae kingdom Diplazoptilon are classified in the Asteraceae family, a very la |
| 14253 | REFUTES | Asterales | 2 | 2 | Russowia belongs to the Asteraceae family of the Asterales order, an order of mo |
| 854 | REFUTES | Canton of Aargau | 1 | 12 | Ehrendingen is a municipality in the district of Baden in the canton of Aargau i |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
