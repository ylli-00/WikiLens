# WikiLense evaluation: chunk240_ef_100

Generated 2026-09-17T12:02:20+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [1, 2, 3, 5, 10] |
| max_k | 10 |
| repeats | 5 |
| strategy | none |
| overfetch | - |
| filters | - |
| claim_filters | - |
| ef_search | 100 |
| ef_search_effective | 100 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_device | cuda |
| n_claims | 75 |
| n_evidence_claims | 66 |
| n_pages | 100 |
| n_chunks | 4751 |
| n_sentences | 33637 |
| experiment_group | chunk_size |
| eval_seconds | 3.9 |
| mhnsw_max_cache_size | 536870912 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 6 |
| vector_index_distance | cosine |
| chunk_data_length | 49152 |
| chunk_index_length | 49152 |
| vector_index_tablespace_bytes | 12582912 |
| chunk_words_mean | 153.9 |
| chunk_words_median | 181.0 |
| chunk_words_p95 | 238.0 |
| chunk_words_max | 240 |
| ingest_report | {"label": "chunk240", "chunk_max_words": 240, "chunk_overlap_units": 1, "use_prefix": true, "ingested_at": "2026-09-17T12:02:15+00:00", "counts": {"n_pages": 100, "n_sections": 3173, "n_sentences": 33637, "n_units_empty": 372, "n_chunks": 4751, "n_links": 40542, "n_links_resolved": 373, "n_links_skipped": 0, "n_claims": 75, "n_evidence": 114, "n_evidence_page_resolved": 114, "n_evidence_sentence_resolved": 87}, "seconds": {"schema": 0.08, "parse": 0.9, "embed": 17.43, "load": 2.32, "resolve": 0.0, "total": 20.79}, "chunk_words_mean": 153.9, "chunk_words_median": 181.0, "chunk_words_p95": 238.0, "chunk_words_max": 240, "vector_index_m": 6, "chunk_data_length": 49152, "chunk_index_length": 49152, "vector_index_tablespace_bytes": 12582912} |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| chunk_max_words | 240 |
| chunk_overlap_units | 1 |
| corpus_claims_sha256 | a9a36c5c14f9a3baea97df6b42361f866249a4c636e4504cd259d7e54725b8ac |
| corpus_dir | /home/ylli/Desktop/Projects/WikiLense/wikilense-linux-20260916T222239Z-1-001/wikilense-linux/data/corpus |
| corpus_pages_sha256 | edf0955892c37fe54bb840cda33a6df5a06c44394289e7176a922556d09621dc |
| embedding_dim | 384 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_prefix | true |
| index_distance | cosine |
| index_m | 6 |
| ingested_at | 2026-09-17T12:02:15+00:00 |
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
| 1 | 62 | 75 | 0.827 |
| 2 | 70 | 75 | 0.933 |
| 3 | 73 | 75 | 0.973 |
| 5 | 73 | 75 | 0.973 |
| 10 | 73 | 75 | 0.973 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 35 | 66 | 0.530 |
| 2 | 43 | 66 | 0.652 |
| 3 | 48 | 66 | 0.727 |
| 5 | 49 | 66 | 0.742 |
| 10 | 55 | 66 | 0.833 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.543 |
| 2 | 66 | 0.672 |
| 3 | 66 | 0.747 |
| 5 | 66 | 0.761 |
| 10 | 66 | 0.848 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.69 | 1.44 | 0.80 | 0.49 | 1.89 |
| 2 | 375 | 0.67 | 1.43 | 0.76 | 0.45 | 1.85 |
| 3 | 375 | 0.67 | 1.44 | 0.75 | 0.45 | 1.88 |
| 5 | 375 | 0.69 | 1.46 | 0.78 | 0.48 | 1.81 |
| 10 | 375 | 0.79 | 1.58 | 0.88 | 0.54 | 1.97 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.66 | 4.83 | 4.69 | 4.48 | 8.44 |

## Claims with the worst rank of their gold page

Rank within the first 10 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 43675 | REFUTES | Los Angeles | not in top k | not in top k | Annie Duke born September 13, 1965 lives in Los Angeles, California, U.S. (the s |
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 76756 | SUPPORTS | Asteraceae | 3 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 85631 | REFUTES | Asteraceae | 3 | 3 | Plantae kingdom Diplazoptilon are classified in the Asteraceae family, a very la |
| 95085 | SUPPORTS | Asteraceae | 3 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 7971 | REFUTES | Kabul | 2 | not in top k | Urban decay is a process by which a city falls into a state of disrepair and neg |
| 9154 | REFUTES | Asia | 2 | not in top k | A total of 22 foreign NBA players came from Asia, coming from seven of the 48 As |
| 11439 | REFUTES | Juventus F.C. | 2 | 2 | Roma, colloquially known as Juve (pronounced [ˈjuːve), is a professional footbal |
| 13383 | REFUTES | London | 2 | 3 | Jackie Tyler is introduced in "Rose", she is attacked by shop window dummies and |
| 14253 | REFUTES | Asterales | 2 | 3 | Russowia belongs to the Asteraceae family of the Asterales order, an order of mo |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
