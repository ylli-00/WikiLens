# WikiLense evaluation: noprefix_ef_20

Generated 2026-09-17T12:02:50+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| ef_search_effective | 20 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_device | cuda |
| n_claims | 75 |
| n_evidence_claims | 66 |
| n_pages | 100 |
| n_chunks | 8868 |
| n_sentences | 33637 |
| experiment_group | prefix |
| eval_seconds | 3.8 |
| mhnsw_max_cache_size | 536870912 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 6 |
| vector_index_distance | cosine |
| chunk_data_length | 16384 |
| chunk_index_length | 49152 |
| vector_index_tablespace_bytes | 16777216 |
| chunk_words_mean | 93.0 |
| chunk_words_median | 103.0 |
| chunk_words_p95 | 119.0 |
| chunk_words_max | 174 |
| ingest_report | {"label": "noprefix", "chunk_max_words": 120, "chunk_overlap_units": 1, "use_prefix": false, "ingested_at": "2026-09-17T12:02:42+00:00", "counts": {"n_pages": 100, "n_sections": 3173, "n_sentences": 33637, "n_units_empty": 372, "n_chunks": 8868, "n_links": 40542, "n_links_resolved": 373, "n_links_skipped": 0, "n_claims": 75, "n_evidence": 114, "n_evidence_page_resolved": 114, "n_evidence_sentence_resolved": 87}, "seconds": {"schema": 0.08, "parse": 0.87, "embed": 17.79, "load": 3.17, "resolve": 0.0, "total": 21.97}, "chunk_words_mean": 93.0, "chunk_words_median": 103.0, "chunk_words_p95": 119.0, "chunk_words_max": 174, "vector_index_m": 6, "chunk_data_length": 16384, "chunk_index_length": 49152, "vector_index_tablespace_bytes": 16777216} |

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
| embedding_prefix | false |
| index_distance | cosine |
| index_m | 6 |
| ingested_at | 2026-09-17T12:02:42+00:00 |
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
| 1 | 60 | 75 | 0.800 |
| 3 | 67 | 75 | 0.893 |
| 5 | 68 | 75 | 0.907 |
| 10 | 69 | 75 | 0.920 |
| 20 | 70 | 75 | 0.933 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 32 | 66 | 0.485 |
| 3 | 39 | 66 | 0.591 |
| 5 | 43 | 66 | 0.652 |
| 10 | 50 | 66 | 0.758 |
| 20 | 53 | 66 | 0.803 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.497 |
| 3 | 66 | 0.615 |
| 5 | 66 | 0.688 |
| 10 | 66 | 0.788 |
| 20 | 66 | 0.826 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.64 | 1.15 | 0.71 | 0.48 | 2.70 |
| 3 | 375 | 0.64 | 1.17 | 0.69 | 0.44 | 1.61 |
| 5 | 375 | 0.65 | 1.17 | 0.70 | 0.44 | 1.62 |
| 10 | 375 | 0.72 | 1.24 | 0.77 | 0.50 | 2.36 |
| 20 | 375 | 0.85 | 1.45 | 0.92 | 0.64 | 3.25 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.71 | 4.98 | 4.75 | 4.48 | 7.21 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 5121 | REFUTES | Association for Computing Machinery | not in top k | not in top k | Upsilon Pi Epsilon is a member of the ACHS - Association of College Honor Societ |
| 34024 | REFUTES | Anime | not in top k | not in top k | Panda and the Magic Serpent, also known as The Tale of the White Serpent, is the |
| 43675 | REFUTES | Los Angeles | not in top k | not in top k | Annie Duke born September 13, 1965 lives in Los Angeles, California, U.S. (the s |
| 85881 | REFUTES | Asteraceae | not in top k | not in top k | Moonia belongs to the tribe Coreopsideae under the family Asteraceae which inclu |
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 13383 | REFUTES | London | 14 | not in top k | Jackie Tyler is introduced in "Rose", she is attacked by shop window dummies and |
| 19145 | REFUTES | British Columbia | 6 | not in top k | Westshore velodrome a 33m outdoor bicycle racing track located in Colwood city ( |
| 95085 | SUPPORTS | Asteraceae | 4 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 9154 | REFUTES | Asia | 2 | 3 | A total of 22 foreign NBA players came from Asia, coming from seven of the 48 As |
| 11235 | REFUTES | London Underground | 2 | 14 | London's Underground is one of the few Railway electrification systems that util |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
