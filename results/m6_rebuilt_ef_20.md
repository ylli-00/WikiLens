# WikiLense evaluation: m6_rebuilt_ef_20

Generated 2026-09-17T12:01:21+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| experiment_group | index_m |
| eval_seconds | 3.5 |
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
| index_rebuild | {"m": 6, "m_before": 32, "index_name": "embedding", "drop_seconds": 0.13, "add_seconds": 0.69, "vector_index_m_after": 6, "vector_index_distance_after": "cosine", "chunk_index_length": 49152, "chunk_data_length": 16384, "vector_index_tablespace_bytes": 16777216, "rebuilt_at": "2026-09-17T12:01:17+00:00"} |

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
| ingested_at | 2026-09-16T23:30:33+00:00 |
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
| 1 | 61 | 75 | 0.813 |
| 3 | 67 | 75 | 0.893 |
| 5 | 68 | 75 | 0.907 |
| 10 | 68 | 75 | 0.907 |
| 20 | 68 | 75 | 0.907 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 32 | 66 | 0.485 |
| 3 | 40 | 66 | 0.606 |
| 5 | 43 | 66 | 0.652 |
| 10 | 49 | 66 | 0.742 |
| 20 | 56 | 66 | 0.848 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.497 |
| 3 | 66 | 0.621 |
| 5 | 66 | 0.683 |
| 10 | 66 | 0.763 |
| 20 | 66 | 0.854 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.56 | 1.23 | 0.64 | 0.45 | 2.06 |
| 3 | 375 | 0.50 | 1.12 | 0.61 | 0.43 | 2.00 |
| 5 | 375 | 0.52 | 1.17 | 0.62 | 0.43 | 2.15 |
| 10 | 375 | 0.57 | 1.31 | 0.69 | 0.48 | 2.35 |
| 20 | 375 | 0.72 | 1.69 | 0.83 | 0.59 | 2.70 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.57 | 4.76 | 4.61 | 4.44 | 6.90 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 2835 | REFUTES | Amino acid | not in top k | not in top k | Organic solutes (primarily glucose and amino acids)reabsorb 100% of filtrate via |
| 13383 | REFUTES | London | not in top k | not in top k | Jackie Tyler is introduced in "Rose", she is attacked by shop window dummies and |
| 15823 | REFUTES | Acadia University | not in top k | not in top k | Canadian poet and prose writer Peter Sanger studied at the University of Melbour |
| 19145 | REFUTES | British Columbia | not in top k | not in top k | Westshore velodrome a 33m outdoor bicycle racing track located in Colwood city ( |
| 74482 | REFUTES | Smithfield, Utah | not in top k | not in top k | Smithfield, Utah is the largest city in the area. |
| 79567 | REFUTES | Adelaide | not in top k | not in top k | Testeagles were formed as a techno rock trio in Adelaide (the capital city of th |
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 76756 | SUPPORTS | Asteraceae | 4 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 22563 | REFUTES | Asteraceae | 3 | 15 | Genus Hyoseris is classified under tribe Cichorieae,  a tribe in the plant famil |
| 85631 | REFUTES | Asteraceae | 3 | 3 | Plantae kingdom Diplazoptilon are classified in the Asteraceae family, a very la |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
