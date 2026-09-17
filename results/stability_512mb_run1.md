# WikiLense evaluation: stability_512mb_run1

Generated 2026-09-17T11:59:31+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| experiment_group | stability |
| eval_seconds | 10.6 |
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
| 1 | 63 | 75 | 0.840 |
| 3 | 66 | 75 | 0.880 |
| 5 | 69 | 75 | 0.920 |
| 10 | 73 | 75 | 0.973 |
| 20 | 74 | 75 | 0.987 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 28 | 66 | 0.424 |
| 3 | 36 | 66 | 0.545 |
| 5 | 39 | 66 | 0.591 |
| 10 | 45 | 66 | 0.682 |
| 20 | 53 | 66 | 0.803 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.437 |
| 3 | 66 | 0.571 |
| 5 | 66 | 0.617 |
| 10 | 66 | 0.712 |
| 20 | 66 | 0.811 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.65 | 1.23 | 0.73 | 0.45 | 1.52 |
| 3 | 375 | 0.60 | 1.22 | 0.68 | 0.42 | 1.55 |
| 5 | 375 | 0.59 | 1.20 | 0.69 | 0.43 | 1.78 |
| 10 | 375 | 0.65 | 1.33 | 0.77 | 0.50 | 2.04 |
| 20 | 375 | 0.82 | 1.60 | 0.93 | 0.58 | 1.95 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.69 | 4.88 | 4.72 | 4.53 | 6.81 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 43675 | REFUTES | Los Angeles | 19 | not in top k | Annie Duke born September 13, 1965 lives in Los Angeles, California, U.S. (the s |
| 13383 | REFUTES | London | 7 | 7 | Jackie Tyler is introduced in "Rose", she is attacked by shop window dummies and |
| 45340 | REFUTES | Longitude | 7 | not in top k | Oklahoma State Highway 54 runs north to south and exists in two parts which lie |
| 76756 | SUPPORTS | Asteraceae | 6 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 95085 | SUPPORTS | Asteraceae | 6 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 85631 | REFUTES | Asteraceae | 5 | not in top k | Plantae kingdom Diplazoptilon are classified in the Asteraceae family, a very la |
| 22563 | REFUTES | Asteraceae | 4 | not in top k | Genus Hyoseris is classified under tribe Cichorieae,  a tribe in the plant famil |
| 82761 | REFUTES | Asteraceae | 4 | not in top k | The scientific classification of Prairie coneflowers is kingdom plantae, order a |
| 94092 | NOT ENOUGH INFO | Asteraceae | 3 | not in top k | Tournexia is from the Plantae kingdom and a member of the Asteraceae family. |

Claims whose LIMIT k hits were not the first k of LIMIT max_k, per k: {"10": 1}.

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
