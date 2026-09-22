# WikiLense evaluation: noprefix_inline_ef_20

Generated 2026-09-17T12:18:06+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [1, 3, 5, 10, 20] |
| max_k | 20 |
| repeats | 2 |
| strategy | inline |
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
| ingested_at | 2026-09-17T12:17:52+00:00 |
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
| 1 | 65 | 75 | 0.867 |
| 3 | 72 | 75 | 0.960 |
| 5 | 74 | 75 | 0.987 |
| 10 | 74 | 75 | 0.987 |
| 20 | 74 | 75 | 0.987 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 35 | 66 | 0.530 |
| 3 | 43 | 66 | 0.652 |
| 5 | 48 | 66 | 0.727 |
| 10 | 55 | 66 | 0.833 |
| 20 | 59 | 66 | 0.894 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.543 |
| 3 | 66 | 0.676 |
| 5 | 66 | 0.759 |
| 10 | 66 | 0.864 |
| 20 | 66 | 0.917 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 150 | 4.38 | 5.98 | 4.67 | 3.97 | 10.94 |
| 3 | 150 | 4.35 | 6.51 | 4.66 | 3.87 | 9.28 |
| 5 | 150 | 4.42 | 6.51 | 4.69 | 3.89 | 9.87 |
| 10 | 150 | 4.51 | 7.02 | 4.78 | 3.98 | 13.03 |
| 20 | 150 | 4.56 | 6.17 | 4.86 | 4.14 | 12.96 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 150 | 4.77 | 5.08 | 4.81 | 4.62 | 5.84 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 13383 | REFUTES | London | 4 | 12 | Jackie Tyler is introduced in "Rose", she is attacked by shop window dummies and |
| 95085 | SUPPORTS | Asteraceae | 4 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 9154 | REFUTES | Asia | 2 | 4 | A total of 22 foreign NBA players came from Asia, coming from seven of the 48 As |
| 11235 | REFUTES | London Underground | 2 | 14 | London's Underground is one of the few Railway electrification systems that util |
| 14253 | REFUTES | Asterales | 2 | 5 | Russowia belongs to the Asteraceae family of the Asterales order, an order of mo |
| 22563 | REFUTES | Asteraceae | 2 | 11 | Genus Hyoseris is classified under tribe Cichorieae,  a tribe in the plant famil |
| 40707 | NOT ENOUGH INFO | American Civil War | 2 | 10 | Following the American Civil War, the Army and Navy were unsupportive of each ot |
| 76756 | SUPPORTS | Asteraceae | 2 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 85631 | REFUTES | Asteraceae | 2 | 2 | Plantae kingdom Diplazoptilon are classified in the Asteraceae family, a very la |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
