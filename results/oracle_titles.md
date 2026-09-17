# WikiLense evaluation: oracle_titles

Generated 2026-09-16T23:55:39+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [1, 3, 5, 10, 20] |
| max_k | 20 |
| repeats | 5 |
| strategy | inline |
| overfetch | - |
| filters | - |
| claim_filters | oracle_title_filters |
| ef_search | - |
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
| 1 | 75 | 75 | 1.000 |
| 3 | 75 | 75 | 1.000 |
| 5 | 75 | 75 | 1.000 |
| 10 | 75 | 75 | 1.000 |
| 20 | 75 | 75 | 1.000 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 35 | 66 | 0.530 |
| 3 | 46 | 66 | 0.697 |
| 5 | 50 | 66 | 0.758 |
| 10 | 58 | 66 | 0.879 |
| 20 | 65 | 66 | 0.985 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.543 |
| 3 | 66 | 0.734 |
| 5 | 66 | 0.788 |
| 10 | 66 | 0.902 |
| 20 | 66 | 0.985 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 4.21 | 5.13 | 4.35 | 3.78 | 13.03 |
| 3 | 375 | 4.18 | 5.06 | 4.31 | 3.79 | 11.24 |
| 5 | 375 | 4.18 | 5.02 | 4.32 | 3.81 | 10.40 |
| 10 | 375 | 4.27 | 15.27 | 5.30 | 3.87 | 23.98 |
| 20 | 375 | 4.48 | 15.82 | 6.07 | 4.00 | 24.47 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.58 | 4.69 | 4.58 | 4.46 | 4.88 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 854 | REFUTES | Canton of Aargau | 1 | 16 | Ehrendingen is a municipality in the district of Baden in the canton of Aargau i |
| 2835 | REFUTES | Amino acid | 1 | 6 | Organic solutes (primarily glucose and amino acids)reabsorb 100% of filtrate via |
| 3482 | REFUTES | Jack Brabham | 1 | 7 | Jack Brabham, the winner of the 1966 United States Grand Prix at Watkins Glen, b |
| 4212 | REFUTES | Arsenal F.C. | 1 | 14 | In Chuks Aneke's career from 2011 to 2021, he was part of the Arsenal (football |
| 5121 | REFUTES | Association for Computing Machinery | 1 | 1 | Upsilon Pi Epsilon is a member of the ACHS - Association of College Honor Societ |
| 5753 | REFUTES | Asterales | 1 | 1 | Asterales is a species in the tribe Anthemideae, in the kingdom Plantae. |
| 5848 | SUPPORTS | Biennial plant | 1 | 1 | Some biennial plants have leaves that develop as basal rosettes. |
| 6419 | REFUTES | Kim Philby | 1 | 1 | Kim Philby, part of the Cambridge Five, was an effective American spy while he w |
| 7831 | NOT ENOUGH INFO | Juventus F.C. | 1 | 9 | Juventus, colloquially known as Juve (pronounced [ˈjuːve), is a professional foo |
| 7971 | REFUTES | Kabul | 1 | 3 | Urban decay is a process by which a city falls into a state of disrepair and neg |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
