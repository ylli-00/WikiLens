# WikiLense evaluation: baseline

Generated 2026-09-16T23:55:22+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| 1 | 56 | 75 | 0.747 |
| 3 | 62 | 75 | 0.827 |
| 5 | 64 | 75 | 0.853 |
| 10 | 68 | 75 | 0.907 |
| 20 | 68 | 75 | 0.907 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 22 | 66 | 0.333 |
| 3 | 31 | 66 | 0.470 |
| 5 | 33 | 66 | 0.500 |
| 10 | 39 | 66 | 0.591 |
| 20 | 48 | 66 | 0.727 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.346 |
| 3 | 66 | 0.495 |
| 5 | 66 | 0.527 |
| 10 | 66 | 0.621 |
| 20 | 66 | 0.735 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.61 | 0.85 | 0.62 | 0.46 | 1.31 |
| 3 | 375 | 0.60 | 0.83 | 0.59 | 0.42 | 1.31 |
| 5 | 375 | 0.62 | 0.83 | 0.61 | 0.43 | 1.33 |
| 10 | 375 | 0.67 | 0.96 | 0.67 | 0.49 | 1.40 |
| 20 | 375 | 0.80 | 1.12 | 0.80 | 0.58 | 1.53 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.57 | 4.68 | 4.58 | 4.47 | 4.79 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 7971 | REFUTES | Kabul | not in top k | not in top k | Urban decay is a process by which a city falls into a state of disrepair and neg |
| 15823 | REFUTES | Acadia University | not in top k | not in top k | Canadian poet and prose writer Peter Sanger studied at the University of Melbour |
| 31383 | REFUTES | Louis Pasteur | not in top k | not in top k | Despite cold War tensions Louis Pasteur, a Soviet microbiologist and virologist, |
| 32645 | REFUTES | AFC Ajax | not in top k | not in top k | The 72-titled Netherlands football team, AFC Ajax, has played in the Eredivisie |
| 34024 | REFUTES | Anime | not in top k | not in top k | Panda and the Magic Serpent, also known as The Tale of the White Serpent, is the |
| 43675 | REFUTES | Los Angeles | not in top k | not in top k | Annie Duke born September 13, 1965 lives in Los Angeles, California, U.S. (the s |
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 50355 | REFUTES | Los Angeles | 10 | not in top k | In 2011 Liz Nistico and Louie Diller of Holychild from Los Angeles, officially t |
| 13383 | REFUTES | London | 7 | 7 | Jackie Tyler is introduced in "Rose", she is attacked by shop window dummies and |
| 45340 | REFUTES | Longitude | 7 | not in top k | Oklahoma State Highway 54 runs north to south and exists in two parts which lie |

Claims whose LIMIT k hits were not the first k of LIMIT max_k, per k: {"10": 1}.

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
