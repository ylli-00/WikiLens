# WikiLense evaluation: final_oracle_titles

Generated 2026-09-22T01:42:27+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| n_chunks | 4598 |
| n_sentences | 33637 |
| experiment_group | final |
| configuration | {"chunk_max_words": 240, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": true, "label": "240 words, overlap 1, M=16, prefix on"} |
| eval_seconds | 9.3 |
| n_units_hatnote | 1316 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 16 |
| vector_index_distance | cosine |
| chunk_data_length | 14172160 |
| chunk_index_length | 6111232 |
| vector_index_tablespace_bytes | 13631488 |
| chunk_words_mean | 156.6 |
| chunk_words_median | 183.5 |
| chunk_words_p95 | 238.0 |
| chunk_words_max | 240 |
| ingest_run | {"label": "final", "kind": "ingest", "configuration": {"chunk_max_words": 240, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": true, "label": "240 words, overlap 1, M=16, prefix on"}, "ingested_at": "2026-09-22T01:41:38+00:00", "counts": {"n_pages": 100, "n_sections": 3173, "n_sentences": 33637, "n_units_empty": 372, "n_units_hatnote": 1316, "n_chunks": 4598, "n_links": 40542, "n_links_resolved": 373, "n_links_skipped": 0, "n_claims": 75, "n_evidence": 114, "n_evidence_page_resolved": 114, "n_evidence_sentence_resolved": 87}, "seconds": {"schema": 0.1, "parse": 0.84, "embed": 16.55, "load": 3.58, "resolve": 0.0, "analyze": 0.01, "total": 21.14}, "n_units_hatnote": 1316, "chunk_words_mean": 156.6, "chunk_words_median": 183.5, "chunk_words_p95": 238.0, "chunk_words_max": 240, "vector_index_name": "embedding", "vector_index_m": 16, "vector_index_distance": "cosine", "chunk_data_length": 14172160, "chunk_index_length": 344064, "vector_index_tablespace_bytes": 13631488, "index_rebuild": null} |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| analyze_seconds | 0.008 |
| analyze_tables | chunk,page,section,link |
| chunk_max_words | 240 |
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
| ingested_at | 2026-09-22T01:41:38+00:00 |
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
| 1 | 75 | 75 | 1.000 |
| 3 | 75 | 75 | 1.000 |
| 5 | 75 | 75 | 1.000 |
| 10 | 75 | 75 | 1.000 |
| 20 | 75 | 75 | 1.000 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 42 | 66 | 0.636 |
| 3 | 55 | 66 | 0.833 |
| 5 | 59 | 66 | 0.894 |
| 10 | 64 | 66 | 0.970 |
| 20 | 66 | 66 | 1.000 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.649 |
| 3 | 66 | 0.852 |
| 5 | 66 | 0.913 |
| 10 | 66 | 0.977 |
| 20 | 66 | 1.000 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 2.55 | 3.04 | 2.64 | 2.24 | 6.76 |
| 3 | 375 | 2.56 | 2.88 | 2.63 | 2.25 | 5.48 |
| 5 | 375 | 2.58 | 4.23 | 2.93 | 2.26 | 10.68 |
| 10 | 375 | 2.65 | 9.87 | 3.70 | 2.34 | 15.69 |
| 20 | 375 | 2.80 | 9.96 | 4.64 | 2.50 | 16.32 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.66 | 4.76 | 4.67 | 4.56 | 4.91 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 854 | REFUTES | Canton of Aargau | 1 | 9 | Ehrendingen is a municipality in the district of Baden in the canton of Aargau i |
| 2835 | REFUTES | Amino acid | 1 | 1 | Organic solutes (primarily glucose and amino acids)reabsorb 100% of filtrate via |
| 3482 | REFUTES | Jack Brabham | 1 | 1 | Jack Brabham, the winner of the 1966 United States Grand Prix at Watkins Glen, b |
| 4212 | REFUTES | Arsenal F.C. | 1 | 5 | In Chuks Aneke's career from 2011 to 2021, he was part of the Arsenal (football |
| 5121 | REFUTES | Association for Computing Machinery | 1 | 1 | Upsilon Pi Epsilon is a member of the ACHS - Association of College Honor Societ |
| 5753 | REFUTES | Asterales | 1 | 1 | Asterales is a species in the tribe Anthemideae, in the kingdom Plantae. |
| 5848 | SUPPORTS | Biennial plant | 1 | 1 | Some biennial plants have leaves that develop as basal rosettes. |
| 6419 | REFUTES | Kim Philby | 1 | 1 | Kim Philby, part of the Cambridge Five, was an effective American spy while he w |
| 7831 | NOT ENOUGH INFO | Juventus F.C. | 1 | 1 | Juventus, colloquially known as Juve (pronounced [ˈjuːve), is a professional foo |
| 7971 | REFUTES | Kabul | 1 | 9 | Urban decay is a process by which a city falls into a state of disrepair and neg |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
