# WikiLense evaluation: chunk240_ef_100

Generated 2026-09-22T01:38:48+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

## Parameters

| Parameter | Value |
|---|---|
| ks | [1, 2, 3, 5, 10, 20] |
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
| n_chunks | 4598 |
| n_sentences | 33637 |
| experiment_group | chunk_size |
| configuration | {"chunk_max_words": 240, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": true, "label": "240 words, overlap 1, M=16, prefix on"} |
| eval_seconds | 4.0 |
| n_units_hatnote | 1316 |
| mhnsw_ef_search_global | 20 |
| vector_index_name | embedding |
| vector_index_m | 16 |
| vector_index_distance | cosine |
| chunk_data_length | 14172160 |
| chunk_index_length | 344064 |
| vector_index_tablespace_bytes | 13631488 |
| chunk_words_mean | 156.6 |
| chunk_words_median | 183.5 |
| chunk_words_p95 | 238.0 |
| chunk_words_max | 240 |
| ingest_run | {"label": "chunk240", "kind": "ingest", "configuration": {"chunk_max_words": 240, "chunk_overlap_units": 1, "index_m": 16, "use_prefix": true, "label": "240 words, overlap 1, M=16, prefix on"}, "ingested_at": "2026-09-22T01:38:44+00:00", "counts": {"n_pages": 100, "n_sections": 3173, "n_sentences": 33637, "n_units_empty": 372, "n_units_hatnote": 1316, "n_chunks": 4598, "n_links": 40542, "n_links_resolved": 373, "n_links_skipped": 0, "n_claims": 75, "n_evidence": 114, "n_evidence_page_resolved": 114, "n_evidence_sentence_resolved": 87}, "seconds": {"schema": 0.1, "parse": 0.84, "embed": 16.65, "load": 3.65, "resolve": 0.0, "analyze": 0.01, "total": 21.3}, "n_units_hatnote": 1316, "chunk_words_mean": 156.6, "chunk_words_median": 183.5, "chunk_words_p95": 238.0, "chunk_words_max": 240, "vector_index_name": "embedding", "vector_index_m": 16, "vector_index_distance": "cosine", "chunk_data_length": 14172160, "chunk_index_length": 344064, "vector_index_tablespace_bytes": 13631488, "index_rebuild": null} |

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
| ingested_at | 2026-09-22T01:38:44+00:00 |
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
| 1 | 63 | 75 | 0.840 |
| 2 | 70 | 75 | 0.933 |
| 3 | 74 | 75 | 0.987 |
| 5 | 74 | 75 | 0.987 |
| 10 | 74 | 75 | 0.987 |
| 20 | 74 | 75 | 0.987 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 39 | 66 | 0.591 |
| 2 | 49 | 66 | 0.742 |
| 3 | 52 | 66 | 0.788 |
| 5 | 55 | 66 | 0.833 |
| 10 | 60 | 66 | 0.909 |
| 20 | 65 | 66 | 0.985 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.596 |
| 2 | 66 | 0.763 |
| 3 | 66 | 0.808 |
| 5 | 66 | 0.852 |
| 10 | 66 | 0.917 |
| 20 | 66 | 0.985 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.72 | 0.90 | 0.73 | 0.53 | 1.47 |
| 2 | 375 | 0.69 | 0.86 | 0.69 | 0.48 | 1.45 |
| 3 | 375 | 0.70 | 0.86 | 0.70 | 0.47 | 1.53 |
| 5 | 375 | 0.73 | 0.91 | 0.72 | 0.50 | 1.81 |
| 10 | 375 | 0.81 | 0.97 | 0.81 | 0.57 | 1.80 |
| 20 | 375 | 0.94 | 1.10 | 0.94 | 0.71 | 1.60 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 4.79 | 4.90 | 4.79 | 4.60 | 5.21 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 87976 | NOT ENOUGH INFO | Lincoln, England | not in top k | not in top k | During the Waddington By-Election 17 October 2002, Conservatives won more votes |
| 7971 | REFUTES | Kabul | 3 | 19 | Urban decay is a process by which a city falls into a state of disrepair and neg |
| 76756 | SUPPORTS | Asteraceae | 3 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 85631 | REFUTES | Asteraceae | 3 | 3 | Plantae kingdom Diplazoptilon are classified in the Asteraceae family, a very la |
| 95085 | SUPPORTS | Asteraceae | 3 | n/a | The scientific classification of kingdom plantae, order asterales and family ast |
| 9154 | REFUTES | Asia | 2 | 20 | A total of 22 foreign NBA players came from Asia, coming from seven of the 48 As |
| 11235 | REFUTES | London Underground | 2 | 10 | London's Underground is one of the few Railway electrification systems that util |
| 11439 | REFUTES | Juventus F.C. | 2 | 2 | Roma, colloquially known as Juve (pronounced [ˈjuːve), is a professional footbal |
| 14253 | REFUTES | Asterales | 2 | 3 | Russowia belongs to the Asteraceae family of the Asterales order, an order of mo |
| 22563 | REFUTES | Asteraceae | 2 | 6 | Genus Hyoseris is classified under tribe Cichorieae,  a tribe in the plant famil |

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the first k hits of the LIMIT max_k search of each claim, in the order search() returned them.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
