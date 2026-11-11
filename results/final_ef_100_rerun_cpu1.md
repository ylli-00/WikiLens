# WikiLense evaluation: final_ef_100_rerun_cpu1

Generated 2026-11-11T03:08:06+00:00. 75 claims; 66 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| ef_search | 100 |
| ef_search_source | settings |
| ef_search_effective | 100 |
| mhnsw_max_cache_size | 536870912 |
| index_m | 16 |
| index_distance | cosine |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_device | cpu |
| n_claims | 75 |
| n_evidence_claims | 66 |
| n_pages | 100 |
| n_chunks | 4598 |
| n_sentences | 33637 |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| analyze_seconds | 0.010 |
| analyze_tables | chunk,page,section,link,category,page_category |
| chunk_max_words | 240 |
| chunk_overlap_units | 1 |
| corpus_claims_sha256 | a9a36c5c14f9a3baea97df6b42361f866249a4c636e4504cd259d7e54725b8ac |
| corpus_dir | data/corpus |
| corpus_page_meta_sha256 | 9a892f80de5b63818fe79061445e224e3ec142554abab183531529da630f96ff |
| corpus_pages_sha256 | edf0955892c37fe54bb840cda33a6df5a06c44394289e7176a922556d09621dc |
| embedding_dim | 384 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_prefix | true |
| embedding_revision | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a |
| hatnote_pattern | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\\b\|instance\\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) |
| index_distance | cosine |
| index_m | 16 |
| ingested_at | 2026-10-21T23:46:15+00:00 |
| mariadb_version | 11.8.9-MariaDB-ubu2404 |
| n_units_hatnote | 1316 |
| wikilense_version | 0.1.0 |

## Machine and versions

| Item | Value |
|---|---|
| platform | Linux-7.0.0-43-generic-x86_64-with-glibc2.43 |
| architecture | x86_64 |
| cpu | 13th Gen Intel(R) Core(TM) i9-13900H |
| cpu_count | 20 |
| memory_gb | 14.7 |
| embedding_device | cpu |
| gpu | - |
| python | 3.12.16 |
| wikilense | 0.1.0 |
| mariadb | 11.8.9-MariaDB-ubu2404 |
| PyMySQL | 1.2.3 |
| numpy | 2.5.3 |
| torch | 2.14.0 |
| sentence-transformers | 6.0.1 |

## Article recall@k

A claim counts when a gold page is among the pages of its first k chunks.

| k | claims recalled | of | article recall |
|---|---|---|---|
| 1 | 63 | 75 | 0.840 |
| 3 | 74 | 75 | 0.987 |
| 5 | 74 | 75 | 0.987 |
| 10 | 74 | 75 | 0.987 |
| 20 | 74 | 75 | 0.987 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 39 | 66 | 0.591 |
| 3 | 52 | 66 | 0.788 |
| 5 | 55 | 66 | 0.833 |
| 10 | 60 | 66 | 0.909 |
| 20 | 65 | 66 | 0.985 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 66 | 0.596 |
| 3 | 66 | 0.808 |
| 5 | 66 | 0.852 |
| 10 | 66 | 0.917 |
| 20 | 66 | 0.985 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 375 | 0.50 | 1.08 | 0.57 | 0.29 | 1.77 |
| 3 | 375 | 0.46 | 1.01 | 0.54 | 0.27 | 1.62 |
| 5 | 375 | 0.47 | 1.05 | 0.54 | 0.28 | 1.46 |
| 10 | 375 | 0.53 | 1.12 | 0.61 | 0.33 | 1.67 |
| 20 | 375 | 0.63 | 1.33 | 0.73 | 0.42 | 1.95 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 375 | 17.83 | 22.68 | 18.59 | 14.62 | 26.73 |

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
- Ranking: the metrics at k use the hits of the LIMIT k search of each claim (first timed pass), in the order search() returned them; the per-claim ranks use the LIMIT max_k search.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
