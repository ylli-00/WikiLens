# WikiLense evaluation: s1_check

Generated 2026-11-20T12:46:46+00:00. 3000 claims; 3000 with a sentence-only evidence set (the denominator of evidence recall and unit coverage).

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
| ef_search_source | argument |
| ef_search_effective | 100 |
| mhnsw_max_cache_size | 536870912 |
| index_m | 16 |
| index_distance | cosine |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_device | cuda |
| n_claims | 3000 |
| n_evidence_claims | 3000 |
| n_pages | 7686 |
| n_chunks | 132359 |
| n_sentences | 906342 |

## Ingest parameters (ingest_meta)

| Key | Value |
|---|---|
| analyze_seconds | 0.089 |
| analyze_tables | chunk,page,section,link,category,page_category |
| chunk_max_words | 240 |
| chunk_overlap_units | 1 |
| corpus_claims_sha256 | d29154f4cfe95ab5c2330f20ea05e2836fd2f23a9f4894b25fcc2c89e946297c |
| corpus_dir | data/scaled/wikilense_s1/S1 |
| corpus_page_meta_sha256 | none |
| corpus_pages_sha256 | d98f74eb80ef76e79d12ba932943407bcc2d75db658b2ae150ad275e0e53ff3a |
| embedding_dim | 384 |
| embedding_model | BAAI/bge-small-en-v1.5 |
| embedding_prefix | true |
| embedding_revision | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a |
| hatnote_pattern | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\\b\|instance\\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) |
| index_distance | cosine |
| index_m | 16 |
| ingested_at | 2026-11-20T12:42:29+00:00 |
| mariadb_version | 11.8.9-MariaDB-ubu2404 |
| n_units_hatnote | 25471 |
| wikilense_version | 0.1.0 |

## Machine and versions

| Item | Value |
|---|---|
| platform | Linux-7.0.0-43-generic-x86_64-with-glibc2.43 |
| architecture | x86_64 |
| cpu | 13th Gen Intel(R) Core(TM) i9-13900H |
| cpu_count | 20 |
| memory_gb | 14.7 |
| embedding_device | cuda |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU |
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
| 1 | 2814 | 3000 | 0.938 |
| 3 | 2911 | 3000 | 0.970 |
| 5 | 2934 | 3000 | 0.978 |
| 10 | 2948 | 3000 | 0.983 |
| 20 | 2957 | 3000 | 0.986 |

## Evidence recall@k

A claim counts when every unit of one of its sentence-only evidence sets is inside its first k chunks.

| k | claims recalled | of | evidence recall |
|---|---|---|---|
| 1 | 1304 | 3000 | 0.435 |
| 3 | 2042 | 3000 | 0.681 |
| 5 | 2256 | 3000 | 0.752 |
| 10 | 2478 | 3000 | 0.826 |
| 20 | 2607 | 3000 | 0.869 |

## Unit coverage@k

Mean, over the same claims, of the best share of a set's gold units inside the first k chunks.

| k | of | unit coverage |
|---|---|---|
| 1 | 3000 | 0.614 |
| 3 | 3000 | 0.815 |
| 5 | 3000 | 0.863 |
| 10 | 3000 | 0.910 |
| 20 | 3000 | 0.934 |

## SQL latency per k (ms)

One search() call with LIMIT k, measured by the client; warm.

| k | n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| 1 | 15000 | 1.70 | 2.60 | 1.75 | 0.69 | 5.33 |
| 3 | 15000 | 1.37 | 2.27 | 1.45 | 0.57 | 144.75 |
| 5 | 15000 | 1.30 | 2.00 | 1.34 | 0.55 | 5.76 |
| 10 | 15000 | 1.53 | 2.36 | 1.59 | 0.59 | 5.24 |
| 20 | 15000 | 2.04 | 3.03 | 2.08 | 0.76 | 6.37 |

## Query embedding latency (ms)

One embed_queries([text]) call per claim and repeat; warm.

| n | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| 15000 | 6.06 | 6.24 | 6.07 | 5.75 | 49.62 |

## Claims with the worst rank of their gold page

Rank within the first 20 chunks; "not in top k" means no chunk of a gold page was retrieved.

| claim | label | gold page | gold page rank | evidence rank | claim text |
|---|---|---|---|---|---|
| 893 | NOT ENOUGH INFO | Kelly Holcomb | not in top k | not in top k | In 2005, the running game was the main highlight of an otherwise stagnant team l |
| 1288 | REFUTES | Musical tone, Triangle wave | not in top k | not in top k | DC is one of the two types of waveforms that has the lowest crest factor, the ot |
| 1601 | NOT ENOUGH INFO | Site | not in top k | not in top k | The Baseline for SITE Impact Survey (Adults) commenced in 1975. |
| 2900 | NOT ENOUGH INFO | Thomas Wodehouse | not in top k | not in top k | Thomas Woodhouse and Blanche got married on June 16, 1605 and lived together as |
| 3274 | NOT ENOUGH INFO | Suszki | not in top k | not in top k | Dürrkunzendorf united with the County of Glatz in 1763 before the reorganization |
| 5253 | SUPPORTS | Highbury Korfball Club | not in top k | not in top k | Andrew Speck served as a head coach from 2012-2015, making him the longest head |
| 5589 | SUPPORTS | Ditrigona obliquilinea | not in top k | not in top k | The wingspan of the Corycia pnocaria is thirteen to twenty millimeters for males |
| 6434 | NOT ENOUGH INFO | List of Buffalo Bills head coaches | not in top k | not in top k | In 2005, the running game was led by journeyman Kelly Holcomb and in 2006, Mular |
| 7492 | NOT ENOUGH INFO | Vector | not in top k | not in top k | The weather on day 0 (today) is known to be sunny, which is represented by a vec |
| 7733 | NOT ENOUGH INFO | Dwight Henry | not in top k | not in top k | Democrat Newd McWherter beat Republican Dwight Henry, an unknown one term member |

Claims whose LIMIT k hits were not the first k of LIMIT max_k, per k: {"10": 2}.

## Notes

- article_recall = article_hits / n_claims; evidence_recall = evidence_hits / n_evidence_claims (claims with a sentence-only evidence set whose units all resolved); unit_coverage = mean over the same claims of the best share of a set's units covered.
- Ranking: the metrics at k use the hits of the LIMIT k search of each claim (first timed pass), in the order search() returned them; the per-claim ranks use the LIMIT max_k search.
- sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and p95 use linear interpolation.
- embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per claim and repeat after a warm-up call; the searches use vectors from one batched call.
