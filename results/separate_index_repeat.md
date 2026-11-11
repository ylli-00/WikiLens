# MariaDB against a separate in-process index: separate_index_repeat

Generated 2026-11-11T03:06:38+00:00 by `scripts/compare_separate_index.py`. 4598 chunks of 384 dimensions, 75 queries (the corpus claims), k = 10, 5 timed passes after one untimed pass.

The copy: 7,062,528 bytes of vectors read out of MariaDB in 0.03 s, together with the columns the filters need (4,598 `chunk` rows, 100 `page` rows, 1,123 `page_category` rows, 373 `link (resolved)` rows). A separate index has to repeat this, or apply every change twice, to stay current.

hnswlib 0.8.0: M = 16, ef_construction = 200 (the library's default), ef = 100; built in 0.06 s on all cores, 7,746,632 bytes when saved. Its parallel build makes a slightly different graph every run.

## No filter

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, bare k-NN statement (HNSW, M=16, ef_search 100) | approximate | 0.708 | 1.106 | 1.000 | 75/75 | 0/75 | 0/75 |
| MariaDB, joined statement (strategy inline) | exact | 2.247 | 2.955 | 1.000 | 75/75 | 0/75 | 0/75 |
| NumPy scan of the copy, ids only | exact | 0.204 | 0.338 | 1.000 | 75/75 | 0/75 | 0/75 |
| NumPy scan of the copy, then the hit rows from MariaDB by primary key | exact | 0.502 | 0.838 | 1.000 | 75/75 | 0/75 | 0/75 |
| hnswlib graph over the copy (M=16, ef 100), ids only | approximate | 0.144 | 0.189 | 1.000 | 75/75 | 0/75 | 0/75 |

## Filter: section heading LIKE '%History%'

117 of 4,598 chunks pass.

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 3.562 | 5.454 | 1.000 | 75/75 | 0/75 | 0/75 |
| MariaDB, strategy overfetch 10 | approximate | 1.200 | 1.815 | 0.297 | 6/75 | 69/75 | 18/75 |
| NumPy pre-filter and scan of the copy, ids only | exact | 0.493 | 0.688 | 1.000 | 75/75 | 0/75 | 0/75 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 1.044 | 1.652 | 1.000 | 75/75 | 0/75 | 0/75 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 1.949 | 2.825 | 1.000 | 75/75 | 0/75 | 0/75 |

## Filter: the brief's four filters: category 'Member states of the United Nations', at least 1,000 words, edited since 2020-12-01, linked from 'Asia'

377 of 4,598 chunks pass.

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 3.454 | 4.954 | 1.000 | 75/75 | 0/75 | 0/75 |
| MariaDB, strategy overfetch 10 | approximate | 1.513 | 1.946 | 0.423 | 16/75 | 59/75 | 20/75 |
| NumPy pre-filter and scan of the copy, ids only | exact | 0.176 | 0.238 | 1.000 | 75/75 | 0/75 | 0/75 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 0.730 | 1.004 | 1.000 | 75/75 | 0/75 | 0/75 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 0.988 | 1.402 | 1.000 | 75/75 | 0/75 | 0/75 |

## How to read this

- Every method answers the same queries (the corpus claims, embedded once) with k hits, on one machine, from one client: one untimed pass over the queries, then `repeats` timed passes. Within a pass each query goes to every method of a table in turn, and the method that goes first rotates, so a drift of the machine reaches all of them alike. p50 and p95 use linear interpolation.
- MariaDB times are the wall-clock time of one search() call as the Python client sees it (the statement returns the hit rows with title, section path and text). mhnsw_ef_search is set once for the run, not per call. Because the methods take turns, every statement runs with colder caches than in the evaluation harness, which repeats one statement in a block (results/final_ef_100.md has 0.51 ms p50 for the bare k-NN statement). The times also move between runs on this laptop (the repeat file shows by how much); read the ratios, not the last digit.
- NumPy and hnswlib each use one thread per query, as one MariaDB query does.
- The copy holds the vectors and the columns the filters read, not the text. 'ids only' rows return chunk ids; the rows 'then the hit rows from MariaDB' add the SELECT by primary key an application needs to show a result.
- 'exact' is NumPy over float64 copies of the vectors, cosine distance, ties by chunk_id. overlap = the share of the exact top k a method returned, averaged over the queries; short = queries that returned fewer ids than min(k, passing chunks); no row = those of them that returned nothing.
- The filters run on the copy in Python, inside the timed call. Before timing, the script checks that the copy and the SQL filter keep exactly the same chunks.
- This measures an index in the application's memory, not a vector database server: there is no network hop, no persistence and no concurrent writer. No dedicated vector store was measured.

Machine: 13th Gen Intel(R) Core(TM) i9-13900H, 20 logical cores, 14.7 GB; NumPy BLAS threads used: 1. MariaDB 11.8.9-MariaDB-ubu2404, index M = 16, mhnsw_ef_search = 100. Python 3.12.16, NumPy 2.5.3.
