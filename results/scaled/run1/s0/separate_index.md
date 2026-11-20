# MariaDB against a separate in-process index: separate_index

Generated 2026-11-20T07:44:03+00:00 by `scripts/compare_separate_index.py`. 43134 chunks of 384 dimensions, 3000 queries (the corpus claims), k = 5, 1 timed passes after one untimed pass.

The copy: 66,253,824 bytes of vectors read out of MariaDB in 0.69 s, together with the columns the filters need (43,134 `chunk` rows, 3,851 `page` rows, 27,926 `page_category` rows, 11,413 `link (resolved)` rows). A separate index has to repeat this, or apply every change twice, to stay current.

The process's resident memory grew by 214,454,272 bytes during the export (from 1,414,533,120 to 1,628,987,392): the copy with the Python objects that hold its filter columns, and what the read left behind.

hnswlib 0.8.0: M = 16, ef_construction = 200 (the library's default), ef = 100; built in 1.45 s on all cores, 72,655,480 bytes when saved. Its parallel build makes a slightly different graph every run.

## No filter

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, bare k-NN statement (HNSW, M=16, ef_search 100) | approximate | 1.915 | 2.696 | 0.998 | 2970/3000 | 0/3000 | 0/3000 |
| MariaDB, joined statement (strategy inline) | exact | 16.166 | 17.713 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| NumPy scan of the copy, ids only | exact | 3.270 | 3.319 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| NumPy scan of the copy, then the hit rows from MariaDB by primary key | exact | 3.953 | 4.230 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| hnswlib graph over the copy (M=16, ef 100), ids only | approximate | 0.313 | 0.369 | 0.977 | 2710/3000 | 0/3000 | 0/3000 |

## Filter: section heading LIKE '%History%'

1,588 of 43,134 chunks pass.

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 20.459 | 24.784 | 1.000 | 2998/3000 | 0/3000 | 0/3000 |
| MariaDB, strategy overfetch 10 | approximate | 2.351 | 3.920 | 0.391 | 524/3000 | 2471/3000 | 951/3000 |
| NumPy pre-filter and scan of the copy, ids only | exact | 5.438 | 5.598 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 6.114 | 6.658 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 8.694 | 11.467 | 1.000 | 2993/3000 | 0/3000 | 0/3000 |

## Filter: the four filters: category 'Member states of the United Nations', at least 3,359 words, edited since 2020-12-18T02:38:12, linked from 'List of countries by maternal mortality ratio'

824 of 43,134 chunks pass.

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 23.206 | 38.497 | 1.000 | 2993/3000 | 0/3000 | 0/3000 |
| MariaDB, strategy overfetch 10 | approximate | 4.426 | 5.826 | 0.142 | 166/3000 | 2834/3000 | 2138/3000 |
| NumPy pre-filter and scan of the copy, ids only | exact | 0.622 | 0.677 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 1.179 | 1.432 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 8.306 | 14.964 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |

## How to read this

- Every method answers the same queries (the corpus claims, embedded once) with k hits, on one machine, from one client: one untimed pass over the queries, then `repeats` timed passes. Within a pass each query goes to every method of a table in turn, and the method that goes first rotates, so a drift of the machine reaches all of them alike. p50 and p95 use linear interpolation.
- MariaDB times are the wall-clock time of one search() call as the Python client sees it (the statement returns the hit rows with title, section path and text). mhnsw_ef_search is set once for the run, not per call. Because the methods take turns, every statement runs with colder caches than in the evaluation harness, which repeats one statement in a block, so its times are higher than the harness's for the same statement. The times also move between runs on this laptop (the repeat file shows by how much); read the ratios, not the last digit.
- NumPy and hnswlib each use one thread per query, as one MariaDB query does.
- The copy holds the vectors and the columns the filters read, not the text. 'ids only' rows return chunk ids; the rows 'then the hit rows from MariaDB' add the SELECT by primary key an application needs to show a result.
- 'exact' is NumPy over float64 copies of the vectors, cosine distance, ties by chunk_id. overlap = the share of the exact top k a method returned, averaged over the queries; short = queries that returned fewer ids than min(k, passing chunks); no row = those of them that returned nothing.
- The filters run on the copy in Python, inside the timed call. Before timing, the script checks that the copy and the SQL filter keep exactly the same chunks.
- This measures an index in the application's memory, not a vector database server: there is no network hop, no persistence and no concurrent writer. No dedicated vector store was measured.

Machine: 13th Gen Intel(R) Core(TM) i9-13900H, 20 logical cores, 14.7 GB; NumPy BLAS threads used: 1. MariaDB 11.8.9-MariaDB-ubu2404, index M = 16, mhnsw_ef_search = 100. Python 3.12.16, NumPy 2.5.3.
