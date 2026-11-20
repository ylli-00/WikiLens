# MariaDB against a separate in-process index: separate_index

Generated 2026-11-19T23:39:37+00:00 by `scripts/compare_separate_index.py`. 132359 chunks of 384 dimensions, 3000 queries (the corpus claims), k = 5, 1 timed passes after one untimed pass.

The copy: 203,303,424 bytes of vectors read out of MariaDB in 2.42 s, together with the columns the filters need (132,359 `chunk` rows, 7,686 `page` rows, 58,569 `page_category` rows, 154,762 `link (resolved)` rows). A separate index has to repeat this, or apply every change twice, to stay current.

The process's resident memory grew by 701,804,544 bytes during the export (from 1,414,332,416 to 2,116,136,960): the copy with the Python objects that hold its filter columns, and what the read left behind.

hnswlib 0.8.0: M = 16, ef_construction = 200 (the library's default), ef = 100; built in 5.87 s on all cores, 222,955,496 bytes when saved. Its parallel build makes a slightly different graph every run.

## No filter

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, bare k-NN statement (HNSW, M=16, ef_search 100) | approximate | 2.869 | 4.835 | 0.993 | 2917/3000 | 0/3000 | 0/3000 |
| MariaDB, joined statement (strategy inline) | exact | 27.301 | 32.018 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| NumPy scan of the copy, ids only | exact | 10.193 | 10.529 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| NumPy scan of the copy, then the hit rows from MariaDB by primary key | exact | 11.057 | 13.045 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| hnswlib graph over the copy (M=16, ef 100), ids only | approximate | 0.356 | 0.424 | 0.961 | 2544/3000 | 0/3000 | 0/3000 |

## Filter: section heading LIKE '%History%'

5,363 of 132,359 chunks pass.

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 37.362 | 56.433 | 1.000 | 2996/3000 | 0/3000 | 0/3000 |
| MariaDB, strategy overfetch 10 | approximate | 3.267 | 8.717 | 0.405 | 568/3000 | 2424/3000 | 904/3000 |
| NumPy pre-filter and scan of the copy, ids only | exact | 16.014 | 16.849 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 16.941 | 18.908 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 19.007 | 23.592 | 0.999 | 2981/3000 | 0/3000 | 0/3000 |

## Filter: the four filters: category 'Member states of the United Nations', at least 5,214 words, edited since 2020-12-21T15:34:08, linked from 'Legality of cannabis'

3,107 of 132,359 chunks pass.

| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |
|---|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 58.689 | 189.014 | 0.998 | 2962/3000 | 0/3000 | 0/3000 |
| MariaDB, strategy overfetch 10 | approximate | 3.935 | 8.649 | 0.151 | 198/3000 | 2801/3000 | 2126/3000 |
| NumPy pre-filter and scan of the copy, ids only | exact | 1.686 | 2.001 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 2.438 | 3.384 | 1.000 | 3000/3000 | 0/3000 | 0/3000 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 11.731 | 27.742 | 0.999 | 2990/3000 | 0/3000 | 0/3000 |

## How to read this

- Every method answers the same queries (the corpus claims, embedded once) with k hits, on one machine, from one client: one untimed pass over the queries, then `repeats` timed passes. Within a pass each query goes to every method of a table in turn, and the method that goes first rotates, so a drift of the machine reaches all of them alike. p50 and p95 use linear interpolation.
- MariaDB times are the wall-clock time of one search() call as the Python client sees it (the statement returns the hit rows with title, section path and text). mhnsw_ef_search is set once for the run, not per call. Because the methods take turns, every statement runs with colder caches than in the evaluation harness, which repeats one statement in a block, so its times are higher than the harness's for the same statement. The times also move between runs on this laptop (the repeat file shows by how much); read the ratios, not the last digit.
- NumPy and hnswlib each use one thread per query, as one MariaDB query does.
- The copy holds the vectors and the columns the filters read, not the text. 'ids only' rows return chunk ids; the rows 'then the hit rows from MariaDB' add the SELECT by primary key an application needs to show a result.
- 'exact' is NumPy over float64 copies of the vectors, cosine distance, ties by chunk_id. overlap = the share of the exact top k a method returned, averaged over the queries; short = queries that returned fewer ids than min(k, passing chunks); no row = those of them that returned nothing.
- The filters run on the copy in Python, inside the timed call. Before timing, the script checks that the copy and the SQL filter keep exactly the same chunks.
- This measures an index in the application's memory, not a vector database server: there is no network hop, no persistence and no concurrent writer. No dedicated vector store was measured.

Machine: 13th Gen Intel(R) Core(TM) i9-13900H, 20 logical cores, 14.7 GB; NumPy BLAS threads used: 1. MariaDB 11.8.9-MariaDB-ubu2404, index M = 16, mhnsw_ef_search = 100. Python 3.12.16, NumPy 2.5.3.
