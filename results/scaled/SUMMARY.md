# Scaled run: every table

Written 2026-11-20T07:50:30+00:00 by `scripts/scaled_summary.py` from the files in this folder (no database). The 3,000 claims of `data/scaled/claims_3000.jsonl` on two databases: S0 = the gold pages, S1 = S0 plus one distractor page per gold article (`docs/DATA.md`). Settings: 240-word chunks, overlap 1, bge-small-en-v1.5 with the title prefix, M = 16, cosine, `mhnsw_ef_search` 100. Every run on this laptop with the CPU capped at 50% and the GPU at 2,100 MHz, so compare times only within this folder, not with `results/` at the top level.

## 1. Recall and latency of the HNSW search (`wikilense eval`)

All 3,000 claims, ef_search 100, 5 timed passes per claim after one warm-up. S0: runs 1, 2; cells give run 1 / run 2, recall identical in every run; S1: runs 1, 2; cells give run 1 / run 2, recall identical in every run.

| k | S0 article recall | S0 evidence recall | S0 SQL p50 | S0 SQL p95 | S1 article recall | S1 evidence recall | S1 SQL p50 | S1 SQL p95 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.965 (2,895) | 0.446 (1,337) | 1.28 / 1.27 ms | 1.74 / 1.73 ms | 0.938 (2,814) | 0.434 (1,303) | 1.86 / 1.86 ms | 2.96 / 2.98 ms |
| 3 | 0.979 (2,937) | 0.703 (2,108) | 1.12 / 1.13 ms | 1.57 / 1.56 ms | 0.970 (2,911) | 0.681 (2,042) | 1.52 / 1.52 ms | 2.55 / 2.59 ms |
| 5 | 0.984 (2,951) | 0.780 (2,341) | 1.11 / 1.11 ms | 1.52 / 1.52 ms | 0.978 (2,933) | 0.751 (2,253) | 1.43 / 1.42 ms | 2.32 / 2.30 ms |
| 10 | 0.986 (2,959) | 0.853 (2,558) | 1.21 / 1.22 ms | 1.71 / 1.70 ms | 0.982 (2,947) | 0.826 (2,479) | 1.66 / 1.65 ms | 2.69 / 2.68 ms |
| 20 | 0.990 (2,971) | 0.893 (2,678) | 1.41 / 1.41 ms | 2.06 / 2.06 ms | 0.986 (2,957) | 0.870 (2,609) | 2.17 / 2.17 ms | 3.36 / 3.36 ms |

S0: the SQL p50 and p95 differ between runs by at most 0.9% (largest over every k).

S1: the SQL p50 and p95 differ between runs by at most 1.3% (largest over every k).

Query embedding p50: S0 6.09 / 6.15 ms; S1 6.07 / 6.07 ms.

## 2. Fast against slow statements (`scripts/scaled_timing.py`, k = 5)

3 timed runs per claim and statement after one untimed run; S0: runs 1, 2; cells give run 1 / run 2; S1: runs 1, 2; cells give run 1 / run 2.

| statement | S0 p50 ms | S0 p95 ms | S0 short of 5 rows | S1 p50 ms | S1 p95 ms | S1 short of 5 rows |
|---|---|---|---|---|---|---|
| HNSW index search | 1.43 / 1.45 | 1.95 / 1.97 | 0 / 0 | 1.68 / 1.68 | 2.72 / 2.69 | 0 / 0 |
| exact search (joined statement) | 15.74 / 15.87 | 17.01 / 17.13 | 0 / 0 | 25.81 / 25.79 | 29.70 / 29.66 | 0 / 0 |
| History filter, inline | 16.49 / 16.61 | 18.32 / 18.39 | 0 / 0 | 26.59 / 26.61 | 30.41 / 30.36 | 0 / 0 |
| History filter, overfetch 10 | 1.63 / 1.65 | 2.17 / 2.21 | 2,472 / 2,472 | 1.87 / 1.87 | 2.90 / 2.89 | 2,425 / 2,425 |

S0: p50 and p95 differ between runs by at most 1.5% (largest over the four statements).

S1: p50 and p95 differ between runs by at most 1.0% (largest over the four statements).

HNSW top 5 identical to the exact top 5: S0 2,962 / 2,963; S1 2,911 / 2,911.

## 3. Evidence scores and page coverage

`scripts/scaled_sentence_score.py`: the text units of each claim's top 10 chunks; the best 5 sentences and 25 list items; a claim counts when a complete gold set is among them.

| database | complete set in the top 10 chunks | 5 + 25 by embedding (FEVEROUS-style) | 5 + 25 in chunk order |
|---|---|---|---|
| S0 | 0.853 (2,558) | 0.546 (1,637) | 0.413 (1,239) |
| S1 | 0.825 (2,476) | 0.529 (1,588) | 0.397 (1,190) |

`scripts/scaled_page_coverage.py`: page coverage as in the FEVEROUS baseline code (share of a claim's gold pages among the first p distinct retrieved pages, mean over claims).

| claims | p = 1 | p = 2 | p = 3 | p = 5 | p = 10 |
|---|---|---|---|---|---|
| S0, all claims | 0.811 | 0.923 | 0.948 | 0.960 | 0.967 |
| S0, SUPPORTS + REFUTES | 0.830 | 0.936 | 0.958 | 0.968 | 0.975 |
| S1, all claims | 0.789 | 0.895 | 0.924 | 0.943 | 0.955 |
| S1, SUPPORTS + REFUTES | 0.814 | 0.911 | 0.937 | 0.954 | 0.965 |

## 4. Six groups, challenge tags, FEVEROUS's mix

Full tables: [breakdown/breakdown.md](breakdown/breakdown.md). At k = 5:

| database | overall | article@5 | evidence@5 | FEVEROUS-style |
|---|---|---|---|---|
| S0 | as measured | 0.984 | 0.780 | 0.546 |
| S0 | reweighted, all | 0.993 | 0.891 | 0.667 |
| S0 | reweighted, text | 0.993 | 0.901 | 0.682 |
| S1 | as measured | 0.978 | 0.751 | 0.529 |
| S1 | reweighted, all | 0.990 | 0.870 | 0.650 |
| S1 | reweighted, text | 0.989 | 0.881 | 0.665 |

## 5. Filter values (`scripts/scaled_filters.py`)

S0 (43,134 chunks): [filters_s0.md](filters/filters_s0.md).
S1 (132,359 chunks): [filters_s1.md](filters/filters_s1.md).

## 6. Filtered searches: the join order fixed and free (`join_order`)

### S0: 43,134 chunks, a random sample of 200 of 3,000 claims (seed 20261119), k = 5, 1 timed pass after one untimed pass, one run (run 1)

| filter | chunks passing | as written: plan | rows read | p50 ms | same rows as scan | free: plan | rows read | p50 ms | same rows as scan |
|---|---|---|---|---|---|---|---|---|---|
| none | 43,134 | vector index | 5 | 16.16 | 200/200 | vector index | 5 | 16.11 | 200/200 |
| length_50pct | 21,570 | vector index | 16 | 16.50 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 21,570 | 98.72 | 200/200 |
| length_5pct | 2,166 | vector index | 168 | 16.94 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 2,166 | 8.46 | 200/200 |
| length_0.5pct | 256 | chunk first; chunk ALL no key | 43,134 | 307.35 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 256 | 1.51 | 200/200 |
| length_0.05pct | 139 | chunk first; chunk ALL no key | 43,134 | 296.55 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 139 | 1.07 | 200/200 |
| edited_50pct | 21,578 | vector index | 13 | 16.33 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 21,578 | 91.69 | 200/200 |
| edited_5pct | 2,124 | vector index | 130 | 16.96 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 2,124 | 8.20 | 200/200 |
| edited_0.5pct | 219 | vector index | 1,552 | 24.97 | 198/200 | page first; chunk ref uq_chunk_page_ordinal | 219 | 1.42 | 200/200 |
| edited_0.05pct | 34 | chunk first; chunk ALL no key | 43,134 | 275.00 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 34 | 0.72 | 200/200 |
| category_50pct | 5,081 | vector index | 84 | 17.21 | 199/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 5,081 | 19.83 | 200/200 |
| category_5pct | 1,857 | vector index | 174 | 18.01 | 200/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 1,857 | 7.00 | 200/200 |
| category_0.5pct | 215 | vector index | 4,482 | 52.76 | 192/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 215 | 1.39 | 200/200 |
| category_0.05pct | 22 | vector index | 12,133 | 145.22 | 200/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 22 | 0.82 | 200/200 |
| linked_50pct | 1,629 | vector index | 212 | 19.34 | 199/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 1,629 | 6.57 | 200/200 |
| linked_0.5pct | 216 | vector index | 2,098 | 36.33 | 198/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 216 | 1.51 | 200/200 |
| linked_0.05pct | 22 | vector index | 13,358 | 164.88 | 200/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 22 | 0.91 | 200/200 |
| four_filters | 824 | vector index | 446 | 21.45 | 199/200 | <derived3> first; chunk ref uq_chunk_page_ordinal | 824 | 3.94 | 200/200 |

### S1: 132,359 chunks, a random sample of 200 of 3,000 claims (seed 20261119), k = 5, 1 timed pass after one untimed pass, one run (run 1)

| filter | chunks passing | as written: plan | rows read | p50 ms | same rows as scan | free: plan | rows read | p50 ms | same rows as scan |
|---|---|---|---|---|---|---|---|---|---|
| none | 132,359 | vector index | 5 | 26.77 | 200/200 | vector index | 5 | 26.65 | 200/200 |
| length_50pct | 66,171 | vector index | 17 | 26.68 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 66,171 | 1566.76 | 200/200 |
| length_5pct | 6,616 | vector index | 158 | 37.09 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 6,616 | 26.39 | 200/200 |
| length_0.5pct | 626 | chunk first; chunk ALL no key | 132,359 | 359.63 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 626 | 2.82 | 200/200 |
| length_0.05pct | 139 | chunk first; chunk ALL no key | 132,359 | 335.33 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 139 | 1.06 | 200/200 |
| edited_50pct | 66,208 | vector index | 13 | 26.40 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 66,208 | 1689.04 | 200/200 |
| edited_5pct | 6,599 | vector index | 132 | 34.41 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 6,599 | 25.84 | 200/200 |
| edited_0.5pct | 660 | vector index | 1,924 | 129.95 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 660 | 3.04 | 200/200 |
| edited_0.05pct | 77 | chunk first; chunk ALL no key | 132,359 | 336.89 | 200/200 | page first; chunk ref uq_chunk_page_ordinal | 77 | 0.86 | 200/200 |
| category_50pct | 9,267 | vector index | 154 | 38.78 | 197/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 9,267 | 36.12 | 200/200 |
| category_5pct | 7,313 | vector index | 202 | 40.27 | 200/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 7,313 | 29.64 | 200/200 |
| category_0.5pct | 660 | vector index | 6,148 | 323.29 | 175/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 660 | 2.95 | 200/200 |
| category_0.05pct | 66 | vector index | 19,656 | 1387.73 | 198/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 66 | 0.99 | 200/200 |
| linked_50pct | 8,203 | vector index | 170 | 40.06 | 200/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 8,203 | 34.13 | 200/200 |
| linked_5pct | 6,698 | vector index | 121 | 38.15 | 200/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 6,698 | 27.97 | 200/200 |
| linked_0.5pct | 662 | vector index | 1,888 | 142.39 | 193/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 662 | 3.13 | 200/200 |
| linked_0.05pct | 66 | vector index | 19,656 | 1385.72 | 198/200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 66 | 0.96 | 200/200 |
| four_filters | 3,107 | vector index | 440 | 54.47 | 197/200 | <derived3> first; chunk ref uq_chunk_page_ordinal | 3,107 | 12.69 | 200/200 |

## 7. Three regimes of one statement (`forms`)

### S1: 132,359 chunks, 3,000 claims, k = 5, one run (run 1)

| statement | chunks passing | ef_search | p50 ms | equal to exact | short | rows read | chunk access |
|---|---|---|---|---|---|---|---|
| the bare statement on chunk | 132,359 | 20 | 1.13 | 2,776/3,000 | 0 | 5 | index embedding |
| the same, joined to page | 132,359 | 20 | 26.97 | 3,000/3,000 | 0 | 5 | index embedding |
| WHERE chunk.n_words < 60 | 26,935 | 20 | 4.79 | 2,389/3,000 | 0 | 35 | index embedding |
| WHERE chunk.n_words < 60, joined to page | 26,935 | 20 | 27.14 | 2,995/3,000 | 0 | 34 | index embedding |
| WHERE chunk.n_words < 30 | 12,341 | 20 | 20.82 | 2,209/3,000 | 0 | 120 | index embedding |
| WHERE chunk.n_words < 30, joined to page | 12,341 | 20 | 27.59 | 2,984/3,000 | 0 | 114 | index embedding |
| the bare statement on chunk | 132,359 | 100 | 2.16 | 2,917/3,000 | 0 | 5 | index embedding |
| the same, joined to page | 132,359 | 100 | 26.68 | 3,000/3,000 | 0 | 5 | index embedding |
| WHERE chunk.n_words < 60 | 26,935 | 100 | 9.40 | 2,707/3,000 | 0 | 34 | index embedding |
| WHERE chunk.n_words < 60, joined to page | 26,935 | 100 | 26.75 | 2,995/3,000 | 0 | 34 | index embedding |
| WHERE chunk.n_words < 30 | 12,341 | 100 | 33.30 | 2,514/3,000 | 0 | 118 | index embedding |
| WHERE chunk.n_words < 30, joined to page | 12,341 | 100 | 27.30 | 2,984/3,000 | 0 | 114 | index embedding |

## 8. MariaDB against a copy of the vectors (`scripts/compare_separate_index.py`)

### S0: 43,134 chunks, 3,000 claims, k = 5, 1 timed pass after one untimed pass, one run (run 1)

The copy: 66,253,824 bytes of float32 vectors, exported in 0.69 s; the process grew by 214,454,272 bytes during the export; the hnswlib graph is 72,655,480 bytes saved, built in 1.4 s.

| method | kind | p50 ms | p95 ms | overlap with exact | same order as exact | short |
|---|---|---|---|---|---|---|
| MariaDB, bare k-NN statement (HNSW, M=16, ef_search 100) | approximate | 1.92 | 2.70 | 0.998 | 2,970/3,000 | 0/3,000 |
| MariaDB, joined statement (strategy inline) | exact | 16.17 | 17.71 | 1.000 | 3,000/3,000 | 0/3,000 |
| NumPy scan of the copy, ids only | exact | 3.27 | 3.32 | 1.000 | 3,000/3,000 | 0/3,000 |
| NumPy scan of the copy, then the hit rows from MariaDB by primary key | exact | 3.95 | 4.23 | 1.000 | 3,000/3,000 | 0/3,000 |
| hnswlib graph over the copy (M=16, ef 100), ids only | approximate | 0.31 | 0.37 | 0.977 | 2,710/3,000 | 0/3,000 |

section heading LIKE '%History%': 1,588 chunks pass.

| method | kind | p50 ms | p95 ms | overlap with exact | same order as exact | short |
|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 20.46 | 24.78 | 1.000 | 2,998/3,000 | 0/3,000 |
| MariaDB, strategy overfetch 10 | approximate | 2.35 | 3.92 | 0.391 | 524/3,000 | 2,471/3,000 |
| NumPy pre-filter and scan of the copy, ids only | exact | 5.44 | 5.60 | 1.000 | 3,000/3,000 | 0/3,000 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 6.11 | 6.66 | 1.000 | 3,000/3,000 | 0/3,000 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 8.69 | 11.47 | 1.000 | 2,993/3,000 | 0/3,000 |

the four filters: category 'Member states of the United Nations', at least 3,359 words, edited since 2020-12-18T02:38:12, linked from 'List of countries by maternal mortality ratio': 824 chunks pass.

| method | kind | p50 ms | p95 ms | overlap with exact | same order as exact | short |
|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 23.21 | 38.50 | 1.000 | 2,993/3,000 | 0/3,000 |
| MariaDB, strategy overfetch 10 | approximate | 4.43 | 5.83 | 0.142 | 166/3,000 | 2,834/3,000 |
| NumPy pre-filter and scan of the copy, ids only | exact | 0.62 | 0.68 | 1.000 | 3,000/3,000 | 0/3,000 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 1.18 | 1.43 | 1.000 | 3,000/3,000 | 0/3,000 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 8.31 | 14.96 | 1.000 | 3,000/3,000 | 0/3,000 |

### S1: 132,359 chunks, 3,000 claims, k = 5, 1 timed pass after one untimed pass, one run (run 1)

The copy: 203,303,424 bytes of float32 vectors, exported in 2.42 s; the process grew by 701,804,544 bytes during the export; the hnswlib graph is 222,955,496 bytes saved, built in 5.9 s.

| method | kind | p50 ms | p95 ms | overlap with exact | same order as exact | short |
|---|---|---|---|---|---|---|
| MariaDB, bare k-NN statement (HNSW, M=16, ef_search 100) | approximate | 2.87 | 4.84 | 0.993 | 2,917/3,000 | 0/3,000 |
| MariaDB, joined statement (strategy inline) | exact | 27.30 | 32.02 | 1.000 | 3,000/3,000 | 0/3,000 |
| NumPy scan of the copy, ids only | exact | 10.19 | 10.53 | 1.000 | 3,000/3,000 | 0/3,000 |
| NumPy scan of the copy, then the hit rows from MariaDB by primary key | exact | 11.06 | 13.04 | 1.000 | 3,000/3,000 | 0/3,000 |
| hnswlib graph over the copy (M=16, ef 100), ids only | approximate | 0.36 | 0.42 | 0.961 | 2,544/3,000 | 0/3,000 |

section heading LIKE '%History%': 5,363 chunks pass.

| method | kind | p50 ms | p95 ms | overlap with exact | same order as exact | short |
|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 37.36 | 56.43 | 1.000 | 2,996/3,000 | 0/3,000 |
| MariaDB, strategy overfetch 10 | approximate | 3.27 | 8.72 | 0.405 | 568/3,000 | 2,424/3,000 |
| NumPy pre-filter and scan of the copy, ids only | exact | 16.01 | 16.85 | 1.000 | 3,000/3,000 | 0/3,000 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 16.94 | 18.91 | 1.000 | 3,000/3,000 | 0/3,000 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 19.01 | 23.59 | 0.999 | 2,981/3,000 | 0/3,000 |

the four filters: category 'Member states of the United Nations', at least 5,214 words, edited since 2020-12-21T15:34:08, linked from 'Legality of cannabis': 3,107 chunks pass.

| method | kind | p50 ms | p95 ms | overlap with exact | same order as exact | short |
|---|---|---|---|---|---|---|
| MariaDB, strategy inline (the default) | exact | 58.69 | 189.01 | 0.998 | 2,962/3,000 | 0/3,000 |
| MariaDB, strategy overfetch 10 | approximate | 3.94 | 8.65 | 0.151 | 198/3,000 | 2,801/3,000 |
| NumPy pre-filter and scan of the copy, ids only | exact | 1.69 | 2.00 | 1.000 | 3,000/3,000 | 0/3,000 |
| NumPy pre-filter and scan, then the hit rows from MariaDB by primary key | exact | 2.44 | 3.38 | 1.000 | 3,000/3,000 | 0/3,000 |
| hnswlib graph with its filter callback (ef 100), ids only | approximate | 11.73 | 27.74 | 0.999 | 2,990/3,000 | 0/3,000 |

## 9. The index cache below and above the graph (`cache`)

### S1: 132,359 chunks, 3,000 claims, k = 5, 2 passes per size, configured 512 MB, restored 512 MB, one run (run 1)

| cache | statement | equal to exact (of 3,000), per pass | article recall, per pass | two passes agree | p50 ms |
|---|---|---|---|---|---|
| 512 MB (as configured) | bare | 2,917, 2,917 | 2,933, 2,933 | 3,000 | 2.56 |
| 512 MB (as configured) | joined | 3,000, 3,000 | 2,943, 2,943 | 3,000 | 28.43 |
| 16 MB, table not reopened | bare | 2,151, 2,162 | 2,657, 2,658 | 2,487 | 17.71 |
| 16 MB, table not reopened | joined | 2,314, 2,311 | 2,694, 2,691 | 2,867 | 22.00 |
| 256 MB | bare | 2,902, 2,911 | 2,931, 2,933 | 2,990 | 2.37 |
| 256 MB | joined | 3,000, 3,000 | 2,943, 2,943 | 3,000 | 27.20 |
| 192 MB | bare | 2,902, 2,911 | 2,931, 2,933 | 2,990 | 2.27 |
| 192 MB | joined | 3,000, 3,000 | 2,943, 2,943 | 3,000 | 27.53 |
| 160 MB | bare | 2,819, 2,823 | 2,910, 2,911 | 2,932 | 25.21 |
| 160 MB | joined | 2,734, 2,725 | 2,840, 2,840 | 2,729 | 351.45 |
| 128 MB | bare | 2,722, 2,724 | 2,874, 2,875 | 2,990 | 39.40 |
| 128 MB | joined | 2,440, 2,436 | 2,750, 2,743 | 2,971 | 493.34 |
| 64 MB | bare | 2,561, 2,548 | 2,798, 2,807 | 2,650 | 35.20 |
| 64 MB | joined | 2,283, 2,275 | 2,685, 2,690 | 2,291 | 166.27 |
| 16 MB | bare | 2,163, 2,162 | 2,658, 2,658 | 2,999 | 16.54 |
| 16 MB | joined | 2,314, 2,312 | 2,695, 2,690 | 2,824 | 21.92 |
| 512 MB (as configured, again) | bare | 2,902, 2,911 | 2,931, 2,933 | 2,990 | 2.14 |
| 512 MB (as configured, again) | joined | 3,000, 3,000 | 2,943, 2,943 | 3,000 | 27.31 |

## 10. The same vectors built several times (`builds`)

### S1, single timed build: 132,359 chunks, 1 build(s) per M, 3,000 claims, k = 5; exact ranking finds a gold page for 2,943

| M | ef_search | article recall | equal to exact (of 3,000) | overlap with exact | two builds agree | build s |
|---|---|---|---|---|---|---|
| 16 | 20 | 2,866 | 2,699 | 0.966 | - | 85.6 |
| 16 | 100 | 2,928 | 2,869 | 0.990 | - | 85.6 |

### S1, build series (run 1): 132,359 chunks, 20 build(s) per M, 3,000 claims, k = 5; exact ranking finds a gold page for 2,943

| M | ef_search | article recall | equal to exact (of 3,000) | overlap with exact | two builds agree | build s |
|---|---|---|---|---|---|---|
| 16 | 20 | 2,838 to 2,887 | 2,628 to 2,743 | 0.955 to 0.972 | 2,608 to 2,752 | 86.5 to 102.3 |
| 16 | 100 | 2,922 to 2,937 | 2,862 to 2,918 | 0.988 to 0.993 | 2,838 to 2,907 | 86.5 to 102.3 |
| 6 | 20 | 2,218 to 2,433 | 1,553 to 1,794 | 0.738 to 0.807 | 1,404 to 1,676 | 24.5 to 29.4 |
| 6 | 100 | 2,709 to 2,786 | 2,041 to 2,281 | 0.888 to 0.921 | 1,837 to 2,112 | 24.5 to 29.4 |

## 11. Search over a transaction's own uncommitted rows

### S0: k = 5, 3 builds per dataset, one run (run 1); mean overlap with the exact top k, the smallest and largest build mean

| data | state | ef_search | mean overlap | p50 ms |
|---|---|---|---|---|
| chunks | uncommitted | 20 | 0.872 to 0.881 | 0.326 to 0.366 |
| chunks | uncommitted | 100 | 0.875 to 0.881 | 0.306 to 0.345 |
| chunks | uncommitted | 400 | 0.875 to 0.881 | 0.447 to 0.460 |
| chunks | committed | 20 | 0.973 to 0.983 | 0.365 to 0.428 |
| chunks | committed | 100 | 0.993 to 0.996 | 0.751 to 0.928 |
| chunks | committed | 400 | 0.998 to 0.999 | 2.011 to 2.194 |
| random | uncommitted | 20 | 0.611 to 0.622 | 0.161 to 0.219 |
| random | uncommitted | 100 | 0.649 to 0.654 | 0.218 to 0.285 |
| random | uncommitted | 400 | 0.649 to 0.655 | 0.299 to 0.315 |
| random | committed | 20 | 0.802 to 0.815 | 0.168 to 0.221 |
| random | committed | 100 | 0.991 to 0.993 | 0.258 to 0.348 |
| random | committed | 400 | 0.995 to 0.997 | 0.455 to 0.520 |

Datasets: `chunks`: the 43134 chunk embeddings of wikilense_s0, queried with its 3000 claim embeddings; `random`: 1000 random normal vectors (seed 1), queried with 3000 random vectors.
