# scale

Created 2026-10-21T23:46:16+00:00 by `scripts/run_experiments.py scale` on MariaDB 11.8.9-MariaDB-ubu2404. The 100 committed pages plus random distractor pages of the full FEVEROUS Wikipedia (seed 20201231), ingested with the FINAL settings; the claims and their gold pages are unchanged. Per claim, k 10, ef_search 100, one untimed run and 3 timed runs of each statement: `hnsw` (strategy none, the HNSW search), `exact` (inline without a filter, the exact ranking), and the inline and overfetch-10 statements under `heading LIKE '%History%'`. "short" counts queries with fewer than k hits.

| pages | chunks | graph | HNSW article@10 | exact article@10 | identical hit lists | HNSW ms p50 / p95 | exact ms p50 / p95 | History inline ms p50 / p95 | History overfetch 10: ms p50, short |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 4,598 | 14 MB | 74/75 | 74/75 | 75/75 | 0.83 / 1.90 | 2.52 / 5.16 | 3.30 / 5.04 | 1.03, 69/75 |
| 5,100 | 26,700 | 42 MB | 74/75 | 74/75 | 73/75 | 1.05 / 1.55 | 10.27 / 14.18 | 11.54 / 17.67 | 1.44, 68/75 |
| 20,100 | 89,650 | 113 MB | 71/75 | 71/75 | 72/75 | 1.23 / 3.00 | 19.17 / 27.94 | 21.30 / 32.84 | 1.70, 65/75 |

- 100 pages: ingest 25.54 s (embedding 21.13 s); History inline short 0.
- 5,100 pages: ingest 101.21 s (embedding 74.21 s); History inline short 0.
- 20,100 pages: ingest 328.52 s (embedding 240.12 s); History inline short 0.
