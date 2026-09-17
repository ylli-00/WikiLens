# Re-ingests run by scripts/run_experiments.py

One row per `run_ingest` call, in order. Seconds are the stages of `IngestReport`.

| label | max_words | overlap | prefix | chunks | words mean | words median | total s | embed s | load s | vector index bytes | ingested_at |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chunk60 | 60 | 1 | True | 19467 | 47.9 | 50.0 | 29.2 | 23.0 | 5.28 | 28311552 | 2026-09-17T12:01:50+00:00 |
| chunk240 | 240 | 1 | True | 4751 | 153.9 | 181.0 | 20.79 | 17.43 | 2.32 | 12582912 | 2026-09-17T12:02:15+00:00 |
| noprefix | 120 | 1 | False | 8868 | 93.0 | 103.0 | 21.97 | 17.79 | 3.17 | 16777216 | 2026-09-17T12:02:42+00:00 |
| filters_defaults | 120 | 1 | True | 8868 | 93.0 | 103.0 | 23.16 | 19.14 | 3.07 | 16777216 | 2026-09-17T12:03:13+00:00 |
| restore_defaults | 120 | 1 | True | 8868 | 93.0 | 103.0 | 23.52 | 19.27 | 3.25 | 16777216 | 2026-09-17T12:04:08+00:00 |
