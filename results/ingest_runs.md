# Ingests and index rebuilds run by scripts/run_experiments.py

One row per `run_ingest` call or `ALTER TABLE` rebuild, in order. Seconds are the stages of `IngestReport`; an ingest whose configuration needs another M than the schema's is followed by a rebuild, shown in the same row (`total s`: "+ rebuild M=...").

| label | kind | configuration | chunks | hatnote units | words mean | total s | embed s | vector index bytes | at |
|---|---|---|---|---|---|---|---|---|---|
| stability_old | ingest | 120 words, overlap 1, M=6, prefix on | 8658 | 1316 | 93.8 | 31.43 + rebuild M=6 1.13 | 25.19 | 16777216 | 2026-09-22T01:34:49+00:00 |
| index_m_sweep_m6 | rebuild | M 6 -> 6 | 8658 | - | - | drop 0.83 + add 1.13 + analyze 0.003 | - | 16777216 | 2026-09-22T01:36:37+00:00 |
| index_m_sweep_m16 | rebuild | M 6 -> 16 | 8658 | - | - | drop 0.81 + add 2.94 + analyze 0.002 | - | 17825792 | 2026-09-22T01:36:48+00:00 |
| index_m_sweep_m32 | rebuild | M 16 -> 32 | 8658 | - | - | drop 0.84 + add 8.98 + analyze 0.003 | - | 19922944 | 2026-09-22T01:37:05+00:00 |
| index_m_restore_m6 | rebuild | M 32 -> 6 | 8658 | - | - | drop 0.81 + add 1.1 + analyze 0.002 | - | 16777216 | 2026-09-22T01:37:14+00:00 |
| chunk60 | ingest | 60 words, overlap 1, M=16, prefix on | 19065 | 1316 | 48.1 | 33.92 | 22.51 | 30408704 | 2026-09-22T01:37:48+00:00 |
| chunk120 | ingest | 120 words, overlap 1, M=16, prefix on | 8658 | 1316 | 93.8 | 24.8 | 18.58 | 17825792 | 2026-09-22T01:38:19+00:00 |
| chunk240 | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 21.3 | 16.65 | 13631488 | 2026-09-22T01:38:44+00:00 |
| prefix_on | ingest | 120 words, overlap 1, M=16, prefix on | 8658 | 1316 | 93.8 | 25.17 | 18.68 | 17825792 | 2026-09-22T01:39:13+00:00 |
| prefix_off | ingest | 120 words, overlap 1, M=16, prefix off | 8658 | 1316 | 93.8 | 24.03 | 17.19 | 17825792 | 2026-09-22T01:39:53+00:00 |
| filters_final | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 21.14 | 16.53 | 13631488 | 2026-09-22T01:40:31+00:00 |
| final | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 21.14 | 16.55 | 13631488 | 2026-09-22T01:41:38+00:00 |
| restore_defaults | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 21.0 | 16.44 | 13631488 | 2026-09-22T01:43:02+00:00 |
| stability_old | ingest | 120 words, overlap 1, M=6, prefix on | 8658 | 1316 | 93.8 | 29.0 + rebuild M=6 1.15 | 23.3 | 16777216 | 2026-10-21T22:38:32+00:00 |
| index_m_sweep_m6 | rebuild | M 6 -> 6 | 8658 | - | - | drop 0.84 + add 1.11 + analyze 0.002 | - | 16777216 | 2026-10-21T22:40:16+00:00 |
| index_m_sweep_m16 | rebuild | M 6 -> 16 | 8658 | - | - | drop 0.87 + add 2.94 + analyze 0.002 | - | 17825792 | 2026-10-21T22:40:26+00:00 |
| index_m_sweep_m32 | rebuild | M 16 -> 32 | 8658 | - | - | drop 0.85 + add 9.08 + analyze 0.003 | - | 19922944 | 2026-10-21T22:40:43+00:00 |
| index_m_restore_m6 | rebuild | M 32 -> 6 | 8658 | - | - | drop 0.87 + add 1.15 + analyze 0.002 | - | 16777216 | 2026-10-21T22:40:51+00:00 |
| chunk60 | ingest | 60 words, overlap 1, M=16, prefix on | 19065 | 1316 | 48.1 | 32.37 | 22.81 | 30408704 | 2026-10-21T22:41:24+00:00 |
| chunk120 | ingest | 120 words, overlap 1, M=16, prefix on | 8658 | 1316 | 93.8 | 24.51 | 18.86 | 17825792 | 2026-10-21T22:41:53+00:00 |
| chunk240 | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 21.16 | 16.91 | 13631488 | 2026-10-21T22:42:18+00:00 |
| prefix_on | ingest | 120 words, overlap 1, M=16, prefix on | 8658 | 1316 | 93.8 | 24.78 | 19.01 | 17825792 | 2026-10-21T22:42:46+00:00 |
| prefix_off | ingest | 120 words, overlap 1, M=16, prefix off | 8658 | 1316 | 93.8 | 23.43 | 17.6 | 17825792 | 2026-10-21T22:43:25+00:00 |
| filters_final | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 21.32 | 16.95 | 13631488 | 2026-10-21T22:44:02+00:00 |
| final | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 21.32 | 16.95 | 13631488 | 2026-10-21T22:45:05+00:00 |
| restore_defaults | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 21.14 | 16.83 | 13631488 | 2026-10-21T22:46:21+00:00 |
| scale_restore | ingest | 240 words, overlap 1, M=16, prefix on | 4598 | 1316 | 156.6 | 22.92 | 18.24 | 13631488 | 2026-10-21T23:46:15+00:00 |
