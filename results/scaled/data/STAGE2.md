# Stage 2: S0 and S1 in MariaDB (2026-11-12)

Measured on the night of 2026-11-12. Laptop setting: Turbo profile,
CPU `max_perf_pct` 50, GPU 2,100 MHz (a root helper that holds both caps), the thermal guard, 4 threads,
5 GB memory cap, one job at a time. Code: `scripts/scaled_stage2.py`.

## Databases
`wikilense_s0` and `wikilense_s1`, created 03:03:21 as root through `docker exec` with exactly:
`CREATE DATABASE wikilense_s0; CREATE DATABASE wikilense_s1; GRANT ALL PRIVILEGES ON wikilense_s0.* TO 'wikilense'@'%'; GRANT ALL PRIVILEGES ON wikilense_s1.* TO 'wikilense'@'%';`
The graded `wikilense` and `wikilense_test` were not touched.

## Embedding (once, for S1; S0's chunk texts are a subset)
- 132,359 distinct chunk texts (= 132,359 chunks), 499.5 s on `cuda`,
  BAAI/bge-small-en-v1.5 at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, batch 64, saved in parts of 10,000 under `vectors/`.
- The texts' SHA-256 equals the one `scaled_build.py corpora` recorded (checked before embedding).
- Vectors are computed per part, not in one call, so they are not bit-identical to what one
  `wikilense ingest` call would compute (same model, revision and texts).
- Temperatures: at most CPU 73 C, board 73 C, GPU 72 C; no pause (`temps.csv`).

## Ingest (`ingest.run_ingest` with the saved vectors, looked up by text)
| Folder | Database | Pages | Chunks | Total s | Parse s | Load s (inserts + index) | Resolve s | Vector-index graph (bytes) | Evidence ids with a page / a text unit |
|---|---|---|---|---|---|---|---|---|---|
| S0 | `wikilense_s0` | 3,851 | 43,134 | 54.6 | 9.3 | 43.6 | 0.4 | 58,720,256 | 8,408 / 7,877 of 8,408 |
| S1 | `wikilense_s1` | 7,686 | 132,359 | 212.8 | 29.8 | 171.6 | 8.0 | 163,577,856 | 8,408 / 7,877 of 8,408 |

Graph size: the hidden InnoDB tablespace `chunk#i#04` (`information_schema.innodb_sys_tablespaces`,
read as root as `scripts/run_experiments.py` does). Both below `mhnsw_max_cache_size` (512 MB).
`verify`: pages, chunks, words in chunks, claims, evidence ids and their resolution equal
`corpora.json` in both databases. Smoke query on `wikilense_s1` (claim 85380): top 1 is its gold
chunk (Hongze District, distance 0.0673). No `page_meta.jsonl` yet (Stage 3), so the category and
edit-date columns are empty. Max temperature during the ingests: CPU 67 C, board 66 C, no pause.
