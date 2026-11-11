# Evaluation: method, every result table, and the reasons for each parameter

The [README](../README.md) has the headline numbers. This page has the method in full, the
other runs on the same ingest, the evidence behind each chosen parameter, and how to
reproduce all of it. Every number is read from a file in [results/](../results/);
[results/SUMMARY.md](../results/SUMMARY.md) is the overview of every experiment.

## Method and results

**Method.** Each of the 75 claims is embedded once and searched once per `k` with `LIMIT k`. Those
hits give the metrics at `k`, and the same statement is the one timed.

- Article recall@k: a gold page of the claim is among the pages of the top-k chunks. All 75 claims
  cite one page.
- Evidence recall@k: every sentence or list item of at least one gold evidence set is covered by
  the top-k chunks, through `chunk_sentence`. It is counted over the 66 claims whose gold set is
  made of text units.
- Unit coverage@k: the share of gold units covered.
- Latency: the SQL round trip of one `search()` call, warm, with 5 repeats per claim (375 samples
  per `k`), reported as p50 and p95. The query embedding is timed separately.

Machine: i9-13900H, 15 GB, RTX 4060 Laptop GPU, Ubuntu 24.04, MariaDB 11.8.9 in Docker, Python
3.12.3. The laptop was upgraded to Ubuntu 26.04 afterwards; the files dated 2026-11-11 were
measured there, with Python 3.12.16 and the same package versions.

**Headline** (`results/final_ef_100.md`: 240-word chunks, M=16, `ef_search=100`, strategy `none`):

| k | article recall | evidence recall | unit coverage | SQL p50 | SQL p95 |
|---|---|---|---|---|---|
| 1 | 63/75 (0.84) | 39/66 (0.59) | 0.60 | 0.50 ms | 0.86 ms |
| 3 | 74/75 (0.99) | 52/66 (0.79) | 0.81 | 0.45 ms | 0.78 ms |
| 5 | 74/75 (0.99) | 55/66 (0.83) | 0.85 | 0.45 ms | 0.79 ms |
| 10 | 74/75 (0.99) | 60/66 (0.91) | 0.92 | 0.51 ms | 0.94 ms |
| 20 | 74/75 (0.99) | 65/66 (0.98) | 0.98 | 0.64 ms | 1.10 ms |

Query embedding: 4.6 ms p50 on the GPU.

![Article and evidence recall at k](img/recall_at_k.svg)

**The headline run again, and without the GPU.** On 2026-11-11 the same evaluation was repeated
four times on the default ingest: twice as it is, twice with the GPU hidden
(`CUDA_VISIBLE_DEVICES="" wikilense eval --name ...`), so that the query is embedded on the CPU. A
fifth row is the evaluation in a fresh clone whose chunks were embedded on the CPU too.

| run | query embedded on | SQL p50 / p95 at k=10 | query embedding p50 / p95 | recall |
|---|---|---|---|---|
| `final_ef_100` (2026-10-21) | GPU | 0.51 / 0.94 ms | 4.58 / 4.70 ms | the table above |
| `final_ef_100_rerun_gpu1` | GPU | 0.48 / 0.69 ms | 4.39 / 4.50 ms | identical |
| `final_ef_100_rerun_gpu2` | GPU | 0.56 / 1.16 ms | 4.46 / 4.61 ms | identical |
| `final_ef_100_rerun_cpu1` | CPU | 0.53 / 1.12 ms | 17.83 / 22.68 ms | identical |
| `final_ef_100_rerun_cpu2` | CPU | 0.55 / 1.06 ms | 17.68 / 22.53 ms | identical |
| fresh clone, CPU ingest ([fresh_clone_run.md](../results/fresh_clone_run.md)) | CPU | 0.47 / 0.62 ms | 18.22 / 23.02 ms | identical |

What this shows:

- Recall is the same in every row, at every `k`. The reruns search a different HNSW build than
  the October run (the default ingest was rebuilt after it), and their 20-chunk hit lists equal the
  October ones for 74 of 75 claims.
- Without a GPU the query embedding takes about 18 ms instead of 4.5 ms. The SQL time does not
  depend on it. The hit lists with CPU-embedded queries equal the GPU ones for 75 of 75 claims.
- The SQL time moves between runs on this laptop: 0.48 and 0.56 ms for the same statement a
  minute apart. Compare numbers inside one file; do not read much into the second decimal across
  files.

**The same ingest, other statements** (article recall@10, evidence recall@20, SQL p50/p95 at k=10):

| run | statement | article@10 | evidence@20 | SQL p50 / p95 |
|---|---|---|---|---|
| `final_inline` | exact ranking (joined statement, no filter) | 74/75 | 65/66 | 2.24 / 2.57 ms |
| `final_ef_100` | index, ef_search 100 | 74/75 | 65/66 | 0.51 / 0.94 ms, hit lists = exact for 74/75 |
| `final_ef_20` | index, ef_search 20 | 74/75 | 65/66 | 0.45 / 0.70 ms, hit lists = exact for 65/75 |
| `final_rrf` | vector + full text, RRF | 74/75 | 64/66 | 9.11 / 12.84 ms |
| `final_oracle_titles` | inline, gold page as `titles` filter | 75/75 | 66/66 | 2.45 / 8.59 ms; evidence in the first chunk for 42/66, within 5 for 59/66 |
| `final_ef_100_after_restart` | index, ef_search 100, after a container restart (`docker restart`) | 74/75 | 65/66 | hit lists identical to before for 75/75 |

**Reading the numbers.** The one claim whose gold page never appears is claim 87976, a
NOT ENOUGH INFO claim about a by-election; its gold page is "Lincoln, England". The embedding
itself misses it: the exact ranking misses it too (`results/final_ef_100.md`, worst-cases table).
On these claims the hybrid does not beat the vector ranking, and it costs eighteen times more. The
oracle row bounds what better chunk ranking could give: with the page known, 42 of 66 gold sets
are already in the first chunk.

For comparison, the server defaults, `M=6` and `ef_search=20`, found 68 of 75 gold pages at k=10
on the 120-word ingest of the same corpus (`results/ef_20.md`, 8,658 chunks), where the exact
ranking of those vectors finds 74. On the 240-word ingest the chosen settings find 74, which is
what the exact ranking finds there too.

**Reproduce.**
- After `wikilense ingest`, `wikilense eval --name mine` writes `results/mine.json` and `.md` with
  the headline table for the current ingest. The counts match the table above up to the HNSW
  build: a fresh graph differs in a few chunk lists, while recall stays the same.
- `scripts/run_experiments.py all` reruns the groups in `results/SUMMARY.md` (about 10 minutes
  with a GPU); `scale` runs only when named. The groups are stability across cache sizes and a
  restart, the `ef_search` sweep, the `M` sweep with rebuild times, chunk sizes at equal retrieved
  text, the prefix, the filter strategies, the final configuration, the search before `COMMIT`,
  the optimizer's choice per filter, and the scale test with distractor pages. The scale test
  needs the full FEVEROUS Wikipedia (`feverous_wikiv1.db`, 53.5 GB, path in
  `WIKILENSE_FEVEROUS_DB`).
- Every results file records the ingest parameters, the effective `ef_search`, the cache size, the
  index `M` and the chunk count it was measured on.
- `scripts/make_charts.py` redraws the charts above from the JSON files.
- `scripts/compare_separate_index.py` reruns the comparison with a copy of the vectors in the
  application's memory (`results/separate_index.md`; the README section "Where this approach
  loses"). It only reads the database.
- `scripts/measure_index_behaviour.py` reruns five measurements of the index's behaviour
  ([docs/VECTOR_INDEX.md](VECTOR_INDEX.md)): `forms` (`results/statement_forms.md`) and `join_order`
  (`results/join_order.md`) read the corpus database; `builds` (`results/build_variance.md`, ten
  builds of the graph per `M`) and `isolation` (`results/isolation.md`) work on scratch tables in
  the test database, so do not run them next to the test suite; `cache`
  (`results/cache_size.md`) changes `mhnsw_max_cache_size` for the whole server while it runs and
  puts the configured value back, so run nothing else next to it.

## Parameters and why

| Parameter | Chosen | Evidence (details in results/SUMMARY.md) |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5`, 384 dimensions, 33M parameters, MIT licence | chosen in phase 1 for passage retrieval at this size; no second model was measured. 1,059 passages/s on the RTX 4060 and 78/s on the CPU for 100-word texts (phase-1 measurement in docs/DESIGN.md, not part of `results/`), so a grader without a GPU can ingest; normalised vectors, so cosine distance is `1 - dot` |
| Embedded text | `title > section path: chunk text` | on the exact ranking the prefix raises evidence recall@20 from 61 to 63 of 66 and leaves article recall@10 at 74 of 75 (section 5) |
| Chunk size | 240 words, overlap 1 unit, never across a section boundary | at equal retrieved text the 60-, 120- and 240-word chunkings are close: article recall 74/74/74 of 75 and evidence recall 54/57/55 of 66 at 1,200 words (60/63/60 at 2,400); 240 stores 4,598 vectors instead of 8,658 or 19,065 and a 13 MB graph instead of 17 or 29 (section 4) |
| Hatnotes | "Main article: ...", "See also: ...", "For other uses ..." are stored as sentences but kept out of chunks | 1,316 of 33,637 units; before the change they formed short chunks that ranked high under section filters (found in the phase-4 review; not recorded in `results/`) |
| Index `M` | 16 (server default 6) | on the same vectors at `ef_search=20`: article recall@10 67/74/73 of 75 and evidence recall@20 54/64/62 of 66 for M 6/16/32, against 74 and 63 exact; at `ef_search=100` M=16 returns the exact hit list for 69 of 75 claims (M=32: 73); build 2.9 s (M=6 1.1 s, M=32 9.1 s), graph 17 MB (section 3) |
| `mhnsw_ef_search` | 100 per query (server default 20) | on the final ingest hit lists identical to the exact ranking for 74 of 75 claims against 65 at 20, for 0.51 ms instead of 0.45 ms at k=10 (section 7) |
| Strategy with filters | `inline`; `overfetch` when latency matters more than a full result | inline always fills `k`, at 2.05 to 3.22 ms p50 while it walks the index and 5.17 ms when the optimizer scans instead; overfetch 10 costs 0.76 to 0.78 ms but comes back short under selective filters (sections 6 and 10) |
| Index cache | 512 MB (`--mhnsw-max-cache-size`) | the graph must fit: at 16 MB and at 512 MB the hit lists are the same and exact, but with 8 MB or less for the 13.0 MB graph of the final ingest they were exact for only 57 to 67 of 75 claims and the index query took roughly three to five times as long (`results/cache_size.md`); the graph of the 120-word ingest is 16 MB with `M=6` and 17 MB with `M=16`, at or above the default limit |
