# Evaluation: method, every result table, and the reasons for each parameter

The [README](../README.md) has the headline numbers. This page has the method in full and every
table behind them, for two sets of measurements:

- **The scaled run** is the reported result: 3,000 FEVEROUS claims searched on two corpora of
  43,134 and 132,359 chunks. Its files are in [results/scaled/](../results/scaled/), and
  [results/scaled/SUMMARY.md](../results/scaled/SUMMARY.md) has every table in one place.
- **The demo corpus** is the 100 committed pages and their 75 claims. It is what `make setup` loads
  and `make eval` measures, so a reader can check an install against it, and it is the development
  set on which the settings were chosen. Its files are in [results/](../results/), with
  [results/SUMMARY.md](../results/SUMMARY.md) as the overview.

Every number below is read from one of those files.

## The scaled run

### Method

- **Claims and corpora.** 3,000 claims from FEVEROUS train and dev: per label 700 whose every
  evidence set lies on one page and 300 whose every set spans two or more pages, all with an
  evidence set made only of text ([DATA.md](DATA.md) has the rule, and the claims are in
  [data/scaled/claims_3000.jsonl](../data/scaled/claims_3000.jsonl)). They are searched on two
  databases: S0, the gold pages only (3,851 pages, 43,134 chunks), and S1, S0 plus one
  distractor page per gold article (7,686 pages, 132,359 chunks). The settings are those of the
  demo corpus: 240-word chunks, overlap 1, bge-small-en-v1.5 with the title prefix, `M=16`, cosine,
  `mhnsw_ef_search` 100.
- **Search.** Each claim is embedded once and searched once per `k` (1, 3, 5, 10, 20) with
  `LIMIT k`, through the bare index query (strategy `none`). The headline is k = 5: it is
  FEVEROUS's own `k` and the number of passages an LLM fact-checker reads.
- **Article recall@k**: a gold page of the claim is among the pages of the top-k chunks.
  **Evidence recall@k**: every sentence of at least one gold evidence set made only of sentences is
  covered by the top-k chunks; all 3,000 claims have such a set.
- **Latency**: the SQL round trip of one `search()` call as the Python client sees it, warm, after
  one untimed pass, 5 timed passes per claim (15,000 samples per `k`), p50 and p95. The query
  embedding is timed separately.
- **Each database was measured twice.** Recall was identical in both runs, and the latencies
  differ by at most 1.3% (p50 and p95; [SUMMARY.md](../results/scaled/SUMMARY.md)). The tables
  give run 1 / run 2 where both exist.
- **Machine setting.** Every scaled measurement ran with the laptop's CPU capped at 50% of its
  speed and the GPU at 2,100 MHz, one job at a time, so that long runs stay cool. The times are
  therefore comparable with each other, but not with the 100-page times further down, which ran
  without the caps.

### Recall and latency of the index search

From [results/scaled/eval/](../results/scaled/eval/) (`s0_full_run1` and `s1_full_run1`, and the
`_run2` files):

| k | S0 article recall | S0 evidence recall | S0 SQL p50 | S0 SQL p95 | S1 article recall | S1 evidence recall | S1 SQL p50 | S1 SQL p95 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.965 (2,895) | 0.446 (1,337) | 1.28 / 1.27 ms | 1.74 / 1.73 ms | 0.938 (2,814) | 0.434 (1,303) | 1.86 / 1.86 ms | 2.96 / 2.98 ms |
| 3 | 0.979 (2,937) | 0.703 (2,108) | 1.12 / 1.13 ms | 1.57 / 1.56 ms | 0.970 (2,911) | 0.681 (2,042) | 1.52 / 1.52 ms | 2.55 / 2.59 ms |
| **5** | **0.984 (2,951)** | **0.780 (2,341)** | **1.11 / 1.11 ms** | 1.52 / 1.52 ms | **0.978 (2,933)** | **0.751 (2,253)** | **1.43 / 1.42 ms** | 2.32 / 2.30 ms |
| 10 | 0.986 (2,959) | 0.853 (2,558) | 1.21 / 1.22 ms | 1.71 / 1.70 ms | 0.982 (2,947) | 0.826 (2,479) | 1.66 / 1.65 ms | 2.69 / 2.68 ms |
| 20 | 0.990 (2,971) | 0.893 (2,678) | 1.41 / 1.41 ms | 2.06 / 2.06 ms | 0.986 (2,957) | 0.870 (2,609) | 2.17 / 2.17 ms | 3.36 / 3.36 ms |

The query embedding took 6.09 / 6.15 ms (S0) and 6.07 / 6.07 ms (S1) at the median, on the GPU.

What the distractor pages cost: going from S0 to S1 triples the chunk table, adding for each gold
article the page it links to most. Article recall@5 falls from 0.984 to 0.978, evidence recall@5 from
0.780 to 0.751, and the index search takes 1.43 instead of 1.11 ms.

### Fast and slow statements

`scripts/scaled_timing.py` times four statements on all 3,000 claims at k = 5, one untimed and
3 timed runs per claim and statement, twice per database
([results/scaled/timing/](../results/scaled/timing/)). The History filter is
`section.heading LIKE '%History%'`.

| statement | S0 p50 / p95 ms | S1 p50 / p95 ms | results short of 5 rows, S0 / S1 |
|---|---|---|---|
| HNSW index search | 1.43 / 1.95 | 1.68 / 2.72 | 0 / 0 |
| exact search (the joined statement) | 15.74 / 17.01 | 25.81 / 29.70 | 0 / 0 |
| History filter, `inline` | 16.49 / 18.32 | 26.59 / 30.41 | 0 / 0 |
| History filter, `overfetch 10` | 1.63 / 2.17 | 1.87 / 2.90 | 2,472 / 2,425 |

Run 1 is shown; run 2 is within 1.5% (p50 and p95, both files are in the folder).

- The exact search costs about 11 times the index search on S0 and 15 times on S1, and its cost
  grows with the table. The `inline` History filter costs as much as the exact search.
- `overfetch 10` is as fast as the index search, but returns fewer than 5 rows for 2,472 of 3,000
  claims on S0 and 2,425 on S1.
- The HNSW top 5 equals the exact top 5 for 2,962 (run 2: 2,963) of 3,000 claims on S0 and 2,911
  on S1 (both runs). In this run a gold page is in the top 5 for 2,956 (HNSW) against 2,962 (exact)
  claims on S0, and 2,933 against 2,943 on S1 (the main harness: 2,951 and 2,933; the JSON
  field is named `article_hits_at_10`, but k was 5). The lists differed for a few claims between separate sessions
  on the same graph (2,962 against 2,963); we did not look into why.

### The six claim groups

From [results/scaled/breakdown/breakdown.md](../results/scaled/breakdown/breakdown.md), which has
every `k`. At k = 5, with the FEVEROUS-style score of the next section:

| group (claims) | S0 article@5 | S0 evidence@5 | S0 FEVEROUS-style | S1 article@5 | S1 evidence@5 | S1 FEVEROUS-style |
|---|---|---|---|---|---|---|
| SUPPORTS, one article (700) | 0.994 | 0.960 | 0.750 | 0.994 | 0.956 | 0.739 |
| SUPPORTS, two or more (300) | 1.000 | 0.633 | 0.280 | 1.000 | 0.553 | 0.260 |
| REFUTES, one article (700) | 0.993 | 0.956 | 0.769 | 0.986 | 0.940 | 0.747 |
| REFUTES, two or more (300) | 0.993 | 0.327 | 0.090 | 0.990 | 0.287 | 0.080 |
| NOT ENOUGH INFO, one article (700) | 0.950 | 0.866 | 0.616 | 0.936 | 0.833 | 0.593 |
| NOT ENOUGH INFO, two or more (300) | 0.990 | 0.353 | 0.107 | 0.983 | 0.303 | 0.103 |
| all 3,000 claims | 0.984 | 0.780 | 0.546 | 0.978 | 0.751 | 0.529 |

- **Claims on two or more pages are harder.** For claims on one page the full evidence is in the top 5
  chunks for 0.83 to 0.96 of them. For claims that need two or more pages it is 0.29 to 0.63 (S0 and S1 together): a
  chunk holds text of one page only, so k = 1 never covers such a claim, and the second page has
  to rank high too.
- A gold page is found for almost every claim in every group. NOT ENOUGH INFO claims on one page are the
  hardest group for the page itself (0.950 and 0.936).

### FEVEROUS's challenge tags

The annotator who verified a FEVEROUS claim tagged its main difficulty. The pick did not balance on
the tag; recall per tag at k = 5 (same file):

| challenge tag (claims) | S0 article@5 | S0 evidence@5 | S0 FEVEROUS-style | S1 article@5 | S1 evidence@5 | S1 FEVEROUS-style |
|---|---|---|---|---|---|---|
| Other (1,906) | 0.991 | 0.887 | 0.669 | 0.986 | 0.868 | 0.652 |
| Multi-hop Reasoning (709) | 0.997 | 0.518 | 0.223 | 0.993 | 0.468 | 0.210 |
| Entity Disambiguation (151) | 0.921 | 0.669 | 0.444 | 0.881 | 0.616 | 0.411 |
| Search terms not in claim (131) | 0.885 | 0.733 | 0.519 | 0.885 | 0.687 | 0.504 |
| Numerical Reasoning (71) | 0.972 | 0.789 | 0.606 | 0.972 | 0.746 | 0.606 |
| Combining Tables and Text (32) | 1.000 | 0.969 | 0.812 | 1.000 | 0.969 | 0.812 |

The two tags about finding the page are where the page is missed most: Entity Disambiguation
(0.881 on S1) and Search terms not in claim (0.885). Multi-hop Reasoning finds the page but not
the whole evidence (0.468 on S1). The 32 claims tagged Combining Tables and Text all have a
text-only evidence set too, which is why they are in the pick.

### Reweighted to FEVEROUS's mix

Our pick is balanced by label and by the number of pages; FEVEROUS is not. A third of our claims
are NOT ENOUGH INFO, against 3.5% of FEVEROUS train + dev, and 30% of our REFUTES claims need two
pages. An overall number is therefore given both as measured and reweighted.

- **Weights.** From [analysis/feverous/feverous_claims.csv](../analysis/feverous/feverous_claims.csv),
  one row per FEVEROUS train and dev claim (79,181). A claim is in group (label, one article) when
  every evidence set lies on one page (`max_pages` = 1), and in (label, two or more) when every set
  spans two or more pages (`min_pages` >= 2); a claim with sets of both kinds is in no group.
  Weighting "all" counts every claim of the six groups (78,840); weighting "text" counts only
  those with an evidence set of sentences only or of list items only (34,480).
- **Reweighted value** = the sum over the six groups of (weight x the rate measured on our claims
  of that group).
- The weighting cannot reproduce one detail of the pick: our "one article" also requires every set
  to name the same page, which the table does not record. Its "two or more" counts equal the
  pick's.

| group | our claims | FEVEROUS, all | weight | FEVEROUS, text | weight |
|---|---|---|---|---|---|
| SUPPORTS, one article | 700 | 33,981 | 0.4310 | 13,988 | 0.4057 |
| SUPPORTS, two or more | 300 | 11,529 | 0.1462 | 4,067 | 0.1180 |
| REFUTES, one article | 700 | 29,100 | 0.3691 | 14,453 | 0.4192 |
| REFUTES, two or more | 300 | 1,494 | 0.0189 | 584 | 0.0169 |
| NOT ENOUGH INFO, one article | 700 | 2,059 | 0.0261 | 1,125 | 0.0326 |
| NOT ENOUGH INFO, two or more | 300 | 677 | 0.0086 | 263 | 0.0076 |

| at k = 5 | S0 article | S0 evidence | S0 FEVEROUS-style | S1 article | S1 evidence | S1 FEVEROUS-style |
|---|---|---|---|---|---|---|
| as measured | 0.984 | 0.780 | 0.546 | 0.978 | 0.751 | 0.529 |
| reweighted, all | 0.993 | 0.891 | 0.667 | 0.990 | 0.870 | 0.650 |
| reweighted, text | 0.993 | 0.901 | 0.682 | 0.989 | 0.881 | 0.665 |

On FEVEROUS's mix the numbers are higher, because that mix is mostly SUPPORTS and REFUTES claims
on one page, the easiest groups.

### A FEVEROUS-style evidence score, and page coverage

Our evidence recall counts a claim when a complete gold set is inside the top k chunks, and five
chunks hold up to 1,200 words. FEVEROUS lets a system return at most 5 sentences and 25 table cells
or list items, and counts a claim only when a complete gold set is among them.
`scripts/scaled_sentence_score.py` applies that rule: it takes the text units of each claim's top
10 chunks, ranks them against the claim, keeps the best 5 sentences and 25 list items, and counts
the claims with a complete sentence-only gold set among them
([results/scaled/sentence_score/](../results/scaled/sentence_score/)).

| database | complete set in the top 10 chunks (ours) | 5 + 25, ranked by embedding (FEVEROUS-style) | 5 + 25 in chunk order |
|---|---|---|---|
| S0 | 0.853 (2,558) | **0.546 (1,637)** | 0.413 (1,239) |
| S1 | 0.825 (2,476) | **0.529 (1,588)** | 0.397 (1,190) |

Each unit is embedded once with the same model as the chunks. Ranking the units by embedding
finds a complete set for 1,588 claims on S1, against 1,190 when the first units of the top chunks
are kept in order.

Page coverage as in the FEVEROUS paper's Table 4 uses the definition of the FEVEROUS baseline
code (`page_coverage` in `src/feverous/baseline/retriever/eval_doc_retriever.py`; the paper gives
no formula): per claim, the share of its gold pages (pooled over all evidence sets) among the first
p distinct retrieved pages, averaged over the claims. The pages are the distinct pages of the top 50
chunks in rank order ([results/scaled/page_coverage/](../results/scaled/page_coverage/)).

| claims | p = 1 | p = 2 | p = 3 | p = 5 | p = 10 |
|---|---|---|---|---|---|
| S0, all claims | 0.811 | 0.923 | 0.948 | 0.960 | 0.967 |
| S0, SUPPORTS + REFUTES | 0.830 | 0.936 | 0.958 | 0.968 | 0.975 |
| S1, all claims | 0.789 | 0.895 | 0.924 | 0.943 | 0.955 |
| S1, SUPPORTS + REFUTES | 0.814 | 0.911 | 0.937 | 0.954 | 0.965 |

These numbers are not comparable with those of FEVEROUS's own retriever. The paper's Table 4 (the
dev set, its own retriever, 5.4 million pages; [the paper](https://arxiv.org/abs/2106.05707)) gives 0.39, 0.49, 0.58 and 0.69 at 1, 2, 3 and 5 pages. Our corpus is
nearly three orders of magnitude smaller (7,686 pages for S1), and our claims are the text-only
ones, so even the FEVEROUS-style score is easier here than in the paper.

### Reproduce

The scaled run needs the full FEVEROUS Wikipedia (`feverous_wikiv1.db`, 53.5 GB), a GPU and
several hours. [SETUP.md](SETUP.md#reproduce-the-scaled-run) has the commands in order.

## The demo corpus: 100 pages, 75 claims

These are the measurements on the committed corpus. The settings were chosen on them, and
`make eval` reproduces the headline table, so an install can be checked against it. They ran
without the CPU and GPU caps.

**Method.** Each of the 75 claims is embedded once and searched once per `k` with `LIMIT k`, as
above. All 75 claims cite one page. Evidence recall is counted over the 66 claims whose gold set is
made of text units. Latency: 5 timed passes per claim (375 samples per `k`). Machine: i9-13900H,
15 GB, RTX 4060 Laptop GPU, MariaDB 11.8.9 in Docker, Python 3.12.

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

**The headline run again, and without the GPU.** The same evaluation was repeated four times on
the default ingest, twice as it is and twice with the GPU hidden
(`CUDA_VISIBLE_DEVICES="" wikilense eval --name ...`), so that the query is embedded on the CPU. A
fifth row is the evaluation in a fresh clone whose chunks were embedded on the CPU too.

| run | query embedded on | SQL p50 / p95 at k=10 | query embedding p50 / p95 | recall |
|---|---|---|---|---|
| `final_ef_100` | GPU | 0.51 / 0.94 ms | 4.58 / 4.70 ms | the table above |
| `final_ef_100_rerun_gpu1` | GPU | 0.48 / 0.69 ms | 4.39 / 4.50 ms | identical |
| `final_ef_100_rerun_gpu2` | GPU | 0.56 / 1.16 ms | 4.46 / 4.61 ms | identical |
| `final_ef_100_rerun_cpu1` | CPU | 0.53 / 1.12 ms | 17.83 / 22.68 ms | identical |
| `final_ef_100_rerun_cpu2` | CPU | 0.55 / 1.06 ms | 17.68 / 22.53 ms | identical |
| fresh clone, CPU ingest ([fresh_clone_run.md](../results/fresh_clone_run.md)) | CPU | 0.44 / 0.93 ms | 18.17 / 23.29 ms | identical |

Recall is the same in every row, at every `k`. The reruns search a different HNSW build than the
first run (the default ingest was rebuilt in between), and their 20-chunk hit lists equal the first
ones for 74 of 75 claims. Without a GPU the query embedding takes about 18 ms instead of 4.5 ms; the
SQL time does not depend on it.

**The same ingest, other statements** (article recall@10, evidence recall@20, SQL p50/p95 at k=10):

| run | statement | article@10 | evidence@20 | SQL p50 / p95 |
|---|---|---|---|---|
| `final_inline` | exact ranking (joined statement, no filter) | 74/75 | 65/66 | 2.24 / 2.57 ms |
| `final_ef_100` | index, ef_search 100 | 74/75 | 65/66 | 0.51 / 0.94 ms, hit lists = exact for 74/75 |
| `final_ef_20` | index, ef_search 20 | 74/75 | 65/66 | 0.45 / 0.70 ms, hit lists = exact for 65/75 |
| `final_rrf` | vector + full text, RRF | 74/75 | 64/66 | 9.11 / 12.84 ms |
| `final_oracle_titles` | inline, gold page as `titles` filter | 75/75 | 66/66 | 2.45 / 8.59 ms; evidence in the first chunk for 42/66, within 5 for 59/66 |
| `final_ef_100_after_restart` | index, ef_search 100, after a container restart (`docker restart`) | 74/75 | 65/66 | hit lists identical to before for 75/75 |

The one claim whose gold page never appears is claim 87976, a NOT ENOUGH INFO claim about a
by-election; its gold page is "Lincoln, England", and the exact ranking misses it too. On these
claims the hybrid does not beat the vector ranking and costs eighteen times more. With the page
known, 42 of 66 gold sets are already in the first chunk.

**Reproduce.**
- After `wikilense ingest`, `wikilense eval --name mine` writes `results/mine.json` and `.md` with
  the headline table for the current ingest. The counts match the table above up to the HNSW
  build: a fresh graph differs in a few chunk lists, while recall stays the same.
- `scripts/run_experiments.py all` reruns the groups in `results/SUMMARY.md` (with a GPU);
  `scale` runs only when named and needs `feverous_wikiv1.db` (path in
  `WIKILENSE_FEVEROUS_DB`).
- Every results file records the ingest parameters, the effective `ef_search`, the cache size, the
  index `M` and the chunk count it was measured on.
- `scripts/make_charts.py` redraws the charts from the JSON files.
- `scripts/compare_separate_index.py` and `scripts/measure_index_behaviour.py` rerun the comparison
  with a copy of the vectors and the measurements of [VECTOR_INDEX.md](VECTOR_INDEX.md). `builds`
  and `isolation` use scratch tables in the test database, and `cache` changes
  `mhnsw_max_cache_size` for the whole server while it runs, so run nothing else next to them.

## Parameters and why

The settings were chosen on the demo corpus, the 75-claim development set, and kept for the
scaled run; they were not re-measured at scale. The filter strategy has a scaled measurement.

| Parameter | Chosen | Evidence (details in results/SUMMARY.md) |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5`, 384 dimensions, 33M parameters, MIT licence | chosen for passage retrieval at this size; no second model was measured. 1,059 passages/s on the RTX 4060 and 78/s on the CPU for 100-word texts (a measurement in [DESIGN.md](DESIGN.md), not part of `results/`), so a machine without a GPU can ingest; normalised vectors, so cosine distance is `1 - dot` |
| Embedded text | `title > section path: chunk text` | on the exact ranking the prefix raises evidence recall@20 from 61 to 63 of 66 and leaves article recall@10 at 74 of 75 (section 5) |
| Chunk size | 240 words, overlap 1 unit, never across a section boundary | at equal retrieved text the 60-, 120- and 240-word chunkings are close: article recall 74/74/74 of 75 and evidence recall 54/57/55 of 66 at 1,200 words (60/63/60 at 2,400); 240 stores 4,598 vectors instead of 8,658 or 19,065 and a 13 MB graph instead of 17 or 29 (section 4) |
| Hatnotes | "Main article: ...", "See also: ...", "For other uses ..." are stored as sentences but kept out of chunks | 1,316 of 33,637 units; otherwise they form short chunks that rank high under section filters |
| Index `M` | 16 (server default 6) | on the same vectors at `ef_search=20`: article recall@10 67/74/73 of 75 and evidence recall@20 54/64/62 of 66 for M 6/16/32, against 74 and 63 exact; at `ef_search=100` M=16 returns the exact hit list for 69 of 75 claims (M=32: 73); build 2.9 s (M=6 1.1 s, M=32 9.1 s), graph 17 MB (section 3) |
| `mhnsw_ef_search` | 100 per query (server default 20) | on the final ingest hit lists identical to the exact ranking for 74 of 75 claims against 65 at 20, for 0.51 ms instead of 0.45 ms at k=10 (section 7) |
| Strategy with filters | `inline`; `overfetch` when latency matters more than a full result | on the 3,000 claims under the History filter, `inline` fills all 5 rows for every claim at 16.49 ms (S0) and 26.59 ms (S1), and `overfetch 10` takes 1.63 and 1.87 ms but comes back short for 2,472 and 2,425 of them ([results/scaled/timing/](../results/scaled/timing/)) |
| Index cache | 512 MB (`--mhnsw-max-cache-size`) | the graph must fit, with room: on S1 (a 156.0 MB graph) every count at 192 and 256 MB equals the one at 512 MB, while at 160 MB the joined statement is exact for 2,725 to 2,734 of 3,000 claims and takes 351.45 ms instead of 27.53, and at the 16 MB default the bare search finds a gold page for 2,658 claims instead of 2,933 ([results/scaled/run1/s1/cache_size.md](../results/scaled/run1/s1/cache_size.md)); on the demo corpus, 8 MB or less for its 13.0 MB graph left 57 to 67 of 75 lists exact (`results/cache_size.md`) |
