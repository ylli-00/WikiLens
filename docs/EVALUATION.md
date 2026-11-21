# Evaluation: method, every result table, and the reasons for each parameter

The [README](../README.md) has the headline numbers. This page has the method in full and every
table behind them, for two sets of measurements:

- **The scaled run** is the reported result: 3,000 FEVEROUS claims searched on two corpora. Its
  files are in [results/scaled/](../results/scaled/), every table in
  [results/scaled/SUMMARY.md](../results/scaled/SUMMARY.md).
- **The demo corpus** is the 100 committed pages and their 75 claims: what `make setup` loads and
  `make eval` measures, and the development set the settings were chosen on. Its files are in
  [results/](../results/), with [results/SUMMARY.md](../results/SUMMARY.md) as the overview.

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
  differ by at most 1.3% (p50 and p95), so the tables give run 1;
  [SUMMARY.md](../results/scaled/SUMMARY.md) has both.
- **Machine setting.** Every scaled measurement ran alone, with the laptop's CPU capped at 50% of
  its speed and the GPU at 2,100 MHz, so its times compare with each other but not with the
  uncapped 100-page times below.

### Recall and latency of the index search

From [results/scaled/eval/](../results/scaled/eval/) (`s0_full_run1` and `s1_full_run1`):

| k | S0 article recall | S0 evidence recall | S0 SQL p50 | S0 SQL p95 | S1 article recall | S1 evidence recall | S1 SQL p50 | S1 SQL p95 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.965 (2,895) | 0.446 (1,337) | 1.28 ms | 1.74 ms | 0.938 (2,814) | 0.434 (1,303) | 1.86 ms | 2.96 ms |
| 3 | 0.979 (2,937) | 0.703 (2,108) | 1.12 ms | 1.57 ms | 0.970 (2,911) | 0.681 (2,042) | 1.52 ms | 2.55 ms |
| **5** | **0.984 (2,951)** | **0.780 (2,341)** | **1.11 ms** | 1.52 ms | **0.978 (2,933)** | **0.751 (2,253)** | **1.43 ms** | 2.32 ms |
| 10 | 0.986 (2,959) | 0.853 (2,558) | 1.21 ms | 1.71 ms | 0.982 (2,947) | 0.826 (2,479) | 1.66 ms | 2.69 ms |
| 20 | 0.990 (2,971) | 0.893 (2,678) | 1.41 ms | 2.06 ms | 0.986 (2,957) | 0.870 (2,609) | 2.17 ms | 3.36 ms |

What it means:

- **The first chunk is usually from a gold page, but the full evidence needs several chunks.** At
  k = 1 the page is found for 0.965 of the claims on S0 and the evidence for 0.446; at k = 5 for
  0.984 and 0.780.
- **Tripling the chunk table with distractor pages costs little.** From S0 to S1, article
  recall@5 falls from 0.984 to 0.978, evidence recall@5 from 0.780 to 0.751, and the search takes
  1.43 instead of 1.11 ms.
- **Embedding the claim takes longer than the search:** 6.07 to 6.15 ms at the median on the GPU
  (both databases and runs).

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

Run 1 is shown; run 2 is within 1.5% (p50 and p95).

- The exact search costs about 11 times the index search on S0 and 15 times on S1, and its cost
  grows with the table: a join in the statement makes MariaDB rank exactly
  ([VECTOR_INDEX.md](VECTOR_INDEX.md#what-we-measured)). The `inline` History filter (the filter
  inside that joined statement) costs as much as the exact search, while `overfetch 10` (the 50
  nearest chunks, filtered afterwards) is as fast as the index search but often comes back short.

**The approximate index costs almost nothing in recall.** It finds a gold page for 6 fewer claims
than the exact search on S0 and 10 fewer on S1:

| database | top 5 identical to the exact top 5 | gold page in the top 5, HNSW | gold page in the top 5, exact |
|---|---|---|---|
| S0 | 2,962 (run 2: 2,963) | 2,956 | 2,962 |
| S1 | 2,911 | 2,933 | 2,943 |

Both runs give these gold-page counts (the JSON field `article_hits_at_10` holds k = 5); the main
harness gives 2,951 and 2,933.

### The six claim groups

From [results/scaled/breakdown/breakdown.md](../results/scaled/breakdown/breakdown.md), which has
every `k`. At k = 5, with the
[FEVEROUS-style score](#a-feverous-style-evidence-score-and-page-coverage) (the best 5 sentences
of the top 10 chunks must hold a complete gold set):

| group (claims) | S0 article@5 | S0 evidence@5 | S0 FEVEROUS-style | S1 article@5 | S1 evidence@5 | S1 FEVEROUS-style |
|---|---|---|---|---|---|---|
| SUPPORTS, one article (700) | 0.994 | 0.960 | 0.750 | 0.994 | 0.956 | 0.739 |
| SUPPORTS, two or more (300) | 1.000 | 0.633 | 0.280 | 1.000 | 0.553 | 0.260 |
| REFUTES, one article (700) | 0.993 | 0.956 | 0.769 | 0.986 | 0.940 | 0.747 |
| REFUTES, two or more (300) | 0.993 | 0.327 | 0.090 | 0.990 | 0.287 | 0.080 |
| NOT ENOUGH INFO, one article (700) | 0.950 | 0.866 | 0.616 | 0.936 | 0.833 | 0.593 |
| NOT ENOUGH INFO, two or more (300) | 0.990 | 0.353 | 0.107 | 0.983 | 0.303 | 0.103 |
| all 3,000 claims | 0.984 | 0.780 | 0.546 | 0.978 | 0.751 | 0.529 |

- **Claims on two or more pages are harder.** The full evidence is in the top 5 chunks for 0.83 to
  0.96 of the claims on one page and 0.29 to 0.63 of those on two or more (S0 and S1 together): a
  chunk holds text of one page only, so k = 1 never covers such a claim, and the second page has
  to rank high too. Under the FEVEROUS-style score such claims are rarely complete (0.080 to 0.280).
- **A gold page is found for almost every claim in every group.** NOT ENOUGH INFO on one page is
  the hardest group for the page (0.950 and 0.936).

### FEVEROUS's challenge tags

The annotator who verified a FEVEROUS claim tagged its main difficulty. The pick did not balance on
the tag, and the search does worse on some labels than on others
([the label check](../results/scaled/breakdown/breakdown.md#label-check-and-equal-weight-figures)),
so each tag is given with equal weights: the mean of its three per-label rates, at k = 5 (the
same file has the figures as measured and the intervals):

| challenge tag, claims (S / R / NEI), equal weight | S0 article@5 | S0 evidence@5 | S0 FEVEROUS-style | S1 article@5 | S1 evidence@5 | S1 FEVEROUS-style |
|---|---|---|---|---|---|---|
| Other, 1,906 (602 / 709 / 595) | 0.990 | 0.887 | 0.668 | 0.986 | 0.868 | 0.651 |
| Multi-hop Reasoning, 709 (319 / 169 / 221) | 0.997 | 0.482 | 0.197 | 0.992 | 0.435 | 0.186 |
| Entity Disambiguation, 151 (34 / 46 / 71) | 0.941 | 0.722 | 0.475 | 0.905 | 0.672 | 0.447 |
| Search terms not in claim, 131 (18 / 15 / 98) | 0.930 | 0.809 | 0.609 | 0.930 | 0.770 | 0.583 |
| Numerical Reasoning, 71 (6 / 51 / 14) | 0.938 | 0.818 | 0.584 | 0.938 | 0.799 | 0.584 |
| Combining Tables and Text, 32 (21 / 10 / 1) | 1.000 | 0.984 | 0.552 | 1.000 | 0.984 | 0.552 |

Multi-hop Reasoning against Other within each page count (S1, evidence@5, equal weights; "level"
where the 95% interval of the difference includes zero,
[intervals in breakdown.md](../results/scaled/breakdown/breakdown.md#step-2-s1-multi-hop-reasoning-against-other)):

| claims | Multi-hop Reasoning: claims | evidence@5 | Other: claims | evidence@5 | difference |
|---|---|---|---|---|---|
| on one page | 112 | 0.865 | 1,704 | 0.927 | −0.062, larger than chance |
| on two or more pages | 597 | 0.374 | 202 | 0.389 | −0.016, level |

What it means:

- **Ambiguous names and missing search terms make the page hard to find.** On S1 these two tags
  have the lowest article recall@5: Entity Disambiguation 0.905, Search terms not in claim 0.930.
- **Multi-hop Reasoning is hard mostly because of its page count.** It finds the page (0.992 on
  S1) but not the whole evidence (0.435), and 597 of its 709 claims need two or more pages, where
  it is level with Other.
- **Two tags are too small to judge:** Numerical Reasoning and Combining Tables and Text (whose 32
  claims all have a text-only evidence set too, which is why they are in the pick).

### Reweighted to FEVEROUS's mix

Our pick is balanced by label and by the number of pages; FEVEROUS is not ([DATA.md](DATA.md)
compares the two mixes), so an overall number is also given reweighted.

- **Weights.** From [data/scaled/feverous_claims.csv](../data/scaled/feverous_claims.csv),
  one row per FEVEROUS train and dev claim (79,181). A claim is in group (label, one article) when
  every evidence set lies on one page (`max_pages` = 1), and in (label, two or more) when every set
  spans two or more pages (`min_pages` >= 2); a claim with sets of both kinds is in no group.
  Weighting "all" counts every claim of the six groups (78,840); weighting "text" counts only
  those with an evidence set of sentences only or of list items only (34,480).
- **Reweighted value** = the sum over the six groups of (weight x our rate for that group).
- The weighting cannot reproduce one detail of the pick: our "one article" also requires every set
  to name the same page, which the table does not record. Its "two or more" counts equal the
  pick's.

| group | FEVEROUS, all | weight | FEVEROUS, text | weight |
|---|---|---|---|---|
| SUPPORTS, one article | 33,981 | 0.4310 | 13,988 | 0.4057 |
| SUPPORTS, two or more | 11,529 | 0.1462 | 4,067 | 0.1180 |
| REFUTES, one article | 29,100 | 0.3691 | 14,453 | 0.4192 |
| REFUTES, two or more | 1,494 | 0.0189 | 584 | 0.0169 |
| NOT ENOUGH INFO, one article | 2,059 | 0.0261 | 1,125 | 0.0326 |
| NOT ENOUGH INFO, two or more | 677 | 0.0086 | 263 | 0.0076 |

| at k = 5 | S0 article | S0 evidence | S0 FEVEROUS-style | S1 article | S1 evidence | S1 FEVEROUS-style |
|---|---|---|---|---|---|---|
| as measured | 0.984 | 0.780 | 0.546 | 0.978 | 0.751 | 0.529 |
| reweighted, all | 0.993 | 0.891 | 0.667 | 0.990 | 0.870 | 0.650 |
| reweighted, text | 0.993 | 0.901 | 0.682 | 0.989 | 0.881 | 0.665 |

On FEVEROUS's mix the numbers are higher, because that mix is mostly SUPPORTS and REFUTES claims
on one page, the easiest groups.

### A FEVEROUS-style evidence score, and page coverage

Our evidence recall is more generous than FEVEROUS's rule. It counts a claim when a complete gold
set is inside the top k chunks, and five chunks hold up to 1,200 words. FEVEROUS lets a system
return at most 5 sentences and 25 table cells or list items, and counts a claim only when a
complete gold set is among them. `scripts/scaled_sentence_score.py` applies that rule, for
sentence-only gold sets, to the text units of each claim's top 10 chunks, each embedded once with
the chunks' model and ranked against the claim
([results/scaled/sentence_score/](../results/scaled/sentence_score/)).

| database | complete set in the top 10 chunks (ours) | 5 + 25, ranked by embedding (FEVEROUS-style) | 5 + 25 in chunk order |
|---|---|---|---|
| S0 | 0.853 (2,558) | **0.546 (1,637)** | 0.413 (1,239) |
| S1 | 0.825 (2,476) | **0.529 (1,588)** | 0.397 (1,190) |

- **Five sentences hold a complete gold set far less often than ten chunks do:** 0.546 against
  0.853 on S0, 0.529 against 0.825 on S1.
- **Ranking the sentences by the embedding matters:** kept in chunk order they score 0.413 and
  0.397.

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
| [FEVEROUS paper](https://arxiv.org/abs/2106.05707), Table 4: dev set, its own retriever, 5.4 million pages | 0.39 | 0.49 | 0.58 | 0.69 | - |

- **Two pages already hold nearly all the gold pages that ten find:** on average 0.923 of a
  claim's gold pages on S0 and 0.895 on S1, against 0.967 and 0.955 at ten pages.
- **The paper's row is not comparable.** Our corpus is nearly three orders of magnitude smaller
  (7,686 pages for S1), and our claims are the text-only ones, so even the FEVEROUS-style score
  is easier here than in the paper.

### Reproduce

The scaled run needs the full FEVEROUS Wikipedia (`feverous_wikiv1.db`, 53.5 GB), a GPU and
several hours. [SETUP.md](SETUP.md#reproduce-the-scaled-run) has the commands in order.

## The demo corpus: 100 pages, 75 claims

`make eval` reproduces the headline table, so an install can be checked against it. These
measurements ran without the CPU and GPU caps.

**Method.** Each claim is embedded once and searched once per `k` with `LIMIT k`, as above. All
75 claims cite one page. Evidence recall is counted over the 66 claims whose gold set is made of
text units. Latency: 5 timed passes per claim (375 samples per `k`). Machine: i9-13900H,
15 GB, RTX 4060 Laptop GPU, MariaDB 11.8.9 in Docker, Python 3.12.

**Headline** (`results/final_ef_100.md`: 240-word chunks, M=16, `ef_search=100`, strategy `none`):

| k | article recall | evidence recall | unit coverage | SQL p50 | SQL p95 |
|---|---|---|---|---|---|
| 1 | 63/75 (0.84) | 39/66 (0.59) | 0.60 | 0.50 ms | 0.86 ms |
| 3 | 74/75 (0.99) | 52/66 (0.79) | 0.81 | 0.45 ms | 0.78 ms |
| 5 | 74/75 (0.99) | 55/66 (0.83) | 0.85 | 0.45 ms | 0.79 ms |
| 10 | 74/75 (0.99) | 60/66 (0.91) | 0.92 | 0.51 ms | 0.94 ms |
| 20 | 74/75 (0.99) | 65/66 (0.98) | 0.98 | 0.64 ms | 1.10 ms |

![Article and evidence recall at k](img/recall_at_k.svg)

**The headline run again, and without the GPU.** The same evaluation ran four more times on the
default ingest, two of them with the GPU hidden (`CUDA_VISIBLE_DEVICES="" wikilense eval --name ...`)
to embed the query on the CPU. The last row is a fresh clone whose chunks were embedded on the CPU.

| run | query embedded on | SQL p50 / p95 at k=10 | query embedding p50 / p95 |
|---|---|---|---|
| `final_ef_100` | GPU | 0.51 / 0.94 ms | 4.58 / 4.70 ms |
| `final_ef_100_rerun_gpu1` | GPU | 0.48 / 0.69 ms | 4.39 / 4.50 ms |
| `final_ef_100_rerun_gpu2` | GPU | 0.56 / 1.16 ms | 4.46 / 4.61 ms |
| `final_ef_100_rerun_cpu1` | CPU | 0.53 / 1.12 ms | 17.83 / 22.68 ms |
| `final_ef_100_rerun_cpu2` | CPU | 0.55 / 1.06 ms | 17.68 / 22.53 ms |
| fresh clone, CPU ingest ([fresh_clone_run.md](../results/fresh_clone_run.md)) | CPU | 0.44 / 0.93 ms | 18.17 / 23.29 ms |

Recall is the same in every row, at every `k`. The reruns search a different HNSW build than the
first run (the default ingest was rebuilt in between), and their 20-chunk hit lists equal the first
ones for 74 of 75 claims. Without a GPU the query embedding takes about 18 ms instead of 4.5 ms; the
SQL time does not depend on it.

**The same ingest, other statements** (SQL time at k=10, hit lists of 20 chunks):

| run | statement | article@10 | evidence@20 | SQL p50 / p95 | hit lists equal to the exact ranking |
|---|---|---|---|---|---|
| `final_inline` | exact ranking (joined statement, no filter) | 74/75 | 65/66 | 2.24 / 2.57 ms | the reference |
| `final_ef_100` | index, ef_search 100 | 74/75 | 65/66 | 0.51 / 0.94 ms | 74/75 |
| `final_ef_20` | index, ef_search 20 | 74/75 | 65/66 | 0.45 / 0.70 ms | 65/75 |
| `final_rrf` | vector + full text, RRF | 74/75 | 64/66 | 9.11 / 12.84 ms | - |
| `final_oracle_titles` | inline, gold page as `titles` filter | 75/75 | 66/66 | 2.45 / 8.59 ms | - |

What it means:

- **Every statement finds the same pages,** except the oracle, which is given them. The one claim
  whose gold page never appears, 87976, is missed by the exact ranking too.
- **The index matches the exact ranking's recall at a fraction of its time:** 0.51 against
  2.24 ms. At ef_search 20 its hit lists are exact for only 65 of 75 claims.
- **The RRF hybrid costs eighteen times more and adds nothing on these claims:** 9.11 against
  0.51 ms.
- **Knowing the page does not put the evidence first:** given the gold page, it is in the first
  chunk for 42 of 66 claims and within 5 for 59.

**Reproduce.**
- After `wikilense ingest`, `wikilense eval --name mine` writes `results/mine.json` and `.md` with
  the headline table for the current ingest, equal to the one above up to the HNSW build.
- `scripts/run_experiments.py all` reruns the groups in `results/SUMMARY.md` (with a GPU);
  `scale` runs only when named and needs `feverous_wikiv1.db` (path in
  `WIKILENSE_FEVEROUS_DB`).
- Every results file records the ingest parameters, the effective `ef_search`, the cache size, the
  index `M` and the chunk count it was measured on.
- `scripts/make_charts.py` redraws the charts from the JSON files.
- `scripts/compare_separate_index.py` and `scripts/measure_index_behaviour.py` rerun the comparison
  with a copy of the vectors and the measurements of [VECTOR_INDEX.md](VECTOR_INDEX.md). Run
  nothing else next to `builds` and `isolation` (scratch tables in the test database) or `cache`
  (it changes `mhnsw_max_cache_size` for the whole server).

## Parameters and why

The settings were chosen on the demo corpus, the 75-claim development set, and kept for the
scaled run; they were not re-measured at scale. The filter strategy has a scaled measurement.

| Parameter | Chosen | Evidence (details in results/SUMMARY.md) |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5`, 384 dimensions, 33M parameters, MIT licence | chosen for passage retrieval at this size; no second model was measured. 1,059 passages/s on the RTX 4060 and 78/s on the CPU for 100-word texts (a measurement in [DESIGN.md](DESIGN.md), not part of `results/`), so a machine without a GPU can ingest; normalised vectors, so cosine distance is `1 - dot` |
| Embedded text | `title > section path: chunk text` | on the exact ranking the prefix raises evidence recall@20 from 61 to 63 of 66 and leaves article recall@10 at 74 of 75 (section 5) |
| Chunk size | 240 words, overlap 1 unit, never across a section boundary | about the same recall as 60- and 120-word chunks at equal retrieved text, with the fewest vectors and the smallest graph (the first table below) |
| Hatnotes | "Main article: ...", "See also: ...", "For other uses ..." are stored as sentences but kept out of chunks | 1,316 of 33,637 units; otherwise they form short chunks that rank high under section filters |
| Index `M` | 16 (server default 6) | the only M that reaches the exact ranking's recall at `ef_search` 20; M=32 gives a few more exact hit lists at 100 but takes longer to build (the second table below) |
| `mhnsw_ef_search` | 100 per query (server default 20) | on the final ingest hit lists identical to the exact ranking for 74 of 75 claims against 65 at 20, for 0.51 ms instead of 0.45 ms at k=10 (section 7) |
| Strategy with filters | `inline`; `overfetch` when latency matters more than a full result | complete or fast, not both: on the 3,000 claims under the History filter `inline` always fills 5 rows, `overfetch 10` is as fast as the bare search but mostly comes back short ([Fast and slow statements](#fast-and-slow-statements)) |
| Index cache | 512 MB (`--mhnsw-max-cache-size`) | the graph must fit, with room: 512 MB holds S1's 156.0 MB graph three times over, and at 160 MB the search is already less exact and much slower ([VECTOR_INDEX.md](VECTOR_INDEX.md#the-index-cache-below-and-above-the-graph)) |

**Chunk size** ([results/SUMMARY.md](../results/SUMMARY.md), section 4: one ingest per size,
`M=16`, `ef_search` 100, k chosen so that every size retrieves the same nominal number of words):

| chunk size | vectors | graph | article recall of 75, at 1,200 and 2,400 words | evidence recall of 66, at 1,200 words | at 2,400 words |
|---|---|---|---|---|---|
| 60 words | 19,065 | 29 MB | 74 | 54 | 60 |
| 120 words | 8,658 | 17 MB | 74 | 57 | 63 |
| **240 words** | **4,598** | **13 MB** | 74 | 55 | 60 |

**Index `M`** (section 3: the same vectors, the index rebuilt for each `M`):

| M | article recall@10 of 75, `ef_search` 20 | evidence recall@20 of 66, `ef_search` 20 | hit lists equal to exact of 75, `ef_search` 100 | build |
|---|---|---|---|---|
| 6 | 67 | 54 | 38 | 1.1 s |
| **16** | **74** | **64** | 69 | 2.9 s |
| 32 | 73 | 62 | 73 | 9.1 s |
| exact ranking | 74 | 63 | 75 | - |

### The embedding model's training data

The English BGE models were trained on Wikipedia text and fine-tuned on FEVER, an older
fact-checking set built on Wikipedia: "We first finetune on unsupervised datasets including
datasets like Wikipedia, … We then further fine-tune on supervised datasets including NLI (Gao et
al., 2021c), FEVER (Thorne et al., 2018), …" (Xiao et al., C-Pack,
[arXiv 2309.07597v2](https://arxiv.org/pdf/2309.07597v2), Appendix C; FEVEROUS is not listed, the
paper does not name v1.5, and the
[model card of `bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) names no
training data). So retrieval here runs on familiar ground: the model knows the task and the text,
which is not the same as having seen our test claims, and recall on text it was not trained on is
probably lower than the recall above. The database comparisons (index against exact search,
filters, statement forms, cache, timings) are unaffected, because both sides of each comparison
use the same embeddings.
