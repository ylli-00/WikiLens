# CLAUDE.md - WikiLense

## Project

WikiLense is a project for the MariaDB student database projects 2026-09 (https://mariadb.org/bachelor_hackathon_2026-09/), area 4: "Vector search and RAG, natively". It is part of the Databases and Web Services course at Constructor University. All code is written with Claude Code, in Python.

Goal: retrieval over Wikipedia inside MariaDB. Chunk articles, embed the chunks, store them in a `VECTOR` column with a `VECTOR INDEX`, query with `VEC_DISTANCE_COSINE`, and combine the semantic search with ordinary SQL filters and joins in the same statement. The FEVEROUS fact-verification benchmark is the ground truth: its claims are the queries, and its gold evidence measures recall and latency. A later extension tests whether an LLM fact-checker does better with the retrieved passages.

## Current phase: graded 39/40 by the organizers; the three points they asked for are done (2026-11-11)

- **Evaluation of 27 October 2026 (the organizers ran the brief's evaluation prompt on `main`):
  39 of 40.** Documentation 10, MariaDB depth 10, execution 9, usability 10.
  - What they endorsed: recall and latency measured against the FEVEROUS benchmark's gold evidence
    "rather than self-written questions, which is the stronger ground truth"; the measured account
    of how MariaDB 11.8.9 uses the index; the explicit argument for doing this inside the database.
  - **So the question whether FEVEROUS may be the ground truth is settled: it may, and it stays.**
    Do not replace or supplement it with a hand-written question set, and keep the
    fact-verification framing in the README.
  - What they asked for, all done on 2026-11-11 (branch `fable-review-2026-11-11`):
    1. A short path for a reader who only wants to type one question: the README's first screen is
       now `make setup`, one query and its real output; the README went from 722 to about 330
       lines and the reference material moved to `docs/EXAMPLES.md`, `SETUP.md`, `DATA.md`,
       `VECTOR_INDEX.md` and `EVALUATION.md`.
    2. Where the approach is worse than a separate vector store: the README section "Where this
       approach loses", with a measurement (`scripts/compare_separate_index.py`,
       `results/separate_index.md`) and the documentation of both sides.
    3. `make setup CPU=1` failed on macOS (`torch==2.14.0+cpu` has no macOS wheel):
       `requirements-cpu.lock` now has per-platform PyTorch lines. Checked by resolution only;
       nobody ran it on a Mac (`docs/SETUP.md`, "Platforms").
  - Also measured on 2026-11-11, after a self-grade and a sceptical review of the new claims
    (`scripts/measure_index_behaviour.py`; `docs/VECTOR_INDEX.md` has the tables):
    - A predicate on `chunk` itself keeps the HNSW search going until `k` rows pass: not short,
      but approximate. With a join the result was the exact top 10 in every case measured.
    - `STRAIGHT_JOIN` stays. Left free, the optimizer starts from the filter's table under 8 of
      12 filters: much faster under a very selective filter, 4 to 5 times slower under one loose
      filter (`results/join_order.md`).
    - A REPEATABLE READ transaction that searches the index before and after another
      connection's commit leaves the index stale for every later reader, at any isolation
      level, until the next committed write or `FLUSH TABLES` (`results/isolation.md`, 10 of 10
      trials, three runs). Not reported to MariaDB: that is the owner's step.
    - Twenty builds of the graph at the chosen settings all find 74 of 75 gold pages
      (`results/build_variance.md`).
    - The graph must fit in `mhnsw_max_cache_size` (16 MB by default, 512 MB in our compose
      file): with 8 MB or less for the 13.0 MB graph the lists were exact for only 57 to 67 of 75
      claims, joined or not, and the search was slower (`results/cache_size.md`).
    - Corrected: "a derived table without DISTINCT loses the index" was wrong. The merged plan
      keeps the index; the DISTINCT prevents a chunk coming back once per link row.
    - When a timing is compared, time each statement alone and taking turns: alternating with a
      statement that reads thousands of rows made the other one 0.4 ms slower.
  - The core idea is fixed (owner's rule): the 75 claims, the 100-page corpus rule, 240-word
    chunks, overlap 1, bge-small-en-v1.5 with the title prefix, M=16, `mhnsw_ef_search` 100,
    strategy `inline`. A bug that changes results is fixed and the affected groups rerun; the
    settings are not re-chosen.

### Earlier: the pipeline works end to end on the 100-page corpus (2026-09-22)

Everything the brief asks for is built, measured and pushed (github.com/ylli-00/WikiLens):
corpus selection, parsing, chunking, MariaDB schema with `VECTOR` and a cosine `VECTOR INDEX`,
ingest, hybrid queries (predicates, joins, an RRF full-text hybrid), CLI and web page, the
recall/latency harness, the experiment protocol (`scripts/run_experiments.py`, `results/SUMMARY.md`),
the tests, CI, README. Chosen defaults with evidence: 240-word chunks, overlap 1, bge-small-en-v1.5
with the title prefix, index M=16, `mhnsw_ef_search` 100 per query, strategy `inline` for filtered
queries. The corpus is the 100 pages / 75 claims described below; the decisions of 2026-09-16/17 stand.

- **Status (2026-10-21):**
  - The brief asks for articles "from dumps.wikimedia.org or the MediaWiki API" and for "a small ground-truth set — questions whose correct source article you already know" (see "The MariaDB brief"). Whether FEVEROUS may serve as the ground truth was an open question to the organizers then; the evaluation of 27 October settled it (see above).
  - Meanwhile "Phase 0" runs on the 100-page corpus: check the vector scores, and add category and edit-date filters from the MediaWiki API (each page's December 2020 revision). Then add a transaction demo, rerun the experiments, and rewrite the README. The organizers give a preliminary grade by running the evaluation prompt.
  - Done on 2026-10-22: `data/corpus/page_meta.jsonl` (`scripts/fetch_page_meta.py`), the `category` and `page_category` tables, `page.revision_id` and `page.last_edited`, the filters `--category`, `--edited-since` and `--edited-before`, the experiment groups `uncommitted` and `planner`, and `scripts/make_charts.py`. docs/DESIGN.md, "Page metadata and the brief's four filters", has the measured outcomes.
- **Owner's decision (2026-09-22), superseded by the evaluation:** the results and the README were called drafts and the 75 claims a development set, to be replaced in a next phase. Since 27 October the 75-claim set and the settings are fixed; a larger claim set would be a new project phase, not a correction.
- **Next phase:** deeper analysis of FEVEROUS, pick exactly which claims to run, scale the corpus (the
  inline filtered statement's cost grows with the table and is the number to watch), then run and
  document the experiments; the LLM fact-checker extension. This includes measuring the share of gold
  evidence outside the lead section: the shards' `order` field is enough, and an annotation-context
  proxy is in `analysis/feverous/FEVEROUS_ANALYSIS.md` Appendix A.4.
- **Corpus rule kept from phase 1:** 75 FEVEROUS claims (10 dev, 65 train) whose evidence pages are all
  in `wiki_000.jsonl`, their 54 pages plus 46 link-ranked filler pages; `scripts/build_corpus.py`
  rebuilds it byte for byte (`data/corpus/MANIFEST.md`).
- **Not indexed:** tables and captions (27 of 114 evidence ids are cells); a documented limitation.

## The MariaDB brief

Area 4, verbatim from the page (checked 2026-10-21; the page was last modified 2026-09-30). An earlier summary here left out the dumps.wikimedia.org / MediaWiki API wording, although September session logs show it was on the page then too.

> The project. Build retrieval over Wikipedia. Take a set of articles from dumps.wikimedia.org or the MediaWiki API — one category, one subject area, or a few hundred articles you find interesting — chunk the text, embed the chunks, and store the vectors next to the article metadata with a VECTOR INDEX over them. Query with VEC_DISTANCE_COSINE.
> Then do the thing a dedicated vector database cannot: constrain the semantic search with ordinary SQL in the same statement — only articles in this category, longer than this, edited since that date, linked from that other article — and join the results back to your relational tables, transactionally consistent with the vectors.
> Note. The decisions worth documenting are chunk size and overlap, the embedding model and why you chose it, and the index parameters. Build a small ground-truth set — questions whose correct source article you already know — so that "it works well" becomes a number.

Area criteria, verbatim, with the grading criterion each counts toward:
- a `VECTOR` column with an index over a real Wikipedia corpus, queried with `VEC_DISTANCE_COSINE` (depth)
- semantic search and ordinary SQL predicates in the same statement, joined back to relational tables (depth)
- recall and latency against a ground-truth set you built, not adjectives (execution)
- chunk size, embedding model and index parameters stated, with the reasoning (execution)
- an ingest that runs from the README and a query interface a reader can try (usability)
- what the database-native approach gave you that a separate vector store would not (documentation)

- Vector docs: https://mariadb.com/docs/server/reference/sql-structure/vectors (`VECTOR` from 11.7.1; `VECTOR INDEX` takes `M` and `DISTANCE` = euclidean or cosine). Check syntax against these docs for the installed version.

## Grading (four areas, 0-10 each, equal weight)

| Area | What is checked |
|---|---|
| Documentation | README explains what the project is, what we found, the data-model reasoning, and what the database-native approach gave that a separate vector store would not |
| MariaDB depth | `VECTOR` column with an index over a real Wikipedia corpus, queried with `VEC_DISTANCE_COSINE`; semantic search and SQL predicates in the same statement, joined to relational tables |
| Execution | Recall and latency against our ground-truth set; chunk size, embedding model and index parameters stated with reasoning |
| Usability | An ingest that runs from the README, and a query interface a reader can try |

The brief's evaluation prompt is at the end of the brief page (prompt "last updated 2026-09-04"; checked 2026-10-21 and again 2026-11-11: unchanged). Run it on the repository before submitting; the organizers use it for grading.
- It also scores code quality. Each of these costs a point: hardcoded credentials, SQL built by string interpolation, debug mode left on, empty test files presented as tests, a schema that disagrees with the code.
- Setup as one documented command scores higher.
- Documentation levels: 5-6 needs what the project is and what it found, the data model, and install, run and reproduce (requirements, MariaDB setup, loading the data, version). 7-8 adds screenshots, charts or tables that show the result. 9-10 adds how it was made (architecture or method), the schema design reasoned about, and the team behind it.
- MariaDB depth: 9-10 means "taken past what the area asked for - into its limits, its failure modes, or its interaction with the rest of the server". The grader reads the area's linked MariaDB docs (Vectors) to judge how much of the feature was used.

## Submission rules

"What to hand in", from the brief page (checked 2026-10-21):
- Public GitHub repository, code on `main`, `README.md` at the root.
- The first line of `README.md` names the area and links back to the brief:
  `A project for area 4, "Vector search and RAG, natively" for the [MariaDB student database projects, 2026-09](https://mariadb.org/bachelor_hackathon_2026-09/).`
- MariaDB throughout. State the version developed against (preview releases are fine).
- Data is either in the repository or fetched by a documented, repeatable step.
- The README documents what the project is and what you found; the data model (the schema and the reasoning behind it); how to run or reproduce it, including requirements, MariaDB setup and loading the data; how it was made (the architecture if you built something, the method if you measured something) with the screenshots, charts or tables that show it; and the team behind it.
- The write-up's key results belong in the README itself. Full reports (`results/SUMMARY.md`, `docs/DESIGN.md`) are linked from it.

## Plan

1. Ground truth (next phase): claims picked after the deeper analysis (earlier idea: at most about 5,000); each claim is a query, its gold evidence is the answer.
2. Corpus (next phase; for now about 100 pages, see "Current phase"): the gold evidence pages, plus related distractors (linked pages; same-category pages only if a category source is chosen) and unrelated pages, all from the FEVEROUS Wikipedia data so the evidence text matches.
3. Pipeline: ingest, chunk, embed, store in MariaDB with a vector index, hybrid SQL queries, query interface.
4. Chunking keeps track of which sentence ids each chunk contains (a chunk-to-sentence table), so retrieved chunks can be joined to the gold evidence.
5. Metrics:
   - article recall@k (what the brief asks for)
   - evidence recall@k: a gold sentence is inside the top-k chunks
   - query latency
   - chunk sizes compared at the same amount of retrieved text (chunk size x k)
6. Extension: an LLM fact-checker run three ways - no retrieval, retrieved chunks, gold evidence only.

## Benchmark: FEVEROUS

- Dataset page: https://fever.ai/dataset/feverous.html (use these files; the Zenodo copy behind the DOI has no `challenge` field)
- Paper: https://arxiv.org/abs/2106.05707
- Code: https://github.com/Raldir/FEVEROUS
- Licence: Wikipedia article terms, otherwise CC BY-SA 3.0; code Apache-2.0.

Most of this section is reference for the next phase. For the current phase you mainly need "Annotation files" and "Wikipedia pages".

Why it was chosen: a second annotator, who did not see the writer's label or evidence, searched all of Wikipedia and picked the evidence (on the quality-check samples, verdict agreement between verifiers was kappa 0.65); evidence is sentence- or cell-level; full December 2020 articles ship with the data and evidence can come from any section except the references; NOT ENOUGH INFO is its own label and has evidence.

Not chosen:
- FEVER - intro sections only; NOT ENOUGH INFO claims have no evidence.
- HoVer - gold pages found by link matching, evidence always in the first paragraph, and NOT_SUPPORTED mixes false claims with details missing from the given sentences.
- KILT FEVER task - passage pointers moved to a 2019 snapshot by machine and never checked.
- EX-FEVER - gold evidence is article titles only.

### Annotation files

- JSONL, `train` and `dev` with labels; `test` is unlabelled. The first line is a header (the official reader skips it).
- Record fields: `id`, `claim`, `label` (SUPPORTS / REFUTES / NOT ENOUGH INFO), `evidence`, `annotator_operations`, `challenge`.
- `evidence` is a list of up to three alternative evidence sets. Each set has `content` (element ids) and `context` (element id -> title and section-heading elements). Only `content` counts as evidence.
- Element ids: `[PAGE]_[type]_[numbers]`, type in `sentence`, `cell`, `header_cell`, `table_caption`, `item`. Sentences and table captions carry 1 number, items 2, and cells and header cells 3. A title can contain underscores, so do not split on the first one: parse with `corpus.parse_element_id` (the regex is in docs/DESIGN.md, "Data contracts").
- In a page object:
  - Sentence ids are keys (`sentence_N`).
  - Cell and header-cell ids sit in `table_N.table[row][col].id` (each cell has `id`, `value`, `is_header`, `row_span`, `column_span`).
  - Item ids sit in `list_N.list[i].id`.
  - `table_N.caption` is a plain string with no id. That `table_caption_N` in the evidence means the caption of `table_N` has not been checked yet.
- Normalise page ids to Unicode NFC before joining: 2,011 of the 3,130 non-ASCII page ids in the annotation files are stored decomposed. The shard titles are already NFC (0 of 448 non-ASCII titles in `wiki_000.jsonl` are decomposed).
- The `challenge` tag was assigned by the annotator who verified the claim.

### Counts from the released files

| Measure | Value |
|---|---|
| Claims, train / dev | 71,291 / 7,890 |
| Labels, train (S / R / NEI) | 41,835 / 27,215 / 2,241 |
| Labels, dev (S / R / NEI) | 3,908 / 3,481 / 501 |
| Every evidence set on 2+ pages (strict) | 13,700 of 79,181 (17.30%) |
| At least one evidence set on 2+ pages | 14,041 (17.73%) |
| At least one sentence-only evidence set | 34,479 |
| Sentence-only evidence set on 2+ pages | 5,015 |

- The paper's dev "Sentence+Cells" count is 2,468; the files give 2,198.
- The 8,474 re-annotated agreement samples are not marked in the release, and their extra annotations are not released.

### Wikipedia pages

- **`feverous-wiki-pages-db.zip`:** 10.4 GB, containing one SQLite file, `feverous_wikiv1.db`, of 53.5 GB (a December 2020 version of Wikipedia, full articles).
  - Unpacked on 2026-10-21 onto the Windows partition: `/mnt/windows/Users/yllir/Desktop/Projects/WikiLense/wikilense-linux/data/feverous/feverous_wikiv1.db`. Owner's decision: it stays there and is not copied to Linux (see "Working rules").
  - Schema `wiki (id PRIMARY KEY, data json)`; `data` is the same page object as in the shards. Open it read-only (`sqlite3.connect("file:...?mode=ro", uri=True)`).
  - Ids and text are stored NFD (decomposed). The first 2,000 pages of `wiki_000` equal the shard pages after NFC-normalising both. Look titles up in NFD and normalise what comes back to NFC.
  - Random 100-page sample, plus three pages read by hand (2026-10-21): no category, edit-date or revision data anywhere. The pages end before "See also" content, references and the category box. All 100 pages have links, and 85.7% of the 2,550 link targets are pages in the DB.
  - Category and edit date can come from the MediaWiki API: the revision live at the end of December 2020 matches the FEVEROUS text (Albert Einstein: 466 of 484 sentence probes, against 260 for today's page). Its timestamp is the edit date, and its `[[Category:...]]` lines are the categories. Pages edited during December 2020 need a text match to pick the revision, and moved pages need the move log.
- **`feverous-wiki-pages.zip`:** 9.9 GB, holding the pages as 544 JSONL shards (`FeverousWikiv1/wiki_NNN.jsonl`, 6.7–44.4 MB compressed each, 49.3 GB unpacked in total).
  - The shard numbers run from 000 to 610 with gaps, so take names from `--list`.
  - `scripts/fetch_wiki_shard.py` downloads a single shard with HTTP range requests, without downloading the whole archive, and checks its CRC.
- **Page format:** one JSON object per page, with `title`, `order` (element keys in page order), and `sentence_N`, `table_N`, `section_N`, `list_N`.
  - Sections have `value` and `level`; tables have `table`, `type` and sometimes `caption`; lists have `list` and `type`.
- **Checked on `wiki_000.jsonl` (9,996 pages):**
  - No other keys: no category, edit-date or revision fields.
  - No paragraph unit (1 of 514,984 sentences contains a newline), so chunks are built from consecutive sentences.
  - Hyperlinks are kept inline as `[[Target_title|shown text]]` (e.g. `[[Aare_(given_name)|Aare (given name)]]`): 755,380 links on 9,989 pages. Of these, 446,271 are in sentences, 141,596 in tables and 167,513 in lists. A link table has to read all three.
- The paper (Sec 7.7) says the pages were scraped from Wikipedia with Scrapy and a date stamp was kept for each page, but the stamp is in neither the shards nor the DB.
- Reader classes in the official repo: `FeverousDB` (`src/feverous/database/feverous_db.py`) and `WikiPage` (`src/feverous/utils/wiki_page.py`).

## Decisions (none open)

Everything this section used to list as open is decided; docs/DESIGN.md has the reasons.
- Embedding model, chunk size and overlap, index parameters: bge-small-en-v1.5 with the title
  prefix, 240 words, overlap 1, M=16, `mhnsw_ef_search` 100 (fixed, see "Current phase").
- Tables: not indexed; 27 of the 114 evidence ids are table cells, a documented limitation.
- How the README ingest gets the pages: the extracted subset is committed in `data/corpus/`.
- Category and edit-date metadata: the MediaWiki API, each page's December 2020 revision
  (`scripts/fetch_page_meta.py`, committed as `data/corpus/page_meta.jsonl`).
- FEVEROUS as the ground truth: settled by the evaluation of 27 October 2026.
- Decided on 2026-11-11 (reasons in docs/DESIGN.md, "Review pass (2026-11-11)"):
  `requires-python` stays `>=3.10`; `make setup` keeps installing the version ranges by default
  and the lock files stay opt-in; `make setup` generates the `.env` passwords.

## Existing files

- `analysis/feverous/FEVEROUS_ANALYSIS.md` - analysis report, including the quality-assurance-subset findings. Its "Step 2", "tier" and "brief" refer to the original analysis request, not the MariaDB brief.
- `analysis/feverous/feverous_analysis.py` - analysis script; from the project root: `python3 analysis/feverous/feverous_analysis.py --out-md analysis/feverous/feverous_analysis_output.md`
- `analysis/feverous/feverous_claims.csv` - one row per claim: id, split, label, challenge, n_sets, min_pages, max_pages, has_sentence_only_set, modality_summary, set_pages, strict_multi_article, lenient_multi_article, has_sentence_only_multi_page_set. Multi-valued fields are `|`-separated.
- `analysis/feverous/feverous_analysis_output.md`, `analysis/feverous/feverous_stats.json` - script outputs
- `scripts/fetch_wiki_shard.py` - download one Wikipedia JSONL shard without the full archive (`--list` shows all 544)
- `scripts/fetch_page_meta.py` - each corpus page's December 2020 revision id, timestamp and categories from the MediaWiki API -> `data/corpus/page_meta.jsonl` and `PAGE_META.md` (committed)
- `scripts/make_charts.py` - the README charts (`docs/img/*.svg`) from `results/*.json`
- `scripts/make_env.py` - writes `.env` with two generated passwords on the first `make setup`
- `scripts/compare_separate_index.py` - MariaDB against an in-process copy of the vectors (`results/separate_index.md`); hnswlib is optional and must stay out of every requirements and lock file
- `scripts/measure_index_behaviour.py` - five measured behaviours of the vector index: `forms` (`results/statement_forms.md`), `join_order` (`results/join_order.md`), `builds` (`results/build_variance.md`), `cache` (`results/cache_size.md`) and `isolation` (`results/isolation.md`); `builds` and `isolation` use scratch tables in the test database and `cache` changes `mhnsw_max_cache_size` for the whole server while it runs: never run them next to pytest or another measurement
- `data/corpus/NOTICE.md` - licence and attribution of the committed Wikipedia text and FEVEROUS claims (CC BY-SA 3.0)
- `data/feverous/` - raw downloads: `feverous_train_challenges.jsonl`, `feverous_dev_challenges.jsonl`, `wiki_pages/wiki_000.jsonl`

## Environment (set up 2026-09-17; the operating system changed on 2026-10-26)

- Python 3.12 venv in `.venv` (`python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e .`). Run tools as `.venv/bin/python -m pytest`, `.venv/bin/ruff check .`, `.venv/bin/wikilense`.
  - The laptop was upgraded from Ubuntu 24.04 to 26.04 on 2026-10-26. `/usr/bin/python3` is now 3.14 and the system's Python 3.12 is gone, which broke `.venv`. Since 2026-11-11 `.venv/bin/python3` points at a user-level CPython 3.12.16 installed with uv (`~/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/`); the packages are unchanged (`pip freeze` equals `requirements.lock`). If that directory is removed, `make setup` rebuilds the venv with the system Python.
- `Makefile` (added 2026-09-17, changed 2026-11-11): `make setup` checks the Python version (3.10 for the ranges, 3.12 for a lock file), writes `.env` with two generated passwords when there is none (`scripts/make_env.py`; a hand-made `.env` that still holds the placeholders is refused by `make check-env`), then `docker compose up -d --wait`, the venv (rebuilt when its interpreter no longer works), `pip install -r requirements-dev.txt -e .`, `wikilense ingest`. `make setup CPU=1` installs `requirements-cpu.lock`, `make setup REQS=requirements.lock` the GPU lock. Also `make test`, `make lint`, `make eval` (writes the git-ignored `results/baseline.*`), `make serve`, `make examples` (`sql/examples.sql` through the container's `mariadb` client), `make experiments` (`scripts/run_experiments.py all`), `make down`. Every recipe is echoed; `make -n <target>` shows it. With the docker group only reachable through `sg`: `sg docker -c "make setup"`.
- Pinned versions: `requirements.lock` is the GPU build of `.venv` (PyPI torch with CUDA libraries, what `results/` were produced with; Linux x86_64 only); `requirements-cpu.lock` is the CPU-only build that `.github/workflows/ci.yml` installs from (1.4 GB installed instead of 5.8 GB). Both were made with `pip freeze --exclude-editable`; since 2026-11-11 the CPU file carries environment markers (`torch==2.14.0+cpu` off macOS, `torch==2.14.0` on macOS, colorama on Windows). Both need Python 3.12 or newer. `requirements.txt` / `requirements-dev.txt` keep the unpinned ranges.
- Timings on this laptop drift by tens of percent within a minute (CPU governor `powersave`): compare numbers inside one run, interleave the things compared, and run a headline measurement twice.
- torch with CUDA works on the RTX 4060 (`torch.cuda.is_available()` is true); the embedding model downloads to `~/.cache/huggingface`.
- `docs/DESIGN.md` is the module contract: layout, data formats, parsing and chunking rules, schema, provisional settings and their reasons. Update it when a decision changes.
- `docs/ARCHITECTURE.md` (2026-09-23) is the beginner's guide to how the code is built: the pipeline, key ideas, every file, one query end to end, the database, tests, and what is left for the grading. Keep it true when a file or a flow changes.
- Git: initialised on `main` on 2026-09-17, local identity `ylli <yrada@constructor.university>`. Remote `origin` is github.com/ylli-00/WikiLens. `docs/DESIGN.md` holds the reasoning behind every decision.
- The README must stay short (the evaluators' request; `tests/test_docs.py` caps it at 330 lines and checks its first line, its first-screen output, every link and anchor, and the stated number of tests). Put reference material in `docs/`.

## Working rules

- Development happens on Ubuntu. Keep data and the MariaDB data directory on the Ubuntu (ext4) partition.
  - Exception (owner's decision, 2026-10-21): the 53.5 GB `feverous_wikiv1.db` stays on the Windows partition and is not copied to Linux.
  - `/etc/fstab` mounts that partition at boot at `/mnt/windows` (ntfs-3g, `nofail`; it mounts read-only by itself if Windows is hibernated).
- Wikipedia API (Wikimedia Robot Policy, edited 2026-03-16; checked 2026-10-21): it is free.
  - Unauthenticated Action API: one request at a time and below 5 requests per second. With a logged-in account: 3 at a time and 10 per second.
  - Send a User-Agent with the repository URL as contact. Never put the owner's email in it.
  - Send `Accept-Encoding: gzip`, respect `Retry-After` on HTTP 429, use `maxlag`, and wait 5 s after any request that took over 1 s.
- Machine: i9-13900H, 16 GB RAM, RTX 4060 Laptop GPU with 8 GB.
- MariaDB must be 11.7.1 or newer for `VECTOR` (the brief targets 11.8 LTS).
  - Ubuntu's own `mariadb-server` package is 10.6 on 22.04 and 10.11 on 24.04, neither of which has `VECTOR`; it is 11.8 on 25.10 and 26.04 (packages.ubuntu.com, checked 2026-09-17).
  - Check `SELECT VERSION();` before writing schema code, and state the version in the README.
  - Install route (owner's decision, 2026-09-17): Docker Compose, `docker compose up -d` from the project root with the official `mariadb` image, pinned to `mariadb:11.8.9` since 2026-11-11 (the measured index behaviour and the pinned plans are those of 11.8.9; move the pin only together with a rerun of the tests and of `scripts/measure_index_behaviour.py`). Passwords live in `.env` (copy `.env.example`). The volume `mariadb-data` is on the ext4 root partition.
  - Two databases: `wikilense` (the corpus) and `wikilense_test` (the test suite only, created by `sql/docker-init/01-test-database.sh` on first start). Tests must never touch `wikilense`.
- Keep database credentials in environment variables or an untracked `.env`; never commit them.
- Compute every statistic from the files and report the actual output. Write "not computed" or "not found" instead of estimating.
- Do not modify raw data in `data/feverous/`.
- Do not commit the raw files in `data/feverous/` to Git. `feverous_train_challenges.jsonl` (177.6 MB) and `wiki_000.jsonl` (181.6 MB) are over GitHub's 100 MB per-file limit, so the README has to fetch data with a documented step.
- Ask before downloads over 5 GB and before deleting files.
- Explain results in simple language, conclusion first - this is the owner's first databases course.

## Token efficiency and agents

- Be token efficient. Read only the files and line ranges a task needs, keep replies short, and do not re-read files already in context or repeat what was already established.
- Do not spawn subagents or workflows by default. Most tasks here (editing a file, writing a script, answering a question, checking a value) are done faster and cheaper directly.
- Spawn an agent only when it clearly pays off: a search across many files where only the conclusion is needed, several independent tasks that can genuinely run in parallel, or work too large for one context. Say in one line why an agent is being used.
- Never spawn agents for a one-file edit, a lookup with a known location, a question that can be answered from context, or to "double-check" something already verified.
- Prefer one targeted command over several broad ones, and one agent over many when an agent is justified.
