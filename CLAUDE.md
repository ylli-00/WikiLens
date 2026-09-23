# CLAUDE.md - WikiLense

## Project

WikiLense is a project for the MariaDB student database projects 2026-09 (https://mariadb.org/bachelor_hackathon_2026-09/), area 4: "Vector search and RAG, natively". It is part of the Databases and Web Services course at Constructor University. All code is written with Claude Code, in Python.

Goal: retrieval over Wikipedia inside MariaDB. Chunk articles, embed the chunks, store them in a `VECTOR` column with a `VECTOR INDEX`, query with `VEC_DISTANCE_COSINE`, and combine the semantic search with ordinary SQL filters and joins in the same statement. The FEVEROUS fact-verification benchmark is the ground truth: its claims are the queries, and its gold evidence measures recall and latency. A later extension tests whether an LLM fact-checker does better with the retrieved passages.

## Current phase: the pipeline works end to end on the 100-page corpus (2026-09-22)

Everything the brief asks for is built, measured and pushed (github.com/ylli-00/WikiLens):
corpus selection, parsing, chunking, MariaDB schema with `VECTOR` and a cosine `VECTOR INDEX`,
ingest, hybrid queries (predicates, joins, an RRF full-text hybrid), CLI and web page, the
recall/latency harness, the experiment protocol (`scripts/run_experiments.py`, `results/SUMMARY.md`),
255 tests, CI, README. Chosen defaults with evidence: 240-word chunks, overlap 1, bge-small-en-v1.5
with the title prefix, index M=16, `mhnsw_ef_search` 100 per query, strategy `inline` for filtered
queries. The corpus is the 100 pages / 75 claims described below; the decisions of 2026-09-16/17 stand.

- **Owner's decision (2026-09-22):** the results in `results/` and the README are drafts. The claim set of the current corpus is a development set; the benchmark claims are picked in the next phase, and the README is rewritten with those results.
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

- MariaDB 11.8 LTS ships a native `VECTOR` type with HNSW indexing and SIMD acceleration; the index lives inside the transactional database.
- Take Wikipedia articles, chunk the text, embed the chunks, and store the vectors next to the article metadata with a `VECTOR INDEX`. Query with `VEC_DISTANCE_COSINE`.
- Constrain the semantic search with SQL in the same statement - only articles in a category, longer than a length, edited since a date, linked from another article - and join the results back to relational tables.
- Document with reasoning: chunk size and overlap, the embedding model and why, the index parameters.
- Build a ground-truth set so that "it works well" becomes a number: recall and latency.
- Vector docs: https://mariadb.com/docs/server/reference/sql-structure/vectors (`VECTOR` from 11.7.1; `VECTOR INDEX` takes `M` and `DISTANCE` = euclidean or cosine). Check syntax against these docs for the installed version.

## Grading (four areas, 0-10 each, equal weight)

| Area | What is checked |
|---|---|
| Documentation | README explains what the project is, what we found, the data-model reasoning, and what the database-native approach gave that a separate vector store would not |
| MariaDB depth | `VECTOR` column with an index over a real Wikipedia corpus, queried with `VEC_DISTANCE_COSINE`; semantic search and SQL predicates in the same statement, joined to relational tables |
| Execution | Recall and latency against our ground-truth set; chunk size, embedding model and index parameters stated with reasoning |
| Usability | An ingest that runs from the README, and a query interface a reader can try |

The brief's evaluation prompt, at the end of the brief page, also scores code quality. Each of these costs a point: hardcoded credentials, SQL built by string interpolation, debug mode left on, empty test files presented as tests, a schema that disagrees with the code. Setup as one documented command scores higher. Run that prompt on the repository before submitting.

## Submission rules

- Public GitHub repository, code on `main`, `README.md` at the root.
- The first line of `README.md` names the area and links back to the brief (wording from the brief page, checked 2026-09-17):
  `A project for area 4, "Vector search and RAG, natively" for the [MariaDB student database projects, 2026-09](https://mariadb.org/bachelor_hackathon_2026-09/).`
- Include schema documentation, setup instructions and how to reproduce the results.
- State the MariaDB version developed against (preview releases are fine).
- Data is either in the repository or fetched by a documented, repeatable step.

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
- Element ids: `[PAGE]_[type]_[numbers]`, type in `sentence`, `cell`, `header_cell`, `table_caption`, `item`. Sentences and table captions carry 1 number, items 2, and cells and header cells 3. The page is the text before the first underscore (`el.split("_")[0]`).
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

- **`feverous-wiki-pages-db.zip`:** 10.4 GB, containing one SQLite file, `feverous_wikiv1.db`, of 53.5 GB (a December 2020 version of Wikipedia, full articles). Not downloaded and not needed in the current phase.
- **`feverous-wiki-pages.zip`:** 9.9 GB, holding the pages as 544 JSONL shards (`FeverousWikiv1/wiki_NNN.jsonl`, 6.7–44.4 MB compressed each, 49.3 GB unpacked in total).
  - The shard numbers run from 000 to 610 with gaps, so take names from `--list`.
  - `scripts/fetch_wiki_shard.py` downloads a single shard with HTTP range requests, without downloading the whole archive, and checks its CRC.
- **Page format:** one JSON object per page, with `title`, `order` (element keys in page order), and `sentence_N`, `table_N`, `section_N`, `list_N`.
  - Sections have `value` and `level`; tables have `table`, `type` and sometimes `caption`; lists have `list` and `type`.
- **Checked on `wiki_000.jsonl` (9,996 pages):**
  - No other keys: no category, edit-date or revision fields.
  - No paragraph unit (1 of 514,984 sentences contains a newline), so chunks are built from consecutive sentences.
  - Hyperlinks are kept inline as `[[Target_title|shown text]]` (e.g. `[[Aare_(given_name)|Aare (given name)]]`): 755,380 links on 9,989 pages. Of these, 446,271 are in sentences, 141,596 in tables and 167,513 in lists. A link table has to read all three.
- The paper (Sec 7.7) says a date stamp was kept for each scraped page, but it is not in the shard's page objects.
- Reader classes in the official repo: `FeverousDB` (`src/feverous/database/feverous_db.py`) and `WikiPage` (`src/feverous/utils/wiki_page.py`).

## Open decisions

Current phase (provisional values are allowed, see "Current phase"):
- Embedding model, chunk size and overlap, index parameters.
- Tables: index them as chunks, or use only claims with a sentence-only evidence set (34,479 overall; 65 of the 75 phase-1 claims).
- How the README ingest gets the page subset: fetch the needed shard(s) with `scripts/fetch_wiki_shard.py` (`wiki_000` is a 44.4 MB download) and extract the pages, or commit the extracted subset (the 54 evidence pages are about 4.0 MB) to the repository.

Next phase:
- Source of category and edit-date metadata for the SQL filters.
- The final claim set and corpus size.

## Existing files

- `analysis/feverous/FEVEROUS_ANALYSIS.md` - analysis report, including the quality-assurance-subset findings. Its "Step 2", "tier" and "brief" refer to the original analysis request, not the MariaDB brief; Step 2 (the database) is parked, see "Current phase".
- `analysis/feverous/feverous_analysis.py` - analysis script; from the project root: `python3 analysis/feverous/feverous_analysis.py --out-md analysis/feverous/feverous_analysis_output.md`
- `analysis/feverous/feverous_claims.csv` - one row per claim: id, split, label, challenge, n_sets, min_pages, max_pages, has_sentence_only_set, modality_summary, set_pages, strict_multi_article, lenient_multi_article, has_sentence_only_multi_page_set. Multi-valued fields are `|`-separated.
- `analysis/feverous/feverous_analysis_output.md`, `analysis/feverous/feverous_stats.json` - script outputs
- `scripts/fetch_wiki_shard.py` - download one Wikipedia JSONL shard without the full archive (`--list` shows all 544)
- `data/feverous/` - raw downloads: `feverous_train_challenges.jsonl`, `feverous_dev_challenges.jsonl`, `wiki_pages/wiki_000.jsonl`

## Environment (set up 2026-09-17)

- Python 3.12 venv in `.venv` (`python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e .`). Run tools as `.venv/bin/python -m pytest`, `.venv/bin/ruff check .`, `.venv/bin/wikilense`.
- `Makefile` (added 2026-09-17): `make setup` (copies `.env` from `.env.example` on the first run and stops so the passwords get set, refuses to go on while they are still the placeholders (`make check-env`), then `docker compose up -d --wait`, the venv, `pip install -r requirements-dev.txt -e .`, `wikilense ingest`), `make test`, `make lint`, `make eval`, `make serve`, `make experiments` (`scripts/run_experiments.py all`), `make down`. Every recipe is echoed; `make -n <target>` shows it. With the docker group only reachable through `sg`: `sg docker -c "make setup"`.
- Pinned versions: `requirements.lock` is the GPU build of `.venv` (PyPI torch with CUDA libraries, what `results/` were produced with); `requirements-cpu.lock` is the CPU-only build (torch from the PyTorch CPU index, about 2 GB smaller) that `.github/workflows/ci.yml` installs from. Both were made with `pip freeze --exclude-editable`; the header of each file says how and which to use. `requirements.txt` / `requirements-dev.txt` keep the unpinned ranges.
- torch with CUDA works on the RTX 4060 (`torch.cuda.is_available()` is true); the embedding model downloads to `~/.cache/huggingface`.
- `docs/DESIGN.md` is the module contract: layout, data formats, parsing and chunking rules, schema, provisional settings and their reasons. Update it when a decision changes.
- Git: initialised on `main` on 2026-09-17, local identity `ylli <yrada@constructor.university>`. The owner creates the public GitHub repository and pushes. The README is written last (owner's decision, 2026-09-17); until then, `docs/DESIGN.md` holds the reasoning.

## Working rules

- Development happens on Ubuntu. Keep data and the MariaDB data directory on the Ubuntu (ext4) partition.
- Machine: i9-13900H, 16 GB RAM, RTX 4060 Laptop GPU with 8 GB.
- MariaDB must be 11.7.1 or newer for `VECTOR` (the brief targets 11.8 LTS).
  - Ubuntu's own `mariadb-server` package is 10.6 on 22.04 and 10.11 on 24.04, neither of which has `VECTOR`; it is 11.8 on 25.10 and 26.04 (packages.ubuntu.com, checked 2026-09-17).
  - Check `SELECT VERSION();` before writing schema code, and state the version in the README.
  - Install route (owner's decision, 2026-09-17): Docker Compose, `docker compose up -d` from the project root with the official `mariadb:11.8` image (11.8.9 at the time). Passwords live in `.env` (copy `.env.example`). The volume `mariadb-data` is on the ext4 root partition.
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
