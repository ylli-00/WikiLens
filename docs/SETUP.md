# Setup: what `make setup` does, on which platforms, and what to do when it fails

The [README](../README.md) has the short version: `make setup`, then a query. This page has every
step, the three sets of packages, what was checked on which platform and how, and the known
failures.

## What you need

- Docker with the Compose plugin (`docker compose`).
- Python 3.10 or newer with the `venv` module. The two lock files need 3.12 or newer.
- GNU `make` and `git`.
- Disk: the Python environment is 1.4 GB with the CPU-only packages and 5.8 GB with the GPU build
  (`du -sh .venv`, measured on the author's laptop). The embedding model is another 134 MB in
  `~/.cache/huggingface`.
- A network connection for the first run (the MariaDB image, the packages, the model).

## What `make setup` runs

Every recipe is echoed, and `make -n setup` prints the commands without running them.

1. `python3 scripts/make_env.py`, only when there is no `.env`. It copies `.env.example` and puts
   two random passwords in it (the MariaDB root password and the application user's), readable by
   you only. To choose the passwords yourself, copy `.env.example` to `.env` and edit it before the
   first `make setup`. MariaDB's image sets the passwords once, when it creates the data volume, so
   a later change in `.env` does not change them.
2. `docker compose up -d --wait`: MariaDB 11.8 with `--mhnsw-max-cache-size=536870912` (512 MB),
   bound to `127.0.0.1:3306`, with the databases `wikilense` (the corpus) and `wikilense_test` (the
   test suite). The healthcheck waits until InnoDB is ready.
3. `python3 -m venv .venv` and `.venv/bin/pip install -r requirements-dev.txt -e .`. The package
   reads `sql/schema.sql` and `data/corpus/` from the checkout, so the editable install is required.
4. `.venv/bin/wikilense ingest`: creates the tables, parses the 100 committed pages, loads their
   categories and edit dates, embeds the chunks and loads everything. It took 21 s on the laptop's
   RTX 4060 (17 s of it embedding) and 230.0 s on its CPU (226.0 s embedding;
   [results/fresh_clone_run.md](../results/fresh_clone_run.md)).

Before step 1 `make setup` checks the Python version and stops with one line when it is too old.
Before step 3 it rebuilds `.venv` when the one in place does not work, for example after a system
upgrade removed the Python it was made with.

## Three sets of packages

| Command | Installs | Use it when |
|---|---|---|
| `make setup` | the version ranges of `requirements-dev.txt`, so the newest releases | the default; any Python from 3.10 |
| `make setup CPU=1` | `requirements-cpu.lock`: every version pinned, PyTorch without CUDA | there is no NVIDIA GPU, or you want a pinned environment; this is what CI installs |
| `make setup REQS=requirements.lock` | `requirements.lock`: the pinned GPU build | you want the exact environment `results/` was measured in (Linux x86_64 with an NVIDIA GPU) |

The default uses the ranges because they can adapt: the lock files cannot be installed on Python
3.10 or 3.11, on macOS 13 or on an Intel Mac, and the GPU lock file installs only on Linux. The
downside is that a new release of a dependency can break the default path before we notice. If
that happens, `make setup CPU=1` is the pinned way in.

Everything works without a GPU. Only the embedding is slower: the ingest above, and 18 ms for a
query instead of 4.6 ms (`results/final_ef_100_rerun_cpu1.md` and `_cpu2.md` against
`results/final_ef_100.md`). The SQL times do not change.

## Platforms

"Run" means we ran it. "Resolves" means a package installer finds a version of every package for
that platform; nobody ran it there. We have no Mac and no Windows machine.

| Platform | `make setup` (ranges) | `make setup CPU=1` (lock file) |
|---|---|---|
| Linux x86_64 | run on the author's laptop with Python 3.12 (the environment of `results/`) | run from a fresh clone with Python 3.14 on 2026-11-11; CI installs the same file with Python 3.12 on every push |
| Linux aarch64 | not checked | resolves |
| macOS 14 or newer, Apple Silicon | resolves for Python 3.10 to 3.14 | resolves for Python 3.12 to 3.14 |
| macOS 13, Apple Silicon | resolves, with PyTorch 2.11.0 | does not resolve |
| Intel Mac | resolves only with PyTorch 2.2.2 and Python up to 3.12; we treat it as not supported | does not resolve |
| Windows, WSL 2 | a Linux system: as Linux, not run | as Linux, not run |
| Windows, native | the Makefile needs a POSIX shell; run the four steps above by hand | the lock file resolves; not run |

The sources for the table, all looked up on 2026-11-11:

- PyTorch 2.14.0 has wheels for Linux x86_64 and aarch64, for Windows amd64 and for macOS on Apple
  Silicon from macOS 14 (file names on https://pypi.org/project/torch/2.14.0/#files and on
  https://download.pytorch.org/whl/cpu/torch/). There is none for Intel Macs, where the newest
  PyTorch is 2.2.2, and none for macOS 13, where it is 2.11.0.
- The `+cpu` builds exist for Linux and Windows only, so `requirements-cpu.lock` pins
  `torch==2.14.0` on macOS, where the ordinary wheel is the CPU build, and `torch==2.14.0+cpu`
  elsewhere.
- The lock files need Python 3.12: `numpy==2.5.3` has no release for 3.11 and `networkx==3.6.1`
  none for 3.10. They resolve for 3.12, 3.13 and 3.14.
- The image `mariadb:11.8.9` is published for linux/amd64 and linux/arm64, so Docker on Apple
  Silicon gets a native image (https://hub.docker.com/_/mariadb/tags?name=11.8.9; read with
  `docker manifest inspect mariadb:11.8.9` on 2026-11-11). `docker-compose.yml` pins the patch
  release because the measured index behaviour and the plans the tests pin are those of 11.8.9.
- On a Mac, `python3` may be the 3.9 of the Xcode tools. `make setup` then stops and says so;
  install a newer Python and run, for example, `make setup PYTHON=python3.12`.
- The Makefile uses `grep -E` with POSIX character classes and plain `sh`, and nothing specific to
  GNU tools. Of `make` itself it uses only `?=`, `:=`, `ifeq`, `ifneq` and `$(filter ...)`. It was
  not run with the `make` that macOS ships.

The two commands behind "resolves", for one platform (the others change the platform name):

```
pip install --dry-run --ignore-installed --only-binary=:all: --implementation cp \
    --python-version 3.12 --platform macosx_14_0_arm64 --target /tmp/unused -r requirements-cpu.lock

MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile requirements-cpu.lock --only-binary :all: \
    --python-platform aarch64-apple-darwin --python-version 3.12 --index-strategy unsafe-best-match
```

pip evaluates the `sys_platform` markers of the lock file for the machine it runs on, not for
`--platform`, so on Linux it would pick the Linux line; we ran pip with its marker environment set
to the target's. uv evaluates them for the target by itself.

## When it fails

| What you see | Why | What to do |
|---|---|---|
| `requirements-cpu.lock needs Python 3.12 or newer, and python3 is Python 3.x` | the lock files pin versions that need 3.12 | `make setup PYTHON=python3.12`, or plain `make setup` |
| `Access denied for user 'wikilense' ... (error 1045); the password in .env is not the one this MariaDB data volume was created with` | the volume is from an earlier setup with other passwords | put those passwords back in `.env`, or `docker compose down -v` (this deletes the database) and `make setup` again |
| `Bind for 127.0.0.1:3306 failed: port is already allocated` | another MariaDB or MySQL uses the port | set `WIKILENSE_DB_PORT` in `.env` to a free port and run `make setup` again |
| `The container name "/wikilense-mariadb" is already in use` | a second checkout of this repository has its container up | `docker compose down` in the other checkout |
| `permission denied while trying to connect to the Docker daemon socket` | your user is not in the `docker` group in this shell | `sg docker -c "make setup"`, or log in again |
| `ensurepip is not available` | a stock Debian or Ubuntu without the venv package | install `python3-venv`; the next `make setup` rebuilds the half-made `.venv` |

## Afterwards

`make test` runs the suite, `make eval` writes the headline table to `results/baseline.md` (ignored
by git), `make serve` starts the web page, `make examples` runs the ten examples of
`sql/examples.sql` through the `mariadb` client inside the container (no Python), `make experiments`
reruns the protocol and `make down` stops MariaDB and keeps the data. `docker compose down -v`
deletes the data volume.

`wikilense query` and the web page stop a search after 5 s (`WIKILENSE_STATEMENT_SECONDS` in `.env`,
0 for no limit) through MariaDB's `max_statement_time`, a guard against a vector search that keeps
searching when few rows pass its filters ([VECTOR_INDEX.md](VECTOR_INDEX.md)); `wikilense eval`
and the measurement scripts set no limit.

## Turn on the live fact-check

The web page can show a "Fact-check" button under a free-text search. It sends the statement and
its first 5 passages to Claude Haiku with the LLM experiment's prompt, and shows the verdict, the
model's reason and how often that verdict was right in the experiment
([LLM_EXTENSION.md](LLM_EXTENSION.md#14-on-the-web-page)). The button is off by default. The
"Copy the prompt" box under every search needs none of this and makes no call.

One setting turns it on: `WIKILENSE_FACTCHECK`, in `.env` or the environment, set to
`subscription` or `api`. Restart `make serve` after changing it. At start-up the server prints a
line beginning `fact-check:` that says whether the button is on, or why it cannot run.

- Through a Claude subscription, with `WIKILENSE_FACTCHECK=subscription`:
  - Needs the Claude Code CLI, logged in to a Claude subscription: run `claude` once and log in.
  - The server uses `claude` on the PATH; `WIKILENSE_CLAUDE_BIN=/path/to/claude` names another
    binary.
  - At start-up the server checks the login with `claude auth status`, which makes no model call.
    It refuses a CLI logged in with an API key, so this route never bills per token.
  - Each click counts against the subscription's usage limits.
- Through the Anthropic API, with `WIKILENSE_FACTCHECK=api`:
  - Install the SDK by hand with `.venv/bin/pip install -r requirements-llm.txt`; `make setup`
    does not install it.
  - Set `ANTHROPIC_API_KEY` in the environment or in `.env`. The environment wins.
  - Each click is billed per token. The server stops sending after US$1.00 at list price per
    start.
  - The key stays on the server. It never reaches the browser, a log line or a response.

A key in `.env` turns nothing on by itself: only `WIKILENSE_FACTCHECK` does.

The server still binds to 127.0.0.1 by default. The fact-check answers only requests addressed to
127.0.0.1 or localhost, so with `wikilense serve --host 0.0.0.0` the search is open to the network
but the button is not. `POST /api/factcheck` also needs the header `X-WikiLense-Factcheck: 1`,
which the page sends. Together, the two rules keep another website open in the same browser from
making calls.

Each click is one model call with no retry, not even after a rate limit on the API route, and it
times out after 60 seconds. Some calls turn the button off until the server restarts: one that
shows a sign of paid extra usage, and one that ends with a login, key or wrong-model error.

## Get the scaled database (S1)

`wikilense_s1` is the database behind the README's headline results ([DATA.md](DATA.md) describes
it): 7,686 Wikipedia pages, 132,359 chunks with their vectors, and the pages' categories and edit
dates. A [release file on GitHub](https://github.com/ylli-00/WikiLens/releases/tag/data-s1-v1)
holds the pages, the claims, the saved vectors and the page metadata. The load reads the saved
vectors instead of running the embedding model, so it needs neither the FEVEROUS Wikipedia file nor
a GPU.

| | |
|---|---|
| Download | `wikilense_s1_v1.tar.gz`, 300,308,768 bytes, SHA-256 `86ad18f01ad98bfaf23b800181e9f76dc409bf9b0ce20ef606bcaea0e93e032c` (the Makefile checks it) |
| Unpacked | 607,719,671 bytes in `data/scaled/wikilense_s1/` (git ignores it; the download is deleted after unpacking) |
| The database on disk | 1,375,712,270 bytes in MariaDB's data volume; the free space fell by 1,418,092,544 bytes during the load. With the unpacked folder kept, about 2.03 GB in all |
| Load time | 221.1 s for `make load-s1` on this laptop with the CPU capped at 50% (the ingest 215.35 s, of which 174.5 s inserts and the index build); the download time comes on top |

After `make setup`, from the repository root:

```
make fetch-s1     # download, check the size and SHA-256, unpack into data/scaled/wikilense_s1/
make load-s1      # load it into the database wikilense_s1 and check every count
```

`make load-s1` runs these steps in order (`make -n load-s1` prints them):

1. `scripts/scaled_release.py check-empty` stops when `wikilense_s1` already holds tables, because
   the ingest drops every table of its database. `make load-s1 REPLACE=1` replaces them.
2. `scripts/scaled_stage2.py create-dbs` creates `wikilense_s0` and `wikilense_s1` and grants them
   to the application user. `wikilense_s0` stays empty; only the full build below fills it.
3. `scaled_stage2.py ingest` loads the folder with the saved vectors and builds the vector index.
4. `scaled_stage2.py verify` compares the counts with `corpora.json`: 7,686 pages, 132,359 chunks,
   19,149,694 chunk words, 3,000 claims and 8,408 evidence ids (8,408 with a page, 7,877 with a
   text unit). A difference stops the load.
5. `scripts/scaled_stage3.py load` and `verify` write the categories and edit dates and check them
   against `page_meta_S1.jsonl`.

The vectors are the ones the reported results were measured with, but the HNSW graph is built anew
by each load, and graph builds differ a little. Twenty builds of these vectors found a gold page in
the top 5 for 2,922 to 2,937 of the 3,000 claims ([build_variance.md](../results/scaled/run1/s1/build_variance.md));
the reported run found 2,933. A test load from the release file found 2,934, and evidence recall@5
was 2,256 of 3,000 against the reported 2,253 ([the test load](../results/scaled/release_check/README.md)).

The web page needs no setting: once S1 is loaded, `make serve` offers it in the page's database
selector, next to the demo database. To query S1 from the command line, name it in
`WIKILENSE_DB_NAME`; a variable set in the shell takes precedence over `.env`:

```
WIKILENSE_DB_NAME=wikilense_s1 .venv/bin/wikilense query "The Eiffel Tower is in Paris."
WIKILENSE_DB_NAME=wikilense_s1 .venv/bin/wikilense eval --ef-search 100 --out /tmp/s1_eval --name s1
```

The database does not read `data/scaled/wikilense_s1/` after the load, so the folder can be
deleted; `make fetch-s1` downloads it again.

## Reproduce the scaled run

The scaled run ([EVALUATION.md](EVALUATION.md#the-scaled-run): 3,000 claims on the databases
`wikilense_s0` and `wikilense_s1`) cannot be rebuilt from the repository alone. It needs:

- the full FEVEROUS Wikipedia, `feverous_wikiv1.db` (53.5 GB, the one SQLite file inside the
  10.4 GB `feverous-wiki-pages-db.zip` of https://fever.ai/dataset/feverous.html), with its path in
  `WIKILENSE_FEVEROUS_DB`, and the FEVEROUS train and dev files in `data/feverous/`
  ([DATA.md](DATA.md));
- an NVIDIA GPU for the embedding, the network for the page metadata (the fetch for the 7,686
  pages of S1 took 294 minutes), and a working folder outside the repository for the corpus
  folders, the vectors and the large intermediate files;
- hours of machine time. Run one heavy step at a time: the measurements below are timings, and
  two steps at once spoil each other's times.

The commands, in order, from the repository root after `make setup` (`OUT` is the working folder):

```
export WIKILENSE_FEVEROUS_DB=/path/to/feverous_wikiv1.db
OUT=/path/to/working/folder

# 1. The claims, the distractor pages and the corpus folders S0 and S1 (docs/DATA.md)
.venv/bin/python scripts/scaled_build.py claims --out-dir $OUT
for step in distractors corpora report; do
  .venv/bin/python scripts/scaled_build.py $step --out-dir $OUT --claims $OUT/claims_3000.jsonl.tmp
done
.venv/bin/python scripts/scaled_check.py --out-dir $OUT --claims $OUT/claims_3000.jsonl.tmp
mv $OUT/claims_3000.jsonl.tmp $OUT/claims_3000.jsonl
sha256sum $OUT/claims_3000.jsonl data/scaled/claims_3000.jsonl     # the two must be equal

# 2. Two databases, the embedding (once, for S1, whose chunks include S0's) and the loads
.venv/bin/python scripts/scaled_stage2.py create-dbs
.venv/bin/python scripts/scaled_stage2.py embed --corpus-dir $OUT/S1 --vectors-dir $OUT/vectors \
    --corpora-json $OUT/corpora.json
for db in S0 S1; do
  low=$(echo $db | tr A-Z a-z)
  WIKILENSE_DB_NAME=wikilense_$low .venv/bin/python scripts/scaled_stage2.py ingest \
      --corpus-dir $OUT/$db --vectors-dir $OUT/vectors --out $OUT/ingest_$db.json
  WIKILENSE_DB_NAME=wikilense_$low .venv/bin/python scripts/scaled_stage2.py verify \
      --corpus-dir $OUT/$db --corpora-json $OUT/corpora.json
done

# 3. Categories and edit dates of the S1 pages (one file serves both databases)
.venv/bin/python scripts/scaled_stage3.py fetch --pages $OUT/S1/pages.jsonl --out $OUT/page_meta_S1.jsonl
for db in wikilense_s0 wikilense_s1; do
  .venv/bin/python scripts/scaled_stage3.py load --database $db --meta $OUT/page_meta_S1.jsonl
  .venv/bin/python scripts/scaled_stage3.py verify --database $db --meta $OUT/page_meta_S1.jsonl
done

# 4. Recall and latency, the evidence scores (run each per database: s0, s1)
R=results/scaled
WIKILENSE_DB_NAME=wikilense_s1 .venv/bin/wikilense eval --ef-search 100 --out $R/eval --name s1_full_run1
.venv/bin/python scripts/scaled_timing.py --database wikilense_s1 --out $R/timing/scale_k5_s1_run1.json
.venv/bin/python scripts/scaled_sentence_score.py --database wikilense_s1 \
    --claims data/scaled/claims_3000.jsonl --out-dir $OUT/sentence_score
.venv/bin/python scripts/scaled_page_coverage.py --database wikilense_s1 --out $R/page_coverage/page_coverage_s1.json

# 5. The measurements of docs/VECTOR_INDEX.md (filters, join order, statement forms, the copy of
#    the vectors, the index cache, graph builds, uncommitted rows)
.venv/bin/python scripts/scaled_filters.py --database wikilense_s1 --out $R/filters/filters_s1.json
.venv/bin/python scripts/measure_index_behaviour.py join_order --database wikilense_s1 --k 5 \
    --filters $R/filters/filters_s1.json --sample 200 --repeats 1 --no-turns --out $R/run1/s1
.venv/bin/python scripts/measure_index_behaviour.py forms --database wikilense_s1 --k 5 --repeats 1 --out $R/run1/s1
.venv/bin/python scripts/compare_separate_index.py --database wikilense_s1 --k 5 --repeats 1 \
    --filters $R/filters/filters_s1.json --out $R/run1/s1
.venv/bin/python scripts/measure_index_behaviour.py cache --database wikilense_s1 --k 5 --passes 2 \
    --cache-mb 256,192,160,128,64,16 --out $R/run1/s1
.venv/bin/python scripts/measure_index_behaviour.py builds --database wikilense_s1 --k 5 \
    --builds 1 --build-m 16 --out $R/build_timing/s1
.venv/bin/python scripts/scaled_uncommitted.py --database wikilense_s0 --out $R/run1/s0

# 6. The tables (no database)
.venv/bin/python scripts/scaled_breakdown.py --claims data/scaled/claims_3000.jsonl \
    --eval S0=$R/eval/s0_full_run1.json --eval S1=$R/eval/s1_full_run1.json \
    --sentences S0=$OUT/sentence_score/wikilense_s0/claims.jsonl \
    --sentences S1=$OUT/sentence_score/wikilense_s1/claims.jsonl --out $R/breakdown/breakdown.json
.venv/bin/python scripts/scaled_summary.py
```

- The `cache` step changes `mhnsw_max_cache_size` for the whole server and puts the configured
  value back at the end. `builds` and `scaled_uncommitted.py` work on scratch tables in the test
  database. Run none of them next to the test suite.
- `compare_separate_index.py` adds rows for hnswlib when it is importable. hnswlib is in no
  requirements file; to include it, install it into a folder of its own
  (`pip install --target <folder> hnswlib`) and put that folder on `PYTHONPATH` for this one command.
- A rerun writes into `run2/` (and so on) instead of `run1/`; `scripts/scaled_summary.py` then
  shows each run in its own column.
- The scaled times were measured with the CPU capped at 50% and the GPU at 2,100 MHz. On another
  machine, or without the caps, the times differ; the recall does not depend on them.
