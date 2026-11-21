# WikiLense: one-command setup, tests, lint, evaluation, web page and experiments.
# Every recipe is echoed before it runs, so the commands can be copied into a shell.
#
#   make setup        .env (written on the first run, with generated passwords), MariaDB via
#                     Docker Compose, the .venv, the package, and the ingest of data/corpus/;
#                     on Linux with PyTorch's CPU-only build
#   make setup GPU=1  the same with PyPI's PyTorch, which on Linux is the CUDA build
#   make setup CPU=1  the same with the pinned CPU-only packages (requirements-cpu.lock)
#   make test         run the test suite (db tests need the container, slow tests the model)
#   make lint         ruff
#   make eval         recall and latency of the ingested corpus -> results/
#   make serve        the web query page on http://127.0.0.1:8000
#   make examples     the ten examples of sql/examples.sql through the mariadb client (no Python)
#   make load-demo    the demo database from its dump, data/corpus/wikilense_demo.sql.gz, through
#                     the same client: with .env and MariaDB up, make examples needs no Python
#   make demo-dump    write that dump again from the ingested demo database
#   make experiments  the whole experiment protocol (scripts/run_experiments.py all)
#   make fetch-s1     download the scaled database's release file, check its SHA-256, unpack it
#                     into data/scaled/wikilense_s1/ (docs/SETUP.md, "Get the scaled database")
#   make load-s1      load that folder into the database wikilense_s1 (no GPU, no FEVEROUS file)
#   make down         stop the MariaDB container (the data volume is kept)
#
# Docker Compose is called as `docker compose`; when the docker group is only reachable through
# sg, run e.g. `sg docker -c "make setup"`. Override the tools with e.g. `make PYTHON=python3.12`.

PYTHON  ?= python3
VENV    ?= .venv
COMPOSE ?= docker compose
# What `make setup` installs. The default is the version ranges of requirements-dev.txt.
#   make setup GPU=1                   the same ranges with PyTorch from PyPI, which on Linux is
#                                      the CUDA build (for an NVIDIA GPU)
#   make setup CPU=1                   the pinned CPU-only build (no CUDA wheels, about 2 GB
#                                      smaller); Linux, Windows, and macOS 14+ on Apple Silicon
#   make setup REQS=requirements.lock  the pinned GPU build that results/ was measured with
#                                      (Linux x86_64 with an NVIDIA GPU)
REQS    ?= requirements-dev.txt
ifeq ($(CPU),1)
REQS    := requirements-cpu.lock
endif
# On Linux, PyPI's PyTorch is the CUDA build, so the ranges also look in PyTorch's CPU index. pip
# takes the newest version from either index, and a "+cpu" build sorts above PyPI's build of the
# same release. macOS wheels have no CUDA and get no extra index. A lock file pins its own build:
# "==2.14.0" would also match "2.14.0+cpu".
UNAME   ?= $(shell uname -s)
TORCH_INDEX :=
ifeq ($(UNAME),Linux)
ifneq ($(GPU),1)
ifeq ($(filter %.lock,$(REQS)),)
TORCH_INDEX := --extra-index-url https://download.pytorch.org/whl/cpu
endif
endif
endif
# The oldest Python that can install $(REQS): the package needs 3.10, the lock files pin
# versions that need 3.12 (numpy 2.5.3).
PY_MIN  := 3.10
ifneq ($(filter %.lock,$(REQS)),)
PY_MIN  := 3.12
endif
# The scaled database S1 as a release file: the 7,686 pages, the claims, the saved vectors and the
# page metadata, so that `make load-s1` needs neither the FEVEROUS file nor a GPU. The size and
# SHA-256 are those of the published file; `make fetch-s1` refuses any other bytes.
S1_URL    ?= https://github.com/ylli-00/WikiLens/releases/download/data-s1-v1/wikilense_s1_v1.tar.gz
S1_SHA256 := 86ad18f01ad98bfaf23b800181e9f76dc409bf9b0ce20ef606bcaea0e93e032c
S1_BYTES  := 300308768
S1_DIR    ?= data/scaled/wikilense_s1
# `make load-s1 S1_CHECK=1` loads into the scratch database wikilense_s1_check instead, to test
# the load on a machine whose wikilense_s1 must stay as it is.
S1_DB     := wikilense_s1
S1_FLAGS  :=
ifeq ($(S1_CHECK),1)
S1_DB     := wikilense_s1_check
S1_FLAGS  := --allow-check-database
endif
# The demo database as a dump. `make load-demo` loads it into the database the container was
# created with (MARIADB_DATABASE, from WIKILENSE_DB_NAME), the one `make examples` reads; the
# container's shell expands the name. `make load-demo DEMO_CHECK=1` loads into the scratch database
# wikilense_demo_check instead, to test the load without touching the demo database.
DEMO_DUMP := data/corpus/wikilense_demo.sql.gz
DEMO_DB   := $$MARIADB_DATABASE
ifeq ($(DEMO_CHECK),1)
DEMO_DB   := wikilense_demo_check
endif

.PHONY: setup check-python check-env venv test lint eval serve examples load-demo demo-dump \
	experiments fetch-s1 load-s1 down

# --- setup -----------------------------------------------------------------------------------

setup: check-python .env check-env venv
	$(COMPOSE) up -d --wait
	$(VENV)/bin/pip install -r $(REQS) -e . $(TORCH_INDEX)
	$(VENV)/bin/wikilense ingest

# Stop early, with the reason, when $(PYTHON) is too old: on a Mac without Homebrew, python3 is
# the 3.9 of the Xcode tools, and pip's own message about it comes late and is hard to read.
check-python:
	@$(PYTHON) -c 'import sys; need = tuple(int(part) for part in "$(PY_MIN)".split(".")); sys.exit(0 if sys.version_info[:2] >= need else "$(REQS) needs Python $(PY_MIN) or newer, and $(PYTHON) is Python %d.%d.%d. Install a newer Python and name it, for example: make setup PYTHON=python3.12" % sys.version_info[:3])'

# First run only: write .env from .env.example with two generated passwords, before MariaDB
# initialises its data volume with them. An existing .env is never touched; to choose the
# passwords yourself, copy .env.example to .env and edit it before the first `make setup`.
.env:
	$(PYTHON) scripts/make_env.py

# Refuse to start MariaDB while .env still holds the placeholder passwords of .env.example
# (a hand-made copy that was not edited): the first start initialises the data volume with
# whatever the file says.
check-env: .env
	@if grep -Eq "^[[:space:]]*(export[[:space:]]+)?WIKILENSE_DB_(ROOT_)?PASSWORD[[:space:]]*=[[:space:]]*[\"']?change-me(-too)?[\"']?[[:space:]]*\$$" .env; then \
		echo "WIKILENSE_DB_PASSWORD or WIKILENSE_DB_ROOT_PASSWORD in .env is still the placeholder of .env.example: set both, then run 'make setup' again."; \
		exit 1; \
	fi

# Make the venv when it is missing or does not work. The test is "its python can import pip",
# not "the directory exists": a venv that failed half-way (no ensurepip on a stock Ubuntu
# without python3-venv), or whose interpreter a system upgrade removed, leaves the directory
# behind, and --clear starts it over.
venv:
	@if ! $(VENV)/bin/python -c "import pip" 2>/dev/null; then \
		echo "$(PYTHON) -m venv --clear $(VENV)"; \
		$(PYTHON) -m venv --clear $(VENV); \
	fi

# --- development -----------------------------------------------------------------------------

test:
	$(VENV)/bin/python -m pytest -q

lint:
	$(VENV)/bin/ruff check .

# --- results ---------------------------------------------------------------------------------

eval:
	$(VENV)/bin/wikilense eval

serve:
	$(VENV)/bin/wikilense serve

# sql/examples.sql through the mariadb client that is inside the container: no Python, no model,
# no client on the host. The image keeps the application user's name and password in its own
# environment; the password reaches the client as MYSQL_PWD there and is on no command line.
examples:
	$(COMPOSE) exec -T mariadb sh -c 'MYSQL_PWD="$$MARIADB_PASSWORD" exec mariadb --table -u"$$MARIADB_USER" "$$MARIADB_DATABASE"' < sql/examples.sql

# The dump through the container's own gzip and mariadb client, as the application user: no
# Python, no model. The dump drops and re-creates the eleven tables, so the load first stops when
# the database already holds tables; REPLACE=1 skips that check. `gzip -t` refuses a damaged file
# before any table is touched, and the last statement prints counts to compare with the ingest's.
# DEMO_CHECK=1 first creates the scratch database as root and grants it to the user wikilense,
# the default of .env (the statements are fixed text, like those of scripts/scaled_stage2.py).
load-demo:
	$(COMPOSE) exec -T mariadb gzip -t < $(DEMO_DUMP)
ifeq ($(DEMO_CHECK),1)
	printf '%s\n' 'CREATE DATABASE IF NOT EXISTS wikilense_demo_check;' "GRANT ALL PRIVILEGES ON wikilense_demo_check.* TO 'wikilense'@'%';" | $(COMPOSE) exec -T mariadb sh -c 'MYSQL_PWD="$$MARIADB_ROOT_PASSWORD" exec mariadb -uroot'
endif
ifneq ($(REPLACE),1)
	$(COMPOSE) exec -T mariadb sh -c 'n=$$(MYSQL_PWD="$$MARIADB_PASSWORD" mariadb -u"$$MARIADB_USER" -N -e "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE()" "$(DEMO_DB)") || exit 1; if [ "$$n" != 0 ]; then echo "$(DEMO_DB) already holds $$n tables, and the load would drop and replace them. To replace them, run the load again with REPLACE=1."; exit 1; fi'
endif
	$(COMPOSE) exec -T mariadb sh -c 'gzip -dc | MYSQL_PWD="$$MARIADB_PASSWORD" exec mariadb -u"$$MARIADB_USER" "$(DEMO_DB)"' < $(DEMO_DUMP)
	$(COMPOSE) exec -T mariadb sh -c 'MYSQL_PWD="$$MARIADB_PASSWORD" exec mariadb --table -u"$$MARIADB_USER" -e "SELECT (SELECT COUNT(*) FROM page) AS pages, (SELECT COUNT(*) FROM chunk) AS chunks, (SELECT COUNT(*) FROM claim) AS claims" "$(DEMO_DB)"'

# Write the dump again from the ingested demo database (after `make setup`), with the container's
# mariadb-dump and gzip. The same database gives the same bytes: no dump date, and gzip -n stores
# no file name or time. The file is replaced only when the dump succeeded.
# tests/test_demo_dump.py pins its SHA-256 and checks its ingest record against data/corpus/.
demo-dump:
	$(COMPOSE) exec -T mariadb sh -c 'MYSQL_PWD="$$MARIADB_PASSWORD" mariadb-dump -u"$$MARIADB_USER" --single-transaction --hex-blob --skip-dump-date "$$MARIADB_DATABASE" > /tmp/demo.sql && gzip -9n < /tmp/demo.sql; status=$$?; rm -f /tmp/demo.sql; exit $$status' > $(DEMO_DUMP).part
	mv $(DEMO_DUMP).part $(DEMO_DUMP)

experiments:
	$(VENV)/bin/python scripts/run_experiments.py all

# --- the scaled database S1 ------------------------------------------------------------------

fetch-s1:
	$(VENV)/bin/python scripts/scaled_release.py fetch --url "$(S1_URL)" --sha256 $(S1_SHA256) --bytes $(S1_BYTES) --dest $(S1_DIR)

# The load drops every table of its database, so it first stops when that database already holds
# tables; REPLACE=1 skips the check. Then the databases, the ingest with the saved vectors (no
# model), the count check against corpora.json, and the categories and edit dates.
load-s1:
ifneq ($(REPLACE),1)
	$(VENV)/bin/python scripts/scaled_release.py check-empty --database $(S1_DB)
endif
	$(VENV)/bin/python scripts/scaled_stage2.py create-dbs $(S1_FLAGS)
	WIKILENSE_DB_NAME=$(S1_DB) $(VENV)/bin/python scripts/scaled_stage2.py ingest --corpus-dir $(S1_DIR)/S1 --vectors-dir $(S1_DIR)/vectors --out $(S1_DIR)/ingest_S1.json $(S1_FLAGS)
	WIKILENSE_DB_NAME=$(S1_DB) $(VENV)/bin/python scripts/scaled_stage2.py verify --corpus-dir $(S1_DIR)/S1 --corpora-json $(S1_DIR)/corpora.json $(S1_FLAGS)
	$(VENV)/bin/python scripts/scaled_stage3.py load --database $(S1_DB) --meta $(S1_DIR)/page_meta_S1.jsonl $(S1_FLAGS)
	$(VENV)/bin/python scripts/scaled_stage3.py verify --database $(S1_DB) --meta $(S1_DIR)/page_meta_S1.jsonl $(S1_FLAGS)

# --- teardown --------------------------------------------------------------------------------

down:
	$(COMPOSE) down
