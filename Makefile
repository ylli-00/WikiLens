# WikiLense: one-command setup, tests, lint, evaluation, web page and experiments.
# Every recipe is echoed before it runs, so the commands can be copied into a shell.
#
#   make setup        .env (from .env.example on first run), MariaDB via Docker Compose,
#                     the .venv, the package, and the ingest of data/corpus/
#   make test         run the test suite (db tests need the container, slow tests the model)
#   make lint         ruff
#   make eval         recall and latency of the ingested corpus -> results/
#   make serve        the web query page on http://127.0.0.1:8000
#   make experiments  the whole experiment protocol (scripts/run_experiments.py all)
#   make down         stop the MariaDB container (the data volume is kept)
#
# Docker Compose is called as `docker compose`; when the docker group is only reachable through
# sg, run e.g. `sg docker -c "make setup"`. Override the tools with e.g. `make PYTHON=python3.12`.

PYTHON  ?= python3
VENV    ?= .venv
COMPOSE ?= docker compose
# `make setup CPU=1` installs the pinned CPU-only build (no CUDA wheels, about 2 GB smaller).
REQS    ?= requirements-dev.txt
ifeq ($(CPU),1)
REQS    := requirements-cpu.lock
endif

.PHONY: setup check-env test lint eval serve experiments down

# --- setup -----------------------------------------------------------------------------------

setup: .env check-env $(VENV)/bin/pip
	$(COMPOSE) up -d --wait
	$(VENV)/bin/pip install -r $(REQS) -e .
	$(VENV)/bin/wikilense ingest

# First run only: copy the example file, then stop so the placeholder passwords are replaced
# before MariaDB initialises its data volume with them. Run `make setup` again afterwards.
.env:
	cp .env.example .env
	@echo "Created .env from .env.example: set WIKILENSE_DB_PASSWORD and WIKILENSE_DB_ROOT_PASSWORD in .env, then run 'make setup' again."
	@exit 1

# Refuse to start MariaDB while .env still holds the placeholder passwords of .env.example:
# the first start initialises the data volume with whatever the file says.
check-env: .env
	@if grep -Eq '^WIKILENSE_DB_(ROOT_)?PASSWORD=change-me' .env; then \
		echo "WIKILENSE_DB_PASSWORD or WIKILENSE_DB_ROOT_PASSWORD in .env is still the placeholder of .env.example: set both, then run 'make setup' again."; \
		exit 1; \
	fi

# The target is pip, not the directory: a venv that failed half-way (no ensurepip on a stock
# Ubuntu without python3-venv) leaves the directory behind, and --clear starts it over.
$(VENV)/bin/pip:
	$(PYTHON) -m venv --clear $(VENV)

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

experiments:
	$(VENV)/bin/python scripts/run_experiments.py all

# --- teardown --------------------------------------------------------------------------------

down:
	$(COMPOSE) down
