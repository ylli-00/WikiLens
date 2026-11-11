# Setup: what `make setup` does, on which platforms, and what to do when it fails

The [README](../README.md) has the short version: `make setup`, then a query. This page has every
step, the three sets of packages, what was checked on which platform and how, and the failures we
know.

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

The default stays with the ranges on purpose. The lock files cannot be installed on Python 3.10
or 3.11, on macOS 13 or on an Intel Mac, and the GPU lock file only on Linux; the ranges can adapt.
The price is that a new release of a dependency can break the default path before we notice. If
that happens, `make setup CPU=1` is the pinned way in.

Without a GPU everything works. Only the embedding is slower: the ingest above, and 18 ms for a
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

What stands behind the table, all looked up on 2026-11-11:

- PyTorch 2.14.0 has wheels for Linux x86_64 and aarch64, for Windows amd64 and for macOS on Apple
  Silicon from macOS 14 (file names on https://pypi.org/project/torch/2.14.0/#files and on
  https://download.pytorch.org/whl/cpu/torch/). There is none for Intel Macs, where the newest
  PyTorch is 2.2.2, and none for macOS 13, where it is 2.11.0.
- The `+cpu` builds exist for Linux and Windows only. `requirements-cpu.lock` used to pin
  `torch==2.14.0+cpu` everywhere, so on a Mac pip stopped with "No matching distribution found for
  torch==2.14.0+cpu". The file now pins `torch==2.14.0` on macOS, where the ordinary wheel is the
  CPU build, and `torch==2.14.0+cpu` elsewhere.
- The lock files need Python 3.12: `numpy==2.5.3` has no release for 3.11 and `networkx==3.6.1`
  none for 3.10. They resolve for 3.12, 3.13 and 3.14.
- The image `mariadb:11.8.9` is published for linux/amd64 and linux/arm64, so Docker on Apple
  Silicon gets a native image (https://hub.docker.com/_/mariadb/tags?name=11.8.9; read with
  `docker manifest inspect mariadb:11.8.9` on 2026-11-11). `docker-compose.yml` pins the patch
  release because the measured index behaviour and the plans the tests pin are those of 11.8.9.
- On a Mac, `python3` may be the 3.9 of the Xcode tools. `make setup` then stops and says so;
  install a newer Python and run, for example, `make setup PYTHON=python3.12`.
- The Makefile uses `grep -E` with POSIX character classes and plain `sh`, nothing specific to GNU
  tools, and of `make` itself only `?=`, `:=`, `ifeq`, `ifneq` and `$(filter ...)`. Not run with
  the `make` that macOS ships.

The two commands behind "resolves", for one platform (the others change the platform name):

```
pip install --dry-run --ignore-installed --only-binary=:all: --implementation cp \
    --python-version 3.12 --platform macosx_14_0_arm64 --target /tmp/unused -r requirements-cpu.lock

MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile requirements-cpu.lock --only-binary :all: \
    --python-platform aarch64-apple-darwin --python-version 3.12 --index-strategy unsafe-best-match
```

Two things to know before repeating them. pip evaluates the `sys_platform` markers of the lock
file for the machine it runs on, not for `--platform`, so on Linux it would pick the Linux line; we
ran pip with its marker environment set to the target's. uv evaluates them for the target by
itself. And pip's dry run still downloads every wheel it resolves, about 200 MB per platform; uv
reads the metadata only.

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
