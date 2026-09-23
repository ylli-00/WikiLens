#!/usr/bin/env python3
"""Run the WikiLense experiment protocol on the main database and write the files in results/.

Every evaluation goes through ``wikilense.evaluate.evaluate`` and ``write_results`` (one JSON and
one Markdown file per run, named as in the protocol); every ingest through
``wikilense.ingest.run_ingest`` with ``dataclasses.replace(settings, chunk_max_words=...,
chunk_overlap_units=...)`` and an explicit ``use_prefix``. Nothing is inherited from
the environment or from an earlier run: every group names its configuration (chunk size,
overlap, index M, prefix) as a :class:`Configuration` and either ingests it or verifies, from
``ingest_meta`` and ``SHOW CREATE TABLE chunk``, that the database holds exactly it; every
evaluation passes ``mhnsw_ef_search``, the strategy, the ks and the overfetch factor explicitly,
and :func:`run_eval` refuses to run when the database does not hold the configuration it was
given. The evaluations run one after the other, never two at once, so the latency samples never
overlap.

``sql/schema.sql`` builds the vector index with M=16 (``db.VECTOR_INDEX_M``), so a configuration
with another M rebuilds the index right after the ingest: ``ALTER TABLE chunk DROP INDEX <name>``
then ``ALTER TABLE chunk ADD VECTOR INDEX <name> (embedding) M=<m> DISTANCE=cosine``, where
``<name>`` is the index name that ``SHOW CREATE TABLE chunk`` reports (checked against
``IDENTIFIER_RE``) and ``<m>`` one of ``INDEX_M_VALUES``, followed by ``ANALYZE TABLE chunk`` as
after an ingest. The ANALYZE is not cosmetic: measured on 2026-09-22 (MariaDB 11.8.9), a rebuilt
index that had not been analysed made every vector query after the next server restart re-read
the whole hidden index table (``Handler_read_rnd_next`` about the number of chunks per query,
p50 2.84 ms instead of 0.88 ms on 4,598 chunks, 4.1 ms instead of 0.56 ms on 8,658), and
``ANALYZE TABLE chunk`` restored the normal cost; an index built by ``CREATE TABLE`` at ingest,
which is analysed, did not show this. All three statements are timed, the size of the hidden
InnoDB tablespace that holds the HNSW graph is recorded, and ``ingest_meta.index_m`` is updated
so that it keeps describing the index that is in place.

Every result file records, besides the harness's own fields (the effective ``mhnsw_ef_search``,
the global ``mhnsw_max_cache_size``, the index M and DISTANCE from ``SHOW CREATE TABLE chunk``
and the chunk count), the configuration of the group, the hatnote count of the ingest
(``ingest_meta.n_units_hatnote``), the sizes of the chunk table and of the graph tablespace and
the chunk word statistics (``parameters`` in the JSON, the "Parameters" table in the Markdown).

Configurations: ``OLD`` = 120-word chunks, overlap 1, M=6, prefix on (phases 1 and 2);
``FINAL`` = 240 words, overlap 1, M=16, prefix on (the defaults of ``wikilense.config`` and
``sql/schema.sql``). ks 1, 3, 5, 10, 20 (plus the equal-text ks of the chunk-size group) and
5 repeats.

Groups (``all`` runs them in this order; each can be rerun on its own):

  stability   fresh ingest of OLD; three runs at ef 20 with the 512 MB HNSW cache, a container
              restart and a fourth run, then SET GLOBAL mhnsw_max_cache_size = 16 MB and three
              more runs (512 MB restored); per-claim hit-list comparison
              -> stability_512mb_run1..3, stability_512mb_after_restart, stability_16mb_run1..3,
              stability_comparison.json/.md
  ef_search   OLD: mhnsw_ef_search 20 / 50 / 100 / 200 / 400 with strategy none (ef_<n>), the
              inline statement without filters, which is the exact ranking, at ef 20 and 100
              (inline_ef_<n>), and the rrf hybrid at ef 100 (rrf_120_m6)
  index_m     OLD vectors: the index rebuilt with M=6, 16 and 32 (timed, tablespace sizes), each
              evaluated at ef 20 and 100 -> m<M>_ef_<n>, index_m_rebuild.json/.md; M=6 is
              rebuilt once more at the end so that OLD stays in place
  chunk_size  fresh ingests with 60, 120 and 240 words (overlap 1, M=16, prefix on), evaluated
              at ef 100 with the ks that give the same amount of retrieved text
              -> chunk60_ef_100, chunk120_ef_100, chunk240_ef_100
  prefix      fresh ingests of 120 words (M=16) with and without the "title > section path: "
              prefix, each with an approximate run at ef 100 and the exact inline ranking
              -> prefix_on_ef_100, prefix_on_inline, prefix_off_ef_100, prefix_off_inline
  filters     FINAL: min_words 1000 and heading LIKE '%History%' with the inline, overfetch 10
              and overfetch 50 strategies at ef 100 -> strategy_<strategy>_<filter>
  final       fresh ingest of FINAL; final_ef_100 (the headline run: strategy none, ef 100),
              final_ef_20, final_inline (exact), final_rrf (rrf, ef 100, overfetch 10, the claim
              text as query_text), final_oracle_titles (inline with the claim's gold pages as the
              titles filter), then a container restart and final_ef_100_after_restart /
              final_ef_20_after_restart with the before/after hit-list comparison
              -> final_restart_comparison.json/.md
  restore     fresh ingest of FINAL (what ``wikilense ingest`` produces), the cache checked at
              512 MB and ingest_meta verified -> restore_state.json/.md
  summary     results/SUMMARY.md from the JSON files (no database access)
  all         every group above, in that order

Every ingest and every index rebuild is appended to results/ingest_runs.json/.md.

Root-level SQL (``SET GLOBAL mhnsw_max_cache_size``, the hidden tablespace size) and the
container restart go through ``docker exec`` / ``docker compose`` on the container named in
docker-compose.yml. The root password is read from .env (``WIKILENSE_DB_ROOT_PASSWORD``) and
handed to docker through the environment, never on a command line, and never printed. When the
docker group is only reachable through ``sg``:

    sg docker -c ".venv/bin/python scripts/run_experiments.py all"
    sg docker -c ".venv/bin/python scripts/run_experiments.py ef_search index_m"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymysql
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wikilense import db
from wikilense.cli import make_embedder
from wikilense.config import (
    DEFAULT_CHUNK_MAX_WORDS,
    DEFAULT_CHUNK_OVERLAP_UNITS,
    DEFAULT_EF_SEARCH,
    DEFAULT_ENV_FILE,
    Settings,
    load_settings,
)
from wikilense.embedding import Embedder
from wikilense.evaluate import (
    DEFAULT_RESULTS_DIR,
    ClaimFilters,
    EvalResult,
    evaluate,
    load_ground_truth,
    oracle_title_filters,
    read_ingest_meta,
    write_results,
)
from wikilense.ingest import INDEX_DISTANCE, run_ingest
from wikilense.search import Filters, search
from wikilense.wikitext import HATNOTE_RE

log = logging.getLogger("experiments")

CONTAINER = "wikilense-mariadb"
ROOT_PASSWORD_VARIABLE = "WIKILENSE_DB_ROOT_PASSWORD"

CACHE_512MB = 536870912
CACHE_16MB = 16777216
#: SET GLOBAL statements as fixed literals (root only; run through docker exec).
SET_CACHE_SQL: dict[int, str] = {
    CACHE_512MB: "SET GLOBAL mhnsw_max_cache_size = 536870912",
    CACHE_16MB: "SET GLOBAL mhnsw_max_cache_size = 16777216",
}

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20)
DEFAULT_REPEATS = 5
#: ks of the chunk-size group: the standard ks plus the ones that give the same retrieved text
#: (60 x 20 = 120 x 10 = 240 x 5 = 1,200 words, 60 x 40 = 120 x 20 = 240 x 10 = 2,400 words ...).
CHUNK60_KS: tuple[int, ...] = (1, 2, 3, 5, 6, 10, 20, 40)
CHUNK120_KS: tuple[int, ...] = DEFAULT_KS
CHUNK240_KS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
EF_SWEEP: tuple[int, ...] = (20, 50, 100, 200, 400)
EF_LOW = 20
EF_CHOSEN = DEFAULT_EF_SEARCH  # 100: Settings.ef_search, the application default
INDEX_M_VALUES: tuple[int, ...] = (6, 16, 32)
INDEX_M_EFS: tuple[int, ...] = (EF_LOW, EF_CHOSEN)
OVERFETCH_DEFAULT = 10  # the overfetch factor of the rrf runs and of overfetch10
MIN_WORDS_FILTER = 1000
HEADING_FILTER = "%History%"


@dataclass(frozen=True)
class Configuration:
    """One ingest configuration: what ``run_ingest`` and the index rebuild are pinned to."""

    chunk_max_words: int
    chunk_overlap_units: int
    index_m: int
    use_prefix: bool

    @property
    def label(self) -> str:
        """Return a short human-readable description, e.g. ``120 words, overlap 1, M=6, prefix on``."""
        return (
            f"{self.chunk_max_words} words, overlap {self.chunk_overlap_units}, "
            f"M={self.index_m}, prefix {'on' if self.use_prefix else 'off'}"
        )

    def settings(self, base: Settings) -> Settings:
        """Return ``base`` with the chunking parameters of this configuration (the index M is
        the schema's; :func:`ingest_configuration` rebuilds the index for another one)."""
        return replace(
            base,
            chunk_max_words=self.chunk_max_words,
            chunk_overlap_units=self.chunk_overlap_units,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "label": self.label}

    def ingest_meta_expected(self) -> dict[str, str]:
        """Return the ``ingest_meta`` values an ingest of this configuration writes."""
        return {
            "chunk_max_words": str(self.chunk_max_words),
            "chunk_overlap_units": str(self.chunk_overlap_units),
            "embedding_prefix": "true" if self.use_prefix else "false",
            "index_m": str(self.index_m),
            "index_distance": INDEX_DISTANCE,
            "hatnote_pattern": HATNOTE_RE.pattern,
        }

    def differences(self, meta: dict[str, str], index: dict[str, Any]) -> dict[str, Any]:
        """Return what differs between this configuration and the database state, or ``{}``.

        ``meta`` is ``ingest_meta`` and ``index`` the :func:`index_definition`. The hatnote
        pattern is compared too, so an ingest made by an older version of the code (no
        ``hatnote_pattern`` key, or another rule) does not count as this configuration.
        """
        diffs: dict[str, Any] = {}
        for key, expected in self.ingest_meta_expected().items():
            if meta.get(key) != expected:
                diffs[f"ingest_meta.{key}"] = {"expected": expected, "found": meta.get(key)}
        if index.get("m") != self.index_m:
            diffs["vector_index_m"] = {"expected": self.index_m, "found": index.get("m")}
        if index.get("distance") != INDEX_DISTANCE:
            diffs["vector_index_distance"] = {
                "expected": INDEX_DISTANCE, "found": index.get("distance")
            }
        return diffs


#: The phase-1/2 configuration, on which the stability, ef_search and index_m groups run.
OLD = Configuration(chunk_max_words=120, chunk_overlap_units=1, index_m=6, use_prefix=True)
#: The chosen defaults (wikilense.config, sql/schema.sql): the filters, final and restore groups.
FINAL = Configuration(
    chunk_max_words=DEFAULT_CHUNK_MAX_WORDS,
    chunk_overlap_units=DEFAULT_CHUNK_OVERLAP_UNITS,
    index_m=db.VECTOR_INDEX_M,
    use_prefix=True,
)
CHUNK_SIZE_CASES: tuple[tuple[int, tuple[int, ...], str], ...] = (
    (60, CHUNK60_KS, "chunk60_ef_100"),
    (120, CHUNK120_KS, "chunk120_ef_100"),
    (240, CHUNK240_KS, "chunk240_ef_100"),
)
PREFIX_CASES: tuple[tuple[str, bool], ...] = (("on", True), ("off", False))

#: Identifiers read from the server (the vector index name, the database name) must match this
#: before they are written into an ALTER TABLE or a LIKE pattern.
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")

SHOW_CREATE_CHUNK_SQL = "SHOW CREATE TABLE chunk"
GLOBALS_SQL = "SELECT @@GLOBAL.mhnsw_max_cache_size, @@GLOBAL.mhnsw_ef_search"
UPTIME_SQL = "SHOW GLOBAL STATUS LIKE 'Uptime'"
CHUNK_SIZES_SQL = (
    "SELECT data_length, index_length FROM information_schema.tables "
    "WHERE table_schema = DATABASE() AND table_name = 'chunk'"
)
CHUNK_WORDS_SQL = "SELECT n_words FROM chunk"
CHUNK_COUNT_SQL = "SELECT COUNT(*) FROM chunk"
CHUNKS_MIN_WORDS_SQL = (
    "SELECT COUNT(*) FROM chunk JOIN page ON page.page_id = chunk.page_id WHERE page.n_words >= %s"
)
CHUNKS_HEADING_SQL = (
    "SELECT COUNT(*) FROM chunk JOIN section ON section.section_id = chunk.section_id "
    "WHERE section.heading LIKE %s"
)
PAGES_MIN_WORDS_SQL = "SELECT page_id FROM page WHERE n_words >= %s"
PAGES_HEADING_SQL = "SELECT DISTINCT page_id FROM section WHERE heading LIKE %s"
SENTENCES_HEADING_SQL = (
    "SELECT s.sentence_id FROM sentence AS s JOIN section ON section.section_id = s.section_id "
    "WHERE section.heading LIKE %s"
)
#: Keeps ingest_meta.index_m equal to the index that is in place after a rebuild.
UPDATE_META_INDEX_M_SQL = "UPDATE ingest_meta SET `value` = %s WHERE `key` = 'index_m'"
#: Run after every index rebuild (see the module docstring); the ingest runs it too.
ANALYZE_CHUNK_SQL = "ANALYZE TABLE chunk"
#: Hidden InnoDB tablespaces of the vector index (root: needs the PROCESS privilege). The
#: database name is checked against IDENTIFIER_RE before it is written into the pattern.
HIDDEN_TABLESPACE_SQL = (
    "SELECT name, file_size FROM information_schema.innodb_sys_tablespaces "
    "WHERE name LIKE '{db}/chunk#i#%'"
)

INDEX_DEFINITION_RE = re.compile(r"VECTOR KEY `(?P<name>[^`]+)` \(`embedding`\)(?P<options>[^\n]*)")
INDEX_M_RE = re.compile(r"`M`='(\d+)'")
INDEX_DISTANCE_RE = re.compile(r"`DISTANCE`='(\w+)'")

STABILITY_512_RUNS = ("stability_512mb_run1", "stability_512mb_run2", "stability_512mb_run3")
STABILITY_16_RUNS = ("stability_16mb_run1", "stability_16mb_run2", "stability_16mb_run3")
AFTER_RESTART = "stability_512mb_after_restart"
EXACT_OLD = "inline_ef_20"
RRF_OLD = "rrf_120_m6"
FINAL_HEADLINE = "final_ef_100"
FINAL_EF_20 = "final_ef_20"
FINAL_EXACT = "final_inline"
FINAL_RRF = "final_rrf"
FINAL_ORACLE = "final_oracle_titles"
FINAL_AFTER = {FINAL_HEADLINE: "final_ef_100_after_restart", FINAL_EF_20: "final_ef_20_after_restart"}
FINAL_RUNS = (FINAL_HEADLINE, FINAL_EF_20, FINAL_EXACT, FINAL_RRF, FINAL_ORACLE, *FINAL_AFTER.values())
INGEST_RUNS = "ingest_runs"
RESTORE_STATE = "restore_state"


class ExperimentError(RuntimeError):
    """A step of the protocol could not run; the message says which and why."""


# ---------------------------------------------------------------------------------------------
# root-level operations through docker
# ---------------------------------------------------------------------------------------------


class Server:
    """SET GLOBAL, the container restart and the hidden tablespace size, through docker."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        file_values: dict[str, str] = {}
        if DEFAULT_ENV_FILE.is_file():
            file_values = {k: v for k, v in dotenv_values(DEFAULT_ENV_FILE).items() if v}
        self._root_password = os.environ.get(ROOT_PASSWORD_VARIABLE) or file_values.get(
            ROOT_PASSWORD_VARIABLE
        )

    def _docker(self, args: Sequence[str], env: dict[str, str] | None = None) -> str:
        """Run ``docker <args>`` from the repository root and return stdout; ExperimentError on failure."""
        try:
            proc = subprocess.run(
                ["docker", *args],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ExperimentError(
                "docker is not on PATH; the restarts, SET GLOBAL and the tablespace sizes need it"
            ) from exc
        if proc.returncode != 0:
            raise ExperimentError(
                f"docker {' '.join(args[:2])} failed (exit {proc.returncode}): "
                f"{proc.stderr.strip()[:500]}"
            )
        return proc.stdout

    def root_sql(self, sql: str) -> list[list[str]]:
        """Run one statement as root inside the container; rows as lists of strings."""
        if not self._root_password:
            raise ExperimentError(
                f"{ROOT_PASSWORD_VARIABLE} is not set in {DEFAULT_ENV_FILE.name} or the environment"
            )
        env = {**os.environ, "MYSQL_PWD": self._root_password}
        out = self._docker(
            ["exec", "-e", "MYSQL_PWD", CONTAINER, "mariadb", "-uroot", "--batch",
             "--skip-column-names", "-e", sql],
            env=env,
        )
        return [line.split("\t") for line in out.splitlines() if line]

    def set_cache_size(self, nbytes: int) -> None:
        """SET GLOBAL mhnsw_max_cache_size to one of the two protocol values."""
        statement = SET_CACHE_SQL.get(nbytes)
        if statement is None:
            raise ValueError(f"cache size {nbytes} is not one of {sorted(SET_CACHE_SQL)}")
        self.root_sql(statement)
        log.info("SET GLOBAL mhnsw_max_cache_size = %d", nbytes)

    def vector_index_tablespace_bytes(self) -> int | None:
        """Return the file size of the hidden InnoDB tablespace(s) of the vector index, or None."""
        if not IDENTIFIER_RE.match(self.settings.db_name):
            raise ExperimentError(f"database name {self.settings.db_name!r} is not a plain identifier")
        try:
            rows = self.root_sql(HIDDEN_TABLESPACE_SQL.format(db=self.settings.db_name))
        except ExperimentError as exc:
            log.warning("hidden tablespace size not available: %s", exc)
            return None
        return sum(int(row[1]) for row in rows) if rows else None

    def restart(self, timeout_s: float = 300.0) -> dict[str, Any]:
        """``docker compose restart``, wait until healthy and connectable; returns timings.

        The dict has ``container_restart_seconds`` (from the command to the first successful
        connection) and ``server_uptime_seconds_after_restart`` (the server's ``Uptime`` status
        right after, which shows that a new server process answered).
        """
        start = time.perf_counter()
        log.info("restarting the container ...")
        self._docker(["compose", "restart"])
        deadline = start + timeout_s
        while True:
            status = self._docker(
                ["inspect", "--format", "{{.State.Health.Status}}", CONTAINER]
            ).strip()
            if status == "healthy":
                break
            if time.perf_counter() > deadline:
                raise ExperimentError(f"container not healthy after {timeout_s:.0f} s ({status})")
            time.sleep(1.0)
        while True:
            try:
                conn = db.connect(self.settings)
                try:
                    with conn.cursor() as cur:
                        cur.execute(UPTIME_SQL)
                        uptime = int(cur.fetchone()[1])
                finally:
                    conn.close()
                break
            except pymysql.err.OperationalError as exc:
                if time.perf_counter() > deadline:
                    raise ExperimentError(f"server not connectable after restart: {exc}") from exc
                time.sleep(1.0)
        seconds = time.perf_counter() - start
        log.info("container healthy and connectable after %.1f s (server uptime %d s)", seconds, uptime)
        return {
            "container_restart_seconds": round(seconds, 1),
            "server_uptime_seconds_after_restart": uptime,
        }


# ---------------------------------------------------------------------------------------------
# server state recorded in every result file
# ---------------------------------------------------------------------------------------------


def index_definition(conn: pymysql.Connection) -> dict[str, Any]:
    """Return ``{name, m, distance}`` of the vector index from SHOW CREATE TABLE chunk.

    Raises ExperimentError when the index name is not a plain identifier (it is written into
    the ALTER TABLE statements of :func:`rebuild_index`).
    """
    with conn.cursor() as cur:
        cur.execute(SHOW_CREATE_CHUNK_SQL)
        ddl = str(cur.fetchone()[1])
    match = INDEX_DEFINITION_RE.search(ddl)
    if match is None:
        return {"name": None, "m": None, "distance": None}
    name = match.group("name")
    if not IDENTIFIER_RE.match(name):
        raise ExperimentError(f"vector index name {name!r} is not a plain identifier")
    options = match.group("options")
    m_match = INDEX_M_RE.search(options)
    d_match = INDEX_DISTANCE_RE.search(options)
    return {
        "name": name,
        "m": int(m_match.group(1)) if m_match else None,
        "distance": d_match.group(1).lower() if d_match else None,
    }


def server_snapshot(conn: pymysql.Connection, server: Server) -> dict[str, Any]:
    """Return the globals, the index definition, the table sizes and the chunk word statistics."""
    with conn.cursor() as cur:
        cur.execute(GLOBALS_SQL)
        cache_size, ef_global = cur.fetchone()
        cur.execute(CHUNK_SIZES_SQL)
        data_length, index_length = cur.fetchone()
        cur.execute(CHUNK_WORDS_SQL)
        words = np.asarray([int(row[0]) for row in cur.fetchall()], dtype=np.float64)
    definition = index_definition(conn)
    return {
        "mhnsw_max_cache_size": int(cache_size),
        "mhnsw_ef_search_global": int(ef_global),
        "vector_index_name": definition["name"],
        "vector_index_m": definition["m"],
        "vector_index_distance": definition["distance"],
        "chunk_data_length": int(data_length),
        "chunk_index_length": int(index_length),
        "vector_index_tablespace_bytes": server.vector_index_tablespace_bytes(),
        "chunk_words_mean": round(float(words.mean()), 1) if words.size else None,
        "chunk_words_median": float(np.median(words)) if words.size else None,
        "chunk_words_p95": float(np.percentile(words, 95)) if words.size else None,
        "chunk_words_max": int(words.max()) if words.size else None,
    }


def hatnote_count(meta: dict[str, str]) -> int | None:
    """Return ``ingest_meta.n_units_hatnote`` as an int, or None when the ingest did not record it."""
    value = meta.get("n_units_hatnote")
    return int(value) if value is not None and value.isdigit() else None


# ---------------------------------------------------------------------------------------------
# context, evaluation, ingest and rebuild wrappers
# ---------------------------------------------------------------------------------------------


@dataclass
class Context:
    """Settings, the docker helper, the output directory and the lazily loaded embedder."""

    settings: Settings
    server: Server
    out_dir: Path
    repeats: int
    _embedder: Embedder | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = make_embedder(self.settings)
        return self._embedder

    def connect(self) -> pymysql.Connection:
        return db.connect(self.settings)

    def database_state(self) -> tuple[dict[str, str], dict[str, Any]]:
        """Return ``(ingest_meta, index_definition)`` from a fresh connection."""
        conn = self.connect()
        try:
            return read_ingest_meta(conn), index_definition(conn)
        finally:
            conn.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    """Return the parsed JSON file, or None when it does not exist."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def md_cell(value: Any) -> str:
    text = "-" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(md_cell(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(md_cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def describe_state(meta: dict[str, str], index: dict[str, Any]) -> str:
    """Return a one-line description of the ingest and index the database holds."""
    return (
        f"chunk_max_words={meta.get('chunk_max_words')}, overlap={meta.get('chunk_overlap_units')}, "
        f"prefix={meta.get('embedding_prefix')}, ingest_meta.index_m={meta.get('index_m')}, "
        f"index {index.get('name')} M={index.get('m')} DISTANCE={index.get('distance')}, "
        f"ingested_at={meta.get('ingested_at')}"
    )


def check_configuration(ctx: Context, config: Configuration, what: str) -> dict[str, str]:
    """Raise ExperimentError unless the database holds ``config``; returns ``ingest_meta``."""
    meta, index = ctx.database_state()
    diffs = config.differences(meta, index)
    if diffs:
        raise ExperimentError(
            f"{what}: the database holds {describe_state(meta, index)}, not "
            f"'{config.label}': {diffs}"
        )
    return meta


def run_eval(
    ctx: Context,
    name: str,
    *,
    group: str,
    config: Configuration,
    ks: Sequence[int],
    ef_search: int,
    strategy: str,
    filters: Filters | None = None,
    claim_filters: ClaimFilters | None = None,
    overfetch: int = OVERFETCH_DEFAULT,
    extra: dict[str, Any] | None = None,
) -> EvalResult:
    """Run one evaluation on a fresh connection, add the server state, write <name>.json/.md.

    The database must hold ``config`` (checked first, ExperimentError otherwise); ``ef_search``
    and ``strategy`` are always given explicitly. ``extra`` is merged into the parameters.
    """
    log.info(
        "eval %s: %s; strategy=%s ef_search=%d ks=%s overfetch=%s filters=%s claim_filters=%s",
        name, config.label, strategy, ef_search, list(ks),
        overfetch if strategy in ("overfetch", "rrf") else "-", filters,
        getattr(claim_filters, "__name__", None),
    )
    check_configuration(ctx, config, name)
    conn = ctx.connect()
    try:
        started = time.perf_counter()
        result = evaluate(
            conn,
            ctx.embedder,
            ks=ks,
            repeats=ctx.repeats,
            strategy=strategy,
            filters=filters,
            ef_search=ef_search,
            overfetch=overfetch,
            claim_filters=claim_filters,
            settings=ctx.settings,
        )
        result.parameters["experiment_group"] = group
        result.parameters["configuration"] = config.to_dict()
        result.parameters["eval_seconds"] = round(time.perf_counter() - started, 1)
        result.parameters["n_units_hatnote"] = hatnote_count(result.ingest_meta)
        result.parameters.update(server_snapshot(conn, ctx.server))
    finally:
        conn.close()
    if extra:
        result.parameters.update(extra)
    json_path, _ = write_results(result, out_dir=ctx.out_dir, name=name)
    summary = ", ".join(
        f"k{m.k}: art {m.article_hits}/{result.n_claims} ev {m.evidence_hits}/"
        f"{result.n_evidence_claims} p50 {m.sql_latency.p50_ms:.2f}ms"
        for m in result.per_k
    )
    log.info("  %s -> %s (unstable %s)", json_path.name, summary, result.unstable_claims)
    return result


def append_ingest_run(ctx: Context, info: dict[str, Any]) -> None:
    """Append one ingest or rebuild record to ingest_runs.json and rewrite ingest_runs.md."""
    runs = load_json(ctx.out_dir / f"{INGEST_RUNS}.json") or []
    runs.append(info)
    write_json(ctx.out_dir / f"{INGEST_RUNS}.json", runs)
    (ctx.out_dir / f"{INGEST_RUNS}.md").write_text(ingest_runs_markdown(runs), encoding="utf-8")


def rebuild_index(ctx: Context, m: int, *, label: str, record: bool = True) -> dict[str, Any]:
    """DROP and re-ADD the vector index with M=``m``, then ANALYZE TABLE chunk (each timed).

    The index name comes from ``SHOW CREATE TABLE chunk`` (a plain identifier, see
    :func:`index_definition`), ``m`` must be one of ``INDEX_M_VALUES`` and the distance is
    ``ingest.INDEX_DISTANCE``; the ANALYZE is explained in the module docstring.
    ``ingest_meta.index_m`` is set to ``m`` afterwards. Returns the timings and sizes. With
    ``record`` the rebuild gets its own row in ingest_runs.json (an ingest that rebuilds right
    away carries the rebuild inside its own row instead).
    """
    if m not in INDEX_M_VALUES:
        raise ValueError(f"M={m} is not one of {INDEX_M_VALUES}")
    conn = ctx.connect()
    try:
        before = index_definition(conn)
        name = before["name"]
        if name is None:
            raise ExperimentError("chunk has no vector index to rebuild (SHOW CREATE TABLE chunk)")
        drop_sql = f"ALTER TABLE chunk DROP INDEX `{name}`"
        add_sql = (
            f"ALTER TABLE chunk ADD VECTOR INDEX `{name}` (embedding) "
            f"M={int(m)} DISTANCE={INDEX_DISTANCE}"
        )
        log.info("rebuilding the vector index `%s`: M=%s -> M=%d (%s)", name, before["m"], m, label)
        with conn.cursor() as cur:
            start = time.perf_counter()
            cur.execute(drop_sql)
            drop_seconds = time.perf_counter() - start
            start = time.perf_counter()
            cur.execute(add_sql)
            add_seconds = time.perf_counter() - start
            start = time.perf_counter()
            cur.execute(ANALYZE_CHUNK_SQL)
            for _table, _op, msg_type, msg_text in cur.fetchall():
                if str(msg_type).lower() == "error":
                    raise ExperimentError(f"ANALYZE TABLE chunk failed: {msg_text}")
            analyze_seconds = time.perf_counter() - start
            cur.execute(UPDATE_META_INDEX_M_SQL, (str(m),))
        conn.commit()
        after = index_definition(conn)
        if after["name"] != name or after["m"] != m or after["distance"] != INDEX_DISTANCE:
            raise ExperimentError(f"index rebuild with M={m} left {after}")
        snapshot = server_snapshot(conn, ctx.server)
        with conn.cursor() as cur:
            cur.execute(CHUNK_COUNT_SQL)
            n_chunks = int(cur.fetchone()[0])
    finally:
        conn.close()
    info = {
        "label": label,
        "kind": "rebuild",
        "m": m,
        "m_before": before["m"],
        "index_name": name,
        "drop_sql": drop_sql,
        "add_sql": add_sql,
        "drop_seconds": round(drop_seconds, 2),
        "add_seconds": round(add_seconds, 2),
        "analyze_seconds": round(analyze_seconds, 3),
        "vector_index_m_after": snapshot["vector_index_m"],
        "vector_index_distance_after": snapshot["vector_index_distance"],
        "n_chunks": n_chunks,
        "chunk_index_length": snapshot["chunk_index_length"],
        "chunk_data_length": snapshot["chunk_data_length"],
        "vector_index_tablespace_bytes": snapshot["vector_index_tablespace_bytes"],
        "rebuilt_at": utc_now(),
    }
    log.info(
        "  drop %.2f s, add %.2f s, analyze %.3f s; index_length %s, vector tablespace %s bytes",
        drop_seconds, add_seconds, analyze_seconds, info["chunk_index_length"],
        info["vector_index_tablespace_bytes"],
    )
    if record:
        append_ingest_run(ctx, info)
    return info


def ingest_configuration(ctx: Context, config: Configuration, *, label: str) -> dict[str, Any]:
    """Fresh ingest of ``config`` (reset), rebuilding the index when its M is not the schema's.

    Returns the record appended to ingest_runs.json (counts, seconds, sizes, the rebuild).
    """
    settings = config.settings(ctx.settings)
    log.info("ingest %s: %s", label, config.label)
    report = run_ingest(settings, embedder=ctx.embedder, use_prefix=config.use_prefix, progress=False)
    conn = ctx.connect()
    try:
        definition = index_definition(conn)
    finally:
        conn.close()
    rebuild = None
    if definition["m"] != config.index_m:
        rebuild = rebuild_index(
            ctx, config.index_m, label=f"{label}_rebuild_m{config.index_m}", record=False
        )
    meta = check_configuration(ctx, config, f"ingest {label}")
    conn = ctx.connect()
    try:
        snapshot = server_snapshot(conn, ctx.server)
    finally:
        conn.close()
    info: dict[str, Any] = {
        "label": label,
        "kind": "ingest",
        "configuration": config.to_dict(),
        "ingested_at": meta.get("ingested_at"),
        "counts": report.counts(),
        "seconds": {stage: round(value, 2) for stage, value in report.seconds.items()},
        "n_units_hatnote": hatnote_count(meta),
        "chunk_words_mean": snapshot["chunk_words_mean"],
        "chunk_words_median": snapshot["chunk_words_median"],
        "chunk_words_p95": snapshot["chunk_words_p95"],
        "chunk_words_max": snapshot["chunk_words_max"],
        "vector_index_name": snapshot["vector_index_name"],
        "vector_index_m": snapshot["vector_index_m"],
        "vector_index_distance": snapshot["vector_index_distance"],
        "chunk_data_length": snapshot["chunk_data_length"],
        "chunk_index_length": snapshot["chunk_index_length"],
        "vector_index_tablespace_bytes": snapshot["vector_index_tablespace_bytes"],
        "index_rebuild": rebuild,
    }
    log.info(
        "  %d chunks (mean %.1f words, %d hatnote units excluded) in %.1f s (embed %.1f s)",
        report.n_chunks, snapshot["chunk_words_mean"] or 0.0, report.n_units_hatnote,
        report.seconds["total"], report.seconds["embed"],
    )
    append_ingest_run(ctx, info)
    return info


def ensure_configuration(ctx: Context, config: Configuration, *, label: str) -> dict[str, Any] | None:
    """Ingest ``config`` unless the database already holds exactly it; returns the ingest record."""
    meta, index = ctx.database_state()
    diffs = config.differences(meta, index)
    if not diffs:
        log.info("database holds '%s' (%s); no ingest", config.label, describe_state(meta, index))
        return None
    log.info(
        "database holds %s, not '%s' (%s); ingesting", describe_state(meta, index), config.label,
        diffs,
    )
    return ingest_configuration(ctx, config, label=label)


def ingest_runs_markdown(runs: list[dict[str, Any]]) -> str:
    rows = []
    for r in runs:
        if r.get("kind") == "rebuild":
            rows.append((
                r["label"], "rebuild", f"M {r['m_before']} -> {r['m']}", r["n_chunks"], "-", "-",
                f"drop {r['drop_seconds']} + add {r['add_seconds']} + analyze {r.get('analyze_seconds')}",
                "-", r["vector_index_tablespace_bytes"], r["rebuilt_at"],
            ))
            continue
        rebuild = r.get("index_rebuild") or {}
        total = str(r["seconds"]["total"])
        if rebuild:
            total += f" + rebuild M={rebuild['m']} {rebuild['add_seconds']}"
        rows.append((
            r["label"], "ingest", r["configuration"]["label"], r["counts"]["n_chunks"],
            r["n_units_hatnote"], r["chunk_words_mean"], total, r["seconds"]["embed"],
            r["vector_index_tablespace_bytes"], r["ingested_at"],
        ))
    return (
        "# Ingests and index rebuilds run by scripts/run_experiments.py\n\n"
        "One row per `run_ingest` call or `ALTER TABLE` rebuild, in order. Seconds are the "
        "stages of `IngestReport` (an ingest whose configuration needs another M than the "
        "schema's is followed by a rebuild, listed on its own row too).\n\n"
        + md_table(
            ("label", "kind", "configuration", "chunks", "hatnote units", "words mean",
             "total s", "embed s", "vector index bytes", "at"),
            rows,
        )
        + "\n"
    )


def ensure_cache_size(ctx: Context, nbytes: int) -> int:
    """Set the global HNSW cache size when it differs; returns the value now in force."""
    conn = ctx.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(GLOBALS_SQL)
            current = int(cur.fetchone()[0])
    finally:
        conn.close()
    if current != nbytes:
        ctx.server.set_cache_size(nbytes)
        return nbytes
    return current


# ---------------------------------------------------------------------------------------------
# comparisons of per-claim hits between result files
# ---------------------------------------------------------------------------------------------


def claim_hits(path: Path) -> dict[int, tuple[int, ...]]:
    data = load_json(path)
    if data is None:
        raise ExperimentError(f"{path} not found; run the group that writes it first")
    return {int(c["claim_id"]): tuple(c["hit_chunk_ids"]) for c in data["claims"]}


def claim_ranks(path: Path) -> dict[int, int | None]:
    data = load_json(path)
    return {int(c["claim_id"]): c["gold_page_rank"] for c in data["claims"]}


def compare_hits(out_dir: Path, reference: str, others: Sequence[str]) -> dict[str, Any]:
    """Compare the max-k hit chunk ids of every claim between ``reference`` and each other run."""
    ref = claim_hits(out_dir / f"{reference}.json")
    ref_ranks = claim_ranks(out_dir / f"{reference}.json")
    runs: dict[str, Any] = {}
    differing_any: set[int] = set()
    for name in others:
        other = claim_hits(out_dir / f"{name}.json")
        other_ranks = claim_ranks(out_dir / f"{name}.json")
        differing = sorted(cid for cid in ref if other.get(cid) != ref[cid])
        same_set = sum(
            1 for cid in ref if cid in other and set(other[cid]) == set(ref[cid])
        )
        differing_any.update(differing)
        runs[name] = {
            "identical_sequence": len(ref) - len(differing),
            "identical_set": same_set,
            "differing_claims": differing,
            "differing_details": [
                {
                    "claim_id": cid,
                    "first_differing_position": next(
                        (i + 1 for i, (a, b) in enumerate(zip(ref[cid], other.get(cid, ())))
                         if a != b),
                        min(len(ref[cid]), len(other.get(cid, ()))) + 1,
                    ),
                    "common_chunks": len(set(ref[cid]) & set(other.get(cid, ()))),
                    "gold_page_rank_reference": ref_ranks.get(cid),
                    "gold_page_rank_other": other_ranks.get(cid),
                }
                for cid in differing
            ],
        }
    return {
        "reference": reference,
        "n_claims": len(ref),
        "runs": runs,
        "identical_in_all": len(ref) - len(differing_any),
        "differing_claims_any": sorted(differing_any),
    }


def comparison_rows(comparison: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for name, run in comparison["runs"].items():
        rows.append([
            comparison["reference"], name, run["identical_sequence"], run["identical_set"],
            comparison["n_claims"], ", ".join(str(c) for c in run["differing_claims"]) or "none",
        ])
    return rows


def lost_gold_pages(out_dir: Path, reference: str, other: str) -> list[int]:
    """Return the claims with a gold page in the top hits of ``reference`` but not of ``other``."""
    ref = claim_ranks(out_dir / f"{reference}.json")
    oth = claim_ranks(out_dir / f"{other}.json")
    return sorted(cid for cid, rank in ref.items() if rank is not None and oth.get(cid) is None)


def run_summary_fields(out_dir: Path, name: str) -> dict[str, Any]:
    data = load_json(out_dir / f"{name}.json")
    if data is None:
        return {"name": name, "found": False}
    return {
        "name": name,
        "found": True,
        "created_at": data["created_at"],
        "ingested_at": data["ingest_meta"].get("ingested_at"),
        "mhnsw_max_cache_size": data["parameters"].get("mhnsw_max_cache_size"),
        "ef_search_effective": data["parameters"].get("ef_search_effective"),
        "vector_index_m": data["parameters"].get("vector_index_m"),
        "n_chunks": data["parameters"].get("n_chunks"),
        "unstable_claims": data["unstable_claims"],
        "article_hits": {str(m["k"]): m["article_hits"] for m in data["per_k"]},
        "evidence_hits": {str(m["k"]): m["evidence_hits"] for m in data["per_k"]},
        "sql_p50_ms": {str(m["k"]): round(m["sql_latency"]["p50_ms"], 3) for m in data["per_k"]},
    }


def comparison_markdown(title: str, intro: str, data: dict[str, Any]) -> str:
    """Return the Markdown of a runs table plus one table per comparison."""
    parts = [
        f"# {title}",
        "",
        f"Generated {data['created_at']}. {intro}",
        "",
        "## Runs",
        "",
        md_table(
            ("run", "created_at", "index ingested_at", "cache bytes", "ef_search", "M", "chunks",
             "unstable claims", "article hits @1/5/10/20", "evidence hits @5/10/20"),
            [
                (
                    r["name"], r.get("created_at"), r.get("ingested_at"),
                    r.get("mhnsw_max_cache_size"), r.get("ef_search_effective"),
                    r.get("vector_index_m"), r.get("n_chunks"),
                    len(r.get("unstable_claims", [])) if r.get("found") else "-",
                    "/".join(str(r["article_hits"].get(k, "-")) for k in ("1", "5", "10", "20"))
                    if r.get("found") else "not found",
                    "/".join(str(r["evidence_hits"].get(k, "-")) for k in ("5", "10", "20"))
                    if r.get("found") else "not found",
                )
                for r in data["runs"]
            ],
        ),
        "",
        "## Comparisons",
        "",
    ]
    for key, comparison in data["comparisons"].items():
        parts.append(f"### {key}")
        parts.append("")
        parts.append(md_table(
            ("reference", "run", "identical (sequence)", "identical (set)", "of", "claims that differ"),
            comparison_rows(comparison),
        ))
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------
# group 1: stability (OLD: 120 words, M=6)
# ---------------------------------------------------------------------------------------------


def group_stability(ctx: Context) -> None:
    info = ingest_configuration(ctx, OLD, label="stability_old")
    cache = ensure_cache_size(ctx, CACHE_512MB)
    log.info("mhnsw_max_cache_size in force: %d", cache)
    common = {"group": "stability", "config": OLD, "ks": DEFAULT_KS, "strategy": "none"}
    for name in STABILITY_512_RUNS:
        run_eval(ctx, name, ef_search=EF_LOW, extra={"ingest_run": info["label"]}, **common)
    restart = ctx.server.restart()
    run_eval(ctx, AFTER_RESTART, ef_search=EF_LOW, extra={"ingest_run": info["label"], **restart},
             **common)
    ctx.server.set_cache_size(CACHE_16MB)
    try:
        for name in STABILITY_16_RUNS:
            run_eval(ctx, name, ef_search=EF_LOW, extra={"ingest_run": info["label"]}, **common)
    finally:
        ctx.server.set_cache_size(CACHE_512MB)
    write_stability_comparison(ctx.out_dir)


def write_stability_comparison(out_dir: Path) -> None:
    """Write stability_comparison.json/.md from the seven stability result files."""
    all_runs = [*STABILITY_512_RUNS, AFTER_RESTART, *STABILITY_16_RUNS]
    comparisons = {
        "within_512mb": compare_hits(out_dir, STABILITY_512_RUNS[0], STABILITY_512_RUNS[1:]),
        "after_restart_vs_512mb": compare_hits(out_dir, STABILITY_512_RUNS[0], [AFTER_RESTART]),
        "after_restart_vs_16mb": compare_hits(out_dir, AFTER_RESTART, [STABILITY_16_RUNS[0]]),
        "within_16mb": compare_hits(out_dir, STABILITY_16_RUNS[0], STABILITY_16_RUNS[1:]),
        "512mb_vs_16mb": compare_hits(out_dir, STABILITY_512_RUNS[0], [*STABILITY_16_RUNS]),
    }
    data = {
        "created_at": utc_now(),
        "configuration": OLD.to_dict(),
        "runs": [run_summary_fields(out_dir, name) for name in all_runs],
        "comparisons": comparisons,
        "gold_page_lost_after_restart": lost_gold_pages(out_dir, STABILITY_512_RUNS[0], AFTER_RESTART),
        "gold_page_gained_after_restart": lost_gold_pages(out_dir, AFTER_RESTART, STABILITY_512_RUNS[0]),
    }
    write_json(out_dir / "stability_comparison.json", data)
    (out_dir / "stability_comparison.md").write_text(
        comparison_markdown(
            "Stability of the HNSW results (per-claim hit chunk ids, LIMIT 20)",
            (f"Configuration: {OLD.label}; strategy none, ef 20. A claim counts as identical when "
             "its 20 hit chunk ids are the same, in the same order; `identical (set)` ignores "
             "the order. The container was restarted between `stability_512mb_run3` and "
             "`stability_512mb_after_restart`; `SET GLOBAL mhnsw_max_cache_size = 16777216` was "
             "issued before `stability_16mb_run1` and 536870912 restored afterwards."),
            data,
        ),
        encoding="utf-8",
    )
    log.info("wrote stability_comparison.json/.md")


# ---------------------------------------------------------------------------------------------
# group 2: ef_search sweep (OLD)
# ---------------------------------------------------------------------------------------------


def group_ef_search(ctx: Context) -> None:
    ensure_configuration(ctx, OLD, label="ef_search_old")
    common = {"group": "ef_search", "config": OLD, "ks": DEFAULT_KS}
    for ef in EF_SWEEP:
        run_eval(ctx, f"ef_{ef}", ef_search=ef, strategy="none", **common)
    for ef in (EF_LOW, EF_CHOSEN):
        run_eval(ctx, f"inline_ef_{ef}", ef_search=ef, strategy="inline", **common)
    run_eval(ctx, RRF_OLD, ef_search=EF_CHOSEN, strategy="rrf", overfetch=OVERFETCH_DEFAULT, **common)


# ---------------------------------------------------------------------------------------------
# group 3: index M sweep (OLD vectors, the index rebuilt with M=6, 16, 32)
# ---------------------------------------------------------------------------------------------


def group_index_m(ctx: Context) -> None:
    ensure_configuration(ctx, OLD, label="index_m_old")
    rebuilds: list[dict[str, Any]] = []
    for m in INDEX_M_VALUES:
        info = rebuild_index(ctx, m, label=f"index_m_sweep_m{m}")
        rebuilds.append(info)
        config = replace(OLD, index_m=m)
        for ef in INDEX_M_EFS:
            run_eval(
                ctx, f"m{m}_ef_{ef}", group="index_m", config=config, ks=DEFAULT_KS,
                ef_search=ef, strategy="none", extra={"index_rebuild": info},
            )
    rebuilds.append(rebuild_index(ctx, OLD.index_m, label="index_m_restore_m6"))
    check_configuration(ctx, OLD, "index_m: after the sweep")
    comparisons: dict[str, Any] = {}
    for reference in (f"ef_{EF_LOW}", STABILITY_512_RUNS[0], AFTER_RESTART):
        if (ctx.out_dir / f"{reference}.json").is_file():
            comparisons[f"m6_ef_20_vs_{reference}"] = compare_hits(
                ctx.out_dir, reference, [f"m6_ef_{EF_LOW}"]
            )
    if (ctx.out_dir / f"{EXACT_OLD}.json").is_file():
        comparisons["exact_vs_rebuilt"] = compare_hits(
            ctx.out_dir, EXACT_OLD, [f"m{m}_ef_{ef}" for m in INDEX_M_VALUES for ef in INDEX_M_EFS]
        )
    data = {
        "created_at": utc_now(),
        "configuration": OLD.to_dict(),
        "rebuilds": rebuilds,
        "comparisons": comparisons,
    }
    write_json(ctx.out_dir / "index_m_rebuild.json", data)
    parts = [
        "# Vector index rebuilds (ALTER TABLE chunk DROP INDEX / ADD VECTOR INDEX ... M=n DISTANCE=cosine)",
        "",
        (f"Generated {data['created_at']}. Vectors of the ingest '{OLD.label}'. `index_length` is "
        "information_schema.tables for `chunk` (all secondary indexes); `vector tablespace` is "
        "the file size of the hidden InnoDB table `chunk#i#NN` that holds the HNSW graph."),
        "",
        md_table(
            ("label", "M before", "M", "drop s", "add s", "analyze s", "index_length",
             "vector tablespace bytes", "rebuilt_at"),
            [
                (r["label"], r["m_before"], r["m"], r["drop_seconds"], r["add_seconds"],
                 r.get("analyze_seconds"), r["chunk_index_length"],
                 r["vector_index_tablespace_bytes"], r["rebuilt_at"])
                for r in rebuilds
            ],
        ),
        "",
    ]
    for key, comparison in comparisons.items():
        parts.append(f"## {key}")
        parts.append("")
        parts.append(md_table(
            ("reference", "run", "identical (sequence)", "identical (set)", "of", "claims that differ"),
            comparison_rows(comparison),
        ))
        parts.append("")
    (ctx.out_dir / "index_m_rebuild.md").write_text("\n".join(parts), encoding="utf-8")
    log.info("wrote index_m_rebuild.json/.md")


# ---------------------------------------------------------------------------------------------
# group 4: chunk size (M=16, ef 100), group 5: prefix (120 words, M=16, ef 100)
# ---------------------------------------------------------------------------------------------


def group_chunk_size(ctx: Context) -> None:
    for words, ks, name in CHUNK_SIZE_CASES:
        config = Configuration(
            chunk_max_words=words, chunk_overlap_units=DEFAULT_CHUNK_OVERLAP_UNITS,
            index_m=db.VECTOR_INDEX_M, use_prefix=True,
        )
        info = ingest_configuration(ctx, config, label=f"chunk{words}")
        run_eval(
            ctx, name, group="chunk_size", config=config, ks=ks, ef_search=EF_CHOSEN,
            strategy="none", extra={"ingest_run": info},
        )


def group_prefix(ctx: Context) -> None:
    for label, use_prefix in PREFIX_CASES:
        config = Configuration(
            chunk_max_words=120, chunk_overlap_units=DEFAULT_CHUNK_OVERLAP_UNITS,
            index_m=db.VECTOR_INDEX_M, use_prefix=use_prefix,
        )
        info = ingest_configuration(ctx, config, label=f"prefix_{label}")
        common = {"group": "prefix", "config": config, "ks": DEFAULT_KS, "extra": {"ingest_run": info}}
        run_eval(ctx, f"prefix_{label}_ef_{EF_CHOSEN}", ef_search=EF_CHOSEN, strategy="none", **common)
        run_eval(ctx, f"prefix_{label}_inline", ef_search=EF_CHOSEN, strategy="inline", **common)


# ---------------------------------------------------------------------------------------------
# group 6: filter strategies (FINAL)
# ---------------------------------------------------------------------------------------------


def filter_ground_truth(conn: pymysql.Connection) -> dict[str, Any]:
    """Return, per filter, how many chunks pass it and how many claims' gold evidence it excludes."""
    truths = load_ground_truth(conn)
    with conn.cursor() as cur:
        cur.execute(CHUNK_COUNT_SQL)
        n_chunks = int(cur.fetchone()[0])
        cur.execute(CHUNKS_MIN_WORDS_SQL, (MIN_WORDS_FILTER,))
        chunks_min_words = int(cur.fetchone()[0])
        cur.execute(CHUNKS_HEADING_SQL, (HEADING_FILTER,))
        chunks_heading = int(cur.fetchone()[0])
        cur.execute(PAGES_MIN_WORDS_SQL, (MIN_WORDS_FILTER,))
        long_pages = {int(r[0]) for r in cur.fetchall()}
        cur.execute(PAGES_HEADING_SQL, (HEADING_FILTER,))
        heading_pages = {int(r[0]) for r in cur.fetchall()}
        cur.execute(SENTENCES_HEADING_SQL, (HEADING_FILTER,))
        heading_sentences = {int(r[0]) for r in cur.fetchall()}
    n_claims = len(truths)
    n_eligible = sum(1 for t in truths if t.eligible)
    excluded_min = sorted(t.claim_id for t in truths if not (t.gold_pages & long_pages))
    excluded_heading = sorted(t.claim_id for t in truths if not (t.gold_pages & heading_pages))
    evidence_min = sum(1 for t in truths if t.eligible and t.gold_pages & long_pages)
    evidence_heading = sum(
        1 for t in truths
        if t.eligible and any(s <= heading_sentences for s in t.sentence_only_sets)
    )
    return {
        "minwords1000": {
            "filter": {"min_words": MIN_WORDS_FILTER},
            "n_chunks": n_chunks,
            "n_chunks_passing": chunks_min_words,
            "n_pages_passing": len(long_pages),
            "claims_gold_page_excluded": len(excluded_min),
            "claims_gold_page_excluded_ids": excluded_min,
            "article_recall_ceiling": n_claims - len(excluded_min),
            "n_claims": n_claims,
            "evidence_recall_ceiling": evidence_min,
            "n_evidence_claims": n_eligible,
        },
        "history": {
            "filter": {"heading_like": HEADING_FILTER},
            "n_chunks": n_chunks,
            "n_chunks_passing": chunks_heading,
            "n_pages_passing": len(heading_pages),
            "claims_gold_page_excluded": len(excluded_heading),
            "claims_gold_page_excluded_ids": excluded_heading,
            "article_recall_ceiling": n_claims - len(excluded_heading),
            "n_claims": n_claims,
            "evidence_recall_ceiling": evidence_heading,
            "n_evidence_claims": n_eligible,
        },
    }


def count_short_results(
    ctx: Context, ks: Sequence[int], filters: Filters, strategy: str, overfetch: int, ef_search: int
) -> dict[str, Any]:
    """Run search() once per claim and k (untimed) and count the queries that returned < k rows."""
    conn = ctx.connect()
    try:
        truths = load_ground_truth(conn)
        vectors = np.asarray(ctx.embedder.embed_queries([t.text for t in truths]), dtype=np.float32)
        out: dict[str, Any] = {}
        for k in ks:
            counts = [
                len(search(conn, vectors[i], k=k, filters=filters, strategy=strategy,
                           overfetch=overfetch, ef_search=ef_search, query_text=truths[i].text))
                for i in range(len(truths))
            ]
            short = [t.claim_id for t, n in zip(truths, counts) if n < k]
            out[str(k)] = {
                "queries_short_of_k": len(short),
                "queries_with_no_rows": sum(1 for n in counts if n == 0),
                "rows_min": min(counts),
                "rows_mean": round(sum(counts) / len(counts), 2),
                "short_claim_ids": short,
            }
    finally:
        conn.close()
    return out


FILTER_CASES: tuple[tuple[str, Filters], ...] = (
    ("minwords1000", Filters(min_words=MIN_WORDS_FILTER)),
    ("history", Filters(heading_like=HEADING_FILTER)),
)
STRATEGY_CASES: tuple[tuple[str, str, int], ...] = (
    ("inline", "inline", OVERFETCH_DEFAULT),
    ("overfetch10", "overfetch", 10),
    ("overfetch50", "overfetch", 50),
)


def group_filters(ctx: Context) -> None:
    ensure_configuration(ctx, FINAL, label="filters_final")
    conn = ctx.connect()
    try:
        truth = filter_ground_truth(conn)
    finally:
        conn.close()
    for filter_name, filters in FILTER_CASES:
        for strategy_name, strategy, overfetch in STRATEGY_CASES:
            name = f"strategy_{strategy_name}_{filter_name}"
            short = count_short_results(ctx, DEFAULT_KS, filters, strategy, overfetch, EF_CHOSEN)
            run_eval(
                ctx, name, group="filters", config=FINAL, ks=DEFAULT_KS, ef_search=EF_CHOSEN,
                strategy=strategy, filters=filters, overfetch=overfetch,
                extra={"filter_ground_truth": truth[filter_name], "short_results": short},
            )


# ---------------------------------------------------------------------------------------------
# group 7: the final configuration
# ---------------------------------------------------------------------------------------------


def group_final(ctx: Context) -> None:
    info = ingest_configuration(ctx, FINAL, label="final")
    cache = ensure_cache_size(ctx, CACHE_512MB)
    log.info("mhnsw_max_cache_size in force: %d", cache)
    common = {"group": "final", "config": FINAL, "ks": DEFAULT_KS, "extra": {"ingest_run": info}}
    run_eval(ctx, FINAL_HEADLINE, ef_search=EF_CHOSEN, strategy="none", **common)
    run_eval(ctx, FINAL_EF_20, ef_search=EF_LOW, strategy="none", **common)
    run_eval(ctx, FINAL_EXACT, ef_search=EF_CHOSEN, strategy="inline", **common)
    run_eval(ctx, FINAL_RRF, ef_search=EF_CHOSEN, strategy="rrf", overfetch=OVERFETCH_DEFAULT, **common)
    run_eval(
        ctx, FINAL_ORACLE, ef_search=EF_CHOSEN, strategy="inline", claim_filters=oracle_title_filters,
        **common,
    )
    restart = ctx.server.restart()
    after_common = {**common, "extra": {"ingest_run": info, **restart}}
    run_eval(ctx, FINAL_AFTER[FINAL_HEADLINE], ef_search=EF_CHOSEN, strategy="none", **after_common)
    run_eval(ctx, FINAL_AFTER[FINAL_EF_20], ef_search=EF_LOW, strategy="none", **after_common)
    write_final_restart_comparison(ctx.out_dir)


def write_final_restart_comparison(out_dir: Path) -> None:
    """Write final_restart_comparison.json/.md: hit lists before and after the restart (M=16)."""
    comparisons: dict[str, Any] = {}
    per_k: dict[str, Any] = {}
    for before, after in FINAL_AFTER.items():
        comparisons[f"{before}_vs_{after}"] = compare_hits(out_dir, before, [after])
        fields = {name: run_summary_fields(out_dir, name) for name in (before, after)}
        per_k[before] = {
            "before": before,
            "after": after,
            "article_hits_before": fields[before]["article_hits"],
            "article_hits_after": fields[after]["article_hits"],
            "evidence_hits_before": fields[before]["evidence_hits"],
            "evidence_hits_after": fields[after]["evidence_hits"],
            "gold_page_lost_after_restart": lost_gold_pages(out_dir, before, after),
            "gold_page_gained_after_restart": lost_gold_pages(out_dir, after, before),
        }
    comparisons["exact_vs_approximate"] = compare_hits(
        out_dir, FINAL_EXACT, [FINAL_HEADLINE, FINAL_AFTER[FINAL_HEADLINE], FINAL_EF_20,
                               FINAL_AFTER[FINAL_EF_20]]
    )
    data = {
        "created_at": utc_now(),
        "configuration": FINAL.to_dict(),
        "runs": [run_summary_fields(out_dir, name) for name in FINAL_RUNS],
        "comparisons": comparisons,
        "per_k": per_k,
    }
    write_json(out_dir / "final_restart_comparison.json", data)
    (out_dir / "final_restart_comparison.md").write_text(
        comparison_markdown(
            "Final configuration: hit lists before and after a container restart (M=16)",
            (f"Configuration: {FINAL.label}. `final_ef_100` / `final_ef_20` ran before "
             "`docker compose restart`, `*_after_restart` after it, on the same on-disk index; "
             "`exact_vs_approximate` compares each with the exact ranking `final_inline`. A claim "
             "counts as identical when its 20 hit chunk ids are the same, in the same order."),
            data,
        ),
        encoding="utf-8",
    )
    log.info("wrote final_restart_comparison.json/.md")


# ---------------------------------------------------------------------------------------------
# group 8: restore the default state
# ---------------------------------------------------------------------------------------------


def group_restore(ctx: Context) -> None:
    info = ingest_configuration(ctx, FINAL, label="restore_defaults")
    cache = ensure_cache_size(ctx, CACHE_512MB)
    meta, index = ctx.database_state()
    expected = {
        **FINAL.ingest_meta_expected(),
        "embedding_model": ctx.settings.embedding_model,
        "embedding_dim": str(ctx.settings.vector_dim),
    }
    checks = [
        {"item": f"ingest_meta.{key}", "expected": value, "found": meta.get(key),
         "ok": meta.get(key) == value}
        for key, value in expected.items()
    ]
    checks += [
        {"item": "vector index M (SHOW CREATE TABLE chunk)", "expected": FINAL.index_m,
         "found": index.get("m"), "ok": index.get("m") == FINAL.index_m},
        {"item": "vector index DISTANCE", "expected": INDEX_DISTANCE, "found": index.get("distance"),
         "ok": index.get("distance") == INDEX_DISTANCE},
        {"item": "@@GLOBAL.mhnsw_max_cache_size", "expected": CACHE_512MB, "found": cache,
         "ok": cache == CACHE_512MB},
    ]
    final_meta = (load_json(ctx.out_dir / f"{FINAL_HEADLINE}.json") or {}).get("ingest_meta")
    if final_meta is not None:
        volatile = {"ingested_at", "analyze_seconds"}
        differences = {
            key: {"final": value, "now": meta.get(key)}
            for key, value in final_meta.items()
            if key not in volatile and meta.get(key) != value
        }
        checks.append({
            "item": f"ingest_meta equals {FINAL_HEADLINE}'s (ingested_at, analyze_seconds excepted)",
            "expected": "identical", "found": differences or "identical", "ok": not differences,
        })
    data = {
        "created_at": utc_now(),
        "configuration": FINAL.to_dict(),
        "ingest_run": info,
        "ingest_meta": meta,
        "vector_index": index,
        "mhnsw_max_cache_size": cache,
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks),
    }
    write_json(ctx.out_dir / f"{RESTORE_STATE}.json", data)
    parts = [
        "# Restored default state",
        "",
        (f"Generated {data['created_at']}. A fresh `run_ingest` with the defaults "
         f"('{FINAL.label}', what `wikilense ingest` produces), then `ingest_meta`, the index and "
         "the HNSW cache size checked."),
        "",
        md_table(("check", "expected", "found", "ok"),
                 [(c["item"], c["expected"], c["found"], "yes" if c["ok"] else "NO") for c in checks]),
        "",
        (f"Ingest: {info['counts']['n_chunks']} chunks, {info['n_units_hatnote']} hatnote units "
         f"excluded, {info['seconds']['total']} s (embed {info['seconds']['embed']} s); vector "
         f"index tablespace {info['vector_index_tablespace_bytes']} bytes."),
        "",
    ]
    (ctx.out_dir / f"{RESTORE_STATE}.md").write_text("\n".join(parts), encoding="utf-8")
    log.info("restore checks: %s", "all ok" if data["all_ok"] else [c for c in checks if not c["ok"]])
    if not data["all_ok"]:
        raise ExperimentError(f"the restored state is not the default: {[c for c in checks if not c['ok']]}")


# ---------------------------------------------------------------------------------------------
# group 9: SUMMARY.md
# ---------------------------------------------------------------------------------------------


class Results:
    """Reads the result JSON files of one directory; a missing file gives None / "not found"."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self._cache: dict[str, Any] = {}

    def get(self, name: str) -> dict[str, Any] | None:
        if name not in self._cache:
            self._cache[name] = load_json(self.out_dir / f"{name}.json")
        return self._cache[name]

    def has(self, *names: str) -> bool:
        return all(self.get(name) is not None for name in names)

    def per_k(self, name: str) -> dict[int, dict[str, Any]]:
        data = self.get(name)
        if data is None:
            return {}
        return {int(m["k"]): m for m in data["per_k"]}

    def art(self, name: str, k: int) -> str:
        m = self.per_k(name).get(k)
        data = self.get(name)
        return f"{m['article_hits']}/{data['n_claims']}" if m else "not found"

    def ev(self, name: str, k: int) -> str:
        m = self.per_k(name).get(k)
        data = self.get(name)
        return f"{m['evidence_hits']}/{data['n_evidence_claims']}" if m else "not found"

    def cov(self, name: str, k: int) -> str:
        m = self.per_k(name).get(k)
        return f"{m['unit_coverage']:.3f}" if m else "not found"

    def p50(self, name: str, k: int) -> str:
        m = self.per_k(name).get(k)
        return f"{m['sql_latency']['p50_ms']:.2f}" if m else "not found"

    def p95(self, name: str, k: int) -> str:
        m = self.per_k(name).get(k)
        return f"{m['sql_latency']['p95_ms']:.2f}" if m else "not found"

    def lat(self, name: str, k: int) -> str:
        return f"{self.p50(name, k)} / {self.p95(name, k)}"

    def emb_p50(self, name: str) -> str:
        data = self.get(name)
        return f"{data['embedding_latency']['p50_ms']:.2f}" if data else "not found"

    def art_n(self, name: str, k: int) -> int | None:
        m = self.per_k(name).get(k)
        return int(m["article_hits"]) if m else None

    def ev_n(self, name: str, k: int) -> int | None:
        m = self.per_k(name).get(k)
        return int(m["evidence_hits"]) if m else None

    def param(self, name: str, key: str, default: Any = "not found") -> Any:
        data = self.get(name)
        if data is None:
            return default
        return data["parameters"].get(key, default)

    def unstable(self, name: str) -> Any:
        data = self.get(name)
        return len(data["unstable_claims"]) if data else "not found"

    def identical(self, reference: str, other: str) -> str:
        """Return "n/75": claims whose max-k hit lists are identical between the two runs."""
        if not self.has(reference, other):
            return "not found"
        comparison = compare_hits(self.out_dir, reference, [other])
        return f"{comparison['runs'][other]['identical_sequence']}/{comparison['n_claims']}"

    def ingest_run(self, label: str) -> dict[str, Any] | None:
        runs = self.get(INGEST_RUNS) or []
        for run in reversed(runs):
            if run.get("label") == label and run.get("kind") == "ingest":
                return run
        return None

    def rebuild(self, label: str) -> dict[str, Any] | None:
        runs = self.get(INGEST_RUNS) or []
        for run in reversed(runs):
            if run.get("label") == label and run.get("kind") == "rebuild":
                return run
        return None


def fmt_bytes_mb(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "not found"
    return f"{value / 1048576:.1f} MB"


def fmt_seconds(value: Any) -> str:
    return f"{value:.2f} s" if isinstance(value, (int, float)) else "not found"


def cmp_word(a: int | None, b: int | None) -> str:
    """Return "equal to", "above" or "below" for ``a`` against ``b`` ("not found" without values)."""
    if a is None or b is None:
        return "not found against"
    if a == b:
        return "equal to"
    return "above" if a > b else "below"


def summary_markdown(out_dir: Path) -> str:
    r = Results(out_dir)
    header = r.get(FINAL_HEADLINE) or r.get("ef_100") or r.get(STABILITY_512_RUNS[0])
    n_claims = header["n_claims"] if header else "?"
    n_evidence = header["n_evidence_claims"] if header else "?"
    intro = (
        f"Generated {utc_now()} by `scripts/run_experiments.py summary`. Every number below is read "
        "from the JSON files in `results/` written by the same script (`write_results` of the "
        "evaluation harness); the per-run Markdown files hold the full tables. "
        f"{n_claims} FEVEROUS claims over the 100-page corpus; article recall counts the claims "
        f"with a gold page among the pages of the top-k chunks (of {n_claims}), evidence recall "
        "the claims whose gold sentences are all inside the top-k chunks (of the "
        f"{n_evidence} claims with a sentence-only evidence set), unit coverage the mean share "
        "of a claim's gold units inside the top-k chunks. Latencies are the SQL time of one "
        "`search()` call in milliseconds, p50 / p95 over 5 repeats x the claims, warm, query "
        "embedding excluded (reported separately). `ef` is `mhnsw_ef_search`, set per session "
        "by the application; the vector index is the HNSW index of `sql/schema.sql` with cosine "
        "distance, its M as stated per section. Two ingest configurations are used: OLD = "
        f"{OLD.label} (phases 1 and 2) and FINAL = {FINAL.label} (the chosen defaults); hatnote "
        "units (\"Main article: ...\") are in no chunk in either. \"Identical to exact\" counts "
        "the claims whose 20 hit chunk ids equal, in order, those of the exact ranking (the "
        "inline statement without filters, which does not depend on the HNSW graph)."
    )
    parts: list[str] = ["# WikiLense experiments: summary", "", intro, ""]
    if header is not None:
        parts += [
            "## Machine and versions",
            "",
            md_table(("Item", "Value"), [*header["machine"].items(), *header["versions"].items()]),
            "",
        ]
    parts += summary_stability(r)
    parts += summary_ef_search(r)
    parts += summary_index_m(r)
    parts += summary_chunk_size(r)
    parts += summary_prefix(r)
    parts += summary_filters(r)
    parts += summary_final(r)
    parts += summary_restore(r)
    parts += summary_defaults(r)
    return "\n".join(parts) + "\n"


def summary_stability(r: Results) -> list[str]:
    runs = [*STABILITY_512_RUNS, AFTER_RESTART, *STABILITY_16_RUNS]
    rows = []
    for name in runs:
        rows.append([
            name, fmt_bytes_mb(r.param(name, "mhnsw_max_cache_size", None)),
            r.param(name, "ef_search_effective"), r.art(name, 1), r.art(name, 10), r.art(name, 20),
            r.ev(name, 10), r.ev(name, 20), r.lat(name, 10), r.unstable(name),
            r.identical(STABILITY_512_RUNS[0], name), r.identical(AFTER_RESTART, name),
        ])
    parts = [
        f"## 1. Stability: the same run repeated (OLD: {OLD.label}; strategy none, ef 20)",
        "",
        ("One fresh ingest, then seven identical evaluations on the same on-disk index: three "
        "with the 512 MB HNSW cache, one after `docker compose restart`, three after "
        "`SET GLOBAL mhnsw_max_cache_size = 16777216` (536870912 restored afterwards). "
        "\"unstable\" counts the claims whose LIMIT-20 hits changed between the five repeats of "
        "one run; the last two columns count the claims whose 20 hit chunk ids equal, in order, "
        "those of `stability_512mb_run1` and of `stability_512mb_after_restart` "
        "(`stability_comparison.md` lists the claims)."),
        "",
        md_table(
            ("run", "cache", "ef", "article@1", "article@10", "article@20", "evidence@10",
             "evidence@20", "SQL p50 / p95 @10 ms", "unstable", "identical to 512mb_run1",
             "identical to after_restart"),
            rows,
        ),
        "",
    ]
    comp = r.get("stability_comparison")
    if comp is None or not r.has(*runs):
        parts += ["Not every stability file exists yet; the reading is written once they do.", ""]
        return parts
    c = comp["comparisons"]
    n = c["within_512mb"]["n_claims"]
    within512 = c["within_512mb"]["identical_in_all"]
    within16 = c["within_16mb"]["identical_in_all"]
    set_global = c["after_restart_vs_16mb"]["runs"][STABILITY_16_RUNS[0]]["identical_sequence"]
    restart = c["after_restart_vs_512mb"]["runs"][AFTER_RESTART]["identical_sequence"]
    lost = comp["gold_page_lost_after_restart"]
    gained = comp["gold_page_gained_after_restart"]
    deterministic = within512 == n and within16 == n
    restart_phrase = (
        "The container restart changed nothing either" if restart == n
        else "The container restart is what changed the results"
    )
    reading = (
        f"Reading: within one server process the search over this M=6 graph is "
        f"{'deterministic' if deterministic else 'not fully deterministic'}: the three 512 MB runs "
        f"return the same 20 chunks in the same order for {within512} of {n} claims, the three "
        f"16 MB runs for {within16} of {n}, and lowering the cache from 512 MB to 16 MB with SET "
        f"GLOBAL left {set_global} of {n} hit lists unchanged. {restart_phrase}: the run after it "
        f"matches the run before it for {restart} of {n} "
        f"claims, {len(lost)} claims lost their gold page from the top 20"
        f"{' (' + ', '.join(map(str, lost)) + ')' if lost else ''} and {len(gained)} gained one, "
        f"and article recall@10 went from {r.art(STABILITY_512_RUNS[0], 10)} to "
        f"{r.art(AFTER_RESTART, 10)} (evidence recall@20 {r.ev(STABILITY_512_RUNS[0], 20)} to "
        f"{r.ev(AFTER_RESTART, 20)}). SQL p50@10 was {r.p50(STABILITY_512_RUNS[0], 10)} ms before "
        f"and {r.p50(AFTER_RESTART, 10)} ms after the restart, {r.p50(STABILITY_16_RUNS[0], 10)} ms "
        f"with the 16 MB cache."
    )
    parts += [reading, ""]
    return parts


def summary_ef_search(r: Results) -> list[str]:
    names = [(f"ef_{ef}", "none", ef) for ef in EF_SWEEP]
    names += [(f"inline_ef_{ef}", "inline (exact)", ef) for ef in (EF_LOW, EF_CHOSEN)]
    names += [(RRF_OLD, f"rrf (overfetch {OVERFETCH_DEFAULT})", EF_CHOSEN)]
    rows = [
        [n, s, r.param(n, "ef_search_effective"), r.art(n, 1), r.art(n, 5), r.art(n, 10),
         r.art(n, 20), r.ev(n, 5), r.ev(n, 10), r.ev(n, 20), r.cov(n, 10), r.lat(n, 10),
         r.identical(EXACT_OLD, n) if n != RRF_OLD else "- (other order by construction)"]
        for n, s, _ in names
    ]
    parts = [
        f"## 2. mhnsw_ef_search sweep (OLD: {OLD.label}; the index state after the restart of section 1)",
        "",
        ("`ef_<n>` is strategy none (the bare `ORDER BY VEC_DISTANCE_COSINE ... LIMIT k`, the HNSW "
        "search proper); `inline_ef_<n>` the joined statement without filters, which ranks "
        "exactly over the index and so does not depend on `ef`; `rrf_120_m6` the hybrid "
        f"(vector top-{10 * OVERFETCH_DEFAULT} at ef 100 fused with the full-text top-"
        f"{10 * OVERFETCH_DEFAULT} of the claim words by reciprocal rank fusion). The reference "
        f"of the last column is `{EXACT_OLD}`."),
        "",
        md_table(
            ("run", "strategy", "ef", "article@1", "article@5", "article@10", "article@20",
             "evidence@5", "evidence@10", "evidence@20", "coverage@10", "SQL p50 / p95 @10 ms",
             "identical to exact"),
            rows,
        ),
        "",
    ]
    if not r.has(*[n for n, _, _ in names]):
        parts += ["Not every file of this group exists yet; the reading is written once they do.", ""]
        return parts
    a = {ef: r.art(f"ef_{ef}", 10) for ef in EF_SWEEP}
    e = {ef: r.ev(f"ef_{ef}", 20) for ef in EF_SWEEP}
    p = {ef: r.p50(f"ef_{ef}", 10) for ef in EF_SWEEP}
    ident = {ef: r.identical(EXACT_OLD, f"ef_{ef}") for ef in EF_SWEEP}
    reading = (
        f"Reading: raising ef from 20 to 400 moves article recall@10 from {a[20]} to {a[400]} and "
        f"evidence recall@20 from {e[20]} to {e[400]} for SQL p50@10 {p[20]} against {p[400]} ms; "
        f"ef 100 gives {a[100]} / {e[100]} at {p[100]} ms. The exact ranking (inline) gives "
        f"{r.art(EXACT_OLD, 10)} / {r.ev(EXACT_OLD, 20)} at p50@10 {r.p50(EXACT_OLD, 10)} ms, with "
        f"hit lists identical at ef 20 and ef 100 for {r.identical(EXACT_OLD, f'inline_ef_{EF_CHOSEN}')} "
        f"claims; the approximate hit lists are identical to it for "
        f"{' / '.join(ident[ef] for ef in EF_SWEEP)} claims at ef {' / '.join(map(str, EF_SWEEP))}. "
        f"The rrf hybrid at ef 100 gives article recall@10 {r.art(RRF_OLD, 10)} and evidence "
        f"recall@20 {r.ev(RRF_OLD, 20)} against {a[100]} / {e[100]} for the vector ranking alone "
        f"at the same ef, at p50@10 {r.p50(RRF_OLD, 10)} ms against {p[100]} ms."
    )
    parts += [reading, ""]
    return parts


def summary_index_m(r: Results) -> list[str]:
    rebuild = r.get("index_m_rebuild")
    rows = []
    for m in INDEX_M_VALUES:
        rb = r.rebuild(f"index_m_sweep_m{m}") or {}
        for ef in INDEX_M_EFS:
            n = f"m{m}_ef_{ef}"
            rows.append([
                n, r.param(n, "vector_index_m"), r.param(n, "ef_search_effective"),
                fmt_seconds(rb.get("drop_seconds")), fmt_seconds(rb.get("add_seconds")),
                fmt_bytes_mb(r.param(n, "vector_index_tablespace_bytes", None)),
                r.art(n, 1), r.art(n, 10), r.art(n, 20), r.ev(n, 10), r.ev(n, 20), r.lat(n, 10),
                r.identical(EXACT_OLD, n),
            ])
    rows.append([
        EXACT_OLD, r.param(EXACT_OLD, "vector_index_m"), "-", "-", "-",
        fmt_bytes_mb(r.param(EXACT_OLD, "vector_index_tablespace_bytes", None)),
        r.art(EXACT_OLD, 1), r.art(EXACT_OLD, 10), r.art(EXACT_OLD, 20), r.ev(EXACT_OLD, 10),
        r.ev(EXACT_OLD, 20), r.lat(EXACT_OLD, 10), r.identical(EXACT_OLD, EXACT_OLD),
    ])
    parts = [
        f"## 3. Index M sweep (OLD vectors: {OLD.label.replace('M=6, ', '')}; strategy none)",
        "",
        ("The vector index rebuilt with `ALTER TABLE chunk DROP INDEX` + `ADD VECTOR INDEX ... "
        "M=n DISTANCE=cosine` for M 6, 16 and 32 (the drop and add timed separately), each "
        "evaluated at ef 20 and 100. `graph tablespace` is the file size of the hidden InnoDB "
        "table `chunk#i#NN` that holds the HNSW graph; the exact ranking `inline_ef_20` is the "
        "last row for reference."),
        "",
        md_table(
            ("run", "M", "ef", "DROP INDEX", "ADD VECTOR INDEX", "graph tablespace", "article@1",
             "article@10", "article@20", "evidence@10", "evidence@20", "SQL p50 / p95 @10 ms",
             "identical to exact"),
            rows,
        ),
        "",
    ]
    names = [f"m{m}_ef_{ef}" for m in INDEX_M_VALUES for ef in INDEX_M_EFS]
    if rebuild is None or not r.has(*names, EXACT_OLD):
        parts += ["Not every file of this group exists yet; the reading is written once they do.", ""]
        return parts
    add = {m: (r.rebuild(f"index_m_sweep_m{m}") or {}).get("add_seconds") for m in INDEX_M_VALUES}
    drop = {m: (r.rebuild(f"index_m_sweep_m{m}") or {}).get("drop_seconds") for m in INDEX_M_VALUES}
    size = {m: fmt_bytes_mb(r.param(f"m{m}_ef_20", "vector_index_tablespace_bytes", None))
            for m in INDEX_M_VALUES}
    ms = " / ".join(str(m) for m in INDEX_M_VALUES)

    def per_m(fn: Any, ef: int) -> str:
        return " / ".join(str(fn(f"m{m}_ef_{ef}")) for m in INDEX_M_VALUES)

    same_vectors = r.identical(f"ef_{EF_LOW}", f"m6_ef_{EF_LOW}")
    reading = (
        f"Reading: ADD VECTOR INDEX took {' / '.join(str(add[m]) for m in INDEX_M_VALUES)} s for "
        f"M {ms} (DROP INDEX {' / '.join(str(drop[m]) for m in INDEX_M_VALUES)} s) and the graph "
        f"tablespace is {' / '.join(size[m] for m in INDEX_M_VALUES)}. At ef 20 article recall@10 "
        f"is {per_m(lambda n: r.art(n, 10), 20)} and evidence recall@20 "
        f"{per_m(lambda n: r.ev(n, 20), 20)} against {r.art(EXACT_OLD, 10)} / {r.ev(EXACT_OLD, 20)} "
        f"for the exact ranking; at ef 100 {per_m(lambda n: r.art(n, 10), 100)} and "
        f"{per_m(lambda n: r.ev(n, 20), 100)}. Hit lists identical to the exact ranking: "
        f"{per_m(lambda n: r.identical(EXACT_OLD, n), 20)} at ef 20 and "
        f"{per_m(lambda n: r.identical(EXACT_OLD, n), 100)} at ef 100, for SQL p50@10 "
        f"{per_m(lambda n: r.p50(n, 10), 20)} ms at ef 20 and {per_m(lambda n: r.p50(n, 10), 100)} "
        f"ms at ef 100. The M=6 graph built here and the M=6 graph of section 2 (the same "
        f"vectors, `ef_20`) give identical hit lists for {same_vectors} claims: every build of "
        f"an M=6 graph is another approximation."
    )
    parts += [reading, ""]
    return parts


CHUNK_ALIGNMENT: tuple[tuple[int, tuple[tuple[str, int] | None, ...]], ...] = (
    (2400, (("chunk60_ef_100", 40), ("chunk120_ef_100", 20), ("chunk240_ef_100", 10))),
    (1200, (("chunk60_ef_100", 20), ("chunk120_ef_100", 10), ("chunk240_ef_100", 5))),
    (720, (None, None, ("chunk240_ef_100", 3))),
    (600, (("chunk60_ef_100", 10), ("chunk120_ef_100", 5), None)),
    (480, (None, None, ("chunk240_ef_100", 2))),
    (360, (("chunk60_ef_100", 6), ("chunk120_ef_100", 3), None)),
    (240, (None, None, ("chunk240_ef_100", 1))),
    (120, (("chunk60_ef_100", 2), ("chunk120_ef_100", 1), None)),
)


def summary_chunk_size(r: Results) -> list[str]:
    cases = [(name, words, f"chunk{words}") for words, _, name in CHUNK_SIZE_CASES]
    rows: list[list[Any]] = []

    def per_case(fn: Any) -> list[Any]:
        return [fn(name, label) for name, _, label in cases]

    rows.append(["chunks", *per_case(lambda n, lb: r.param(n, "n_chunks"))])
    rows.append(["hatnote units excluded", *per_case(lambda n, lb: r.param(n, "n_units_hatnote"))])
    rows.append(["words mean / median / p95 / max", *per_case(
        lambda n, lb: f"{r.param(n, 'chunk_words_mean')} / {r.param(n, 'chunk_words_median')} / "
                      f"{r.param(n, 'chunk_words_p95')} / {r.param(n, 'chunk_words_max')}")])
    rows.append(["ingest total s (embed s)", *per_case(
        lambda n, lb: (f"{r.ingest_run(lb)['seconds']['total']} ({r.ingest_run(lb)['seconds']['embed']})"
                       if r.ingest_run(lb) else "not found"))])
    rows.append(["graph tablespace", *per_case(
        lambda n, lb: fmt_bytes_mb(r.param(n, "vector_index_tablespace_bytes", None)))])
    for words, cells in CHUNK_ALIGNMENT:
        row: list[Any] = [f"{words:,} nominal words"]
        for cell in cells:
            if cell is None:
                row.append("-")
                continue
            name, k = cell
            mean = r.param(name, "chunk_words_mean", None)
            actual = f"{k * mean:.0f}" if isinstance(mean, (int, float)) else "?"
            row.append(
                f"k={k} ({actual} words): {r.art(name, k)} / {r.ev(name, k)} / {r.cov(name, k)} / "
                f"{r.p50(name, k)} ms"
            )
        rows.append(row)
    parts = [
        "## 4. Chunk size at equal retrieved text (60 / 120 / 240 words, overlap 1, M=16, prefix on; strategy none, ef 100)",
        "",
        ("One fresh ingest per chunk size, each with its own M=16 graph built by the schema. The "
        "lower rows align the ks by nominal retrieved words (chunk_max_words x k); the words in "
        "brackets are k x the mean chunk length actually stored. Each cell: article recall / "
        "evidence recall / unit coverage / SQL p50."),
        "",
        md_table(("", "60-word chunks", "120-word chunks", "240-word chunks"), rows),
        "",
    ]
    names = [name for name, _, _ in cases]
    if not r.has(*names):
        parts += ["Not every file of this group exists yet; the reading is written once they do.", ""]
        return parts
    ks1200 = (20, 10, 5)
    art1200 = [r.art_n(n, k) for n, k in zip(names, ks1200)]
    ev1200 = [r.ev_n(n, k) for n, k in zip(names, ks1200)]
    spread_art = max(art1200) - min(art1200) if None not in art1200 else "?"
    spread_ev = max(ev1200) - min(ev1200) if None not in ev1200 else "?"
    reading = (
        f"Reading: at 1,200 nominal words (k = 20 / 10 / 5) the 60 / 120 / 240-word chunkings give "
        f"article recall {' / '.join(r.art(n, k) for n, k in zip(names, ks1200))}, evidence recall "
        f"{' / '.join(r.ev(n, k) for n, k in zip(names, ks1200))} and unit coverage "
        f"{' / '.join(r.cov(n, k) for n, k in zip(names, ks1200))}, so the chunkings differ by at "
        f"most {spread_art} claims in article recall and {spread_ev} in evidence recall there; at "
        f"2,400 words (k = 40 / 20 / 10) {' / '.join(r.art(n, k) for n, k in zip(names, (40, 20, 10)))} "
        f"and {' / '.join(r.ev(n, k) for n, k in zip(names, (40, 20, 10)))}; at 600 words 60x10 gives "
        f"{r.art(names[0], 10)} / {r.ev(names[0], 10)} and 120x5 {r.art(names[1], 5)} / "
        f"{r.ev(names[1], 5)}, with 240x2 (480 words) at {r.art(names[2], 2)} / {r.ev(names[2], 2)} "
        f"and 240x3 (720 words) at {r.art(names[2], 3)} / {r.ev(names[2], 3)}. The cost side: "
        f"{' / '.join(str(r.param(n, 'n_chunks')) for n in names)} chunks, a graph of "
        f"{' / '.join(fmt_bytes_mb(r.param(n, 'vector_index_tablespace_bytes', None)) for n in names)}, "
        f"ingest {' / '.join(str((r.ingest_run(lb) or {}).get('seconds', {}).get('total', '?')) for _, _, lb in cases)} s, "
        f"and SQL p50 {' / '.join(r.p50(n, k) for n, k in zip(names, ks1200))} ms for the 1,200-word "
        f"row. The 240-word chunks retrieve fewer actual words at equal nominal text (mean chunk "
        f"{r.param(names[2], 'chunk_words_mean')} words against {r.param(names[1], 'chunk_words_mean')} "
        f"and {r.param(names[0], 'chunk_words_mean')}) because a chunk never crosses a section "
        f"boundary and many sections are short."
    )
    parts += [reading, ""]
    return parts


def summary_prefix(r: Results) -> list[str]:
    names = [
        ("prefix_on_ef_100", "on", "none", 100, "prefix_on_inline"),
        ("prefix_on_inline", "on", "inline (exact)", 100, "prefix_on_inline"),
        ("prefix_off_ef_100", "off", "none", 100, "prefix_off_inline"),
        ("prefix_off_inline", "off", "inline (exact)", 100, "prefix_off_inline"),
    ]
    rows = [
        [n, p, s, r.param(n, "ef_search_effective"), r.art(n, 1), r.art(n, 5), r.art(n, 10),
         r.art(n, 20), r.ev(n, 5), r.ev(n, 10), r.ev(n, 20), r.cov(n, 10), r.lat(n, 10),
         r.identical(exact, n)]
        for n, p, s, _, exact in names
    ]
    parts = [
        "## 5. Embedding prefix \"title > section path: \" on / off (120 words, overlap 1, M=16; ef 100)",
        "",
        ("One fresh ingest with the prefix and one without, each evaluated approximately "
        "(strategy none, ef 100) and exactly (inline, no filters). The last column compares "
        "each run with the exact ranking of its own vectors."),
        "",
        md_table(
            ("run", "prefix", "strategy", "ef", "article@1", "article@5", "article@10",
             "article@20", "evidence@5", "evidence@10", "evidence@20", "coverage@10",
             "SQL p50 / p95 @10 ms", "identical to own exact"),
            rows,
        ),
        "",
    ]
    if not r.has(*[n for n, *_ in names]):
        parts += ["Not every file of this group exists yet; the reading is written once they do.", ""]
        return parts
    on, off = "prefix_on_inline", "prefix_off_inline"
    reading = (
        f"Reading: the exact rankings give, with the prefix, article recall@1 / @10 {r.art(on, 1)} / "
        f"{r.art(on, 10)} and evidence recall@10 / @20 {r.ev(on, 10)} / {r.ev(on, 20)} (unit "
        f"coverage@10 {r.cov(on, 10)}); without it {r.art(off, 1)} / {r.art(off, 10)} and "
        f"{r.ev(off, 10)} / {r.ev(off, 20)} ({r.cov(off, 10)}). So the exact evidence recall@20 with "
        f"the prefix is {cmp_word(r.ev_n(on, 20), r.ev_n(off, 20))} the one without, by "
        f"{abs((r.ev_n(on, 20) or 0) - (r.ev_n(off, 20) or 0))} claims, and article recall@10 is "
        f"{cmp_word(r.art_n(on, 10), r.art_n(off, 10))} it. The approximate runs (M=16, ef 100) "
        f"give {r.art('prefix_on_ef_100', 10)} / {r.ev('prefix_on_ef_100', 20)} with and "
        f"{r.art('prefix_off_ef_100', 10)} / {r.ev('prefix_off_ef_100', 20)} without the prefix, "
        f"with hit lists identical to their exact ranking for "
        f"{r.identical(on, 'prefix_on_ef_100')} and {r.identical(off, 'prefix_off_ef_100')} claims."
    )
    parts += [reading, ""]
    return parts


def summary_filters(r: Results) -> list[str]:
    rows = []
    truth_by_filter: dict[str, Any] = {}
    for filter_name, _ in FILTER_CASES:
        for strategy_name, _, _ in STRATEGY_CASES:
            n = f"strategy_{strategy_name}_{filter_name}"
            data = r.get(n)
            if data is None:
                rows.append([n, filter_name, strategy_name] + ["not found"] * 12)
                continue
            truth = data["parameters"].get("filter_ground_truth") or {}
            truth_by_filter[filter_name] = truth
            short = data["parameters"].get("short_results", {})
            rows.append([
                n, filter_name, strategy_name,
                f"{truth.get('n_chunks_passing', '?')} of {truth.get('n_chunks', '?')}",
                f"{truth.get('article_recall_ceiling', '?')}/{truth.get('n_claims', '?')}",
                f"{truth.get('evidence_recall_ceiling', '?')}/{truth.get('n_evidence_claims', '?')}",
                r.art(n, 5), r.art(n, 10), r.art(n, 20), r.ev(n, 10), r.ev(n, 20),
                short.get("10", {}).get("queries_short_of_k", "-"),
                short.get("10", {}).get("queries_with_no_rows", "-"),
                short.get("10", {}).get("rows_mean", "-"),
                r.lat(n, 10),
            ])
    parts = [
        f"## 6. Filter strategies (FINAL: {FINAL.label}; ef 100)",
        "",
        ("`min_words >= 1000` (page length) and `heading LIKE '%History%'` (section heading), each "
        "with the inline statement (predicates joined into the index-driven statement), "
        "overfetch 10 and overfetch 50 (inner `LIMIT k x factor` by the index, filtered outside). "
        "`ceiling` is the best value a strategy can reach because the filter itself excludes the "
        "gold pages or gold sentences of some claims; `queries < 10 rows` / `no rows` count, over "
        "the claims, the k=10 queries that returned fewer than 10 rows or none (a separate "
        "untimed pass of `search()`)."),
        "",
        md_table(
            ("run", "filter", "strategy", "chunks passing", "article ceiling", "evidence ceiling",
             "article@5", "article@10", "article@20", "evidence@10", "evidence@20",
             "queries < 10 rows", "no rows @10", "rows mean @10", "SQL p50 / p95 @10 ms"),
            rows,
        ),
        "",
    ]
    names = [f"strategy_{s}_{f}" for f, _ in FILTER_CASES for s, _, _ in STRATEGY_CASES]
    if not r.has(*names) or len(truth_by_filter) < 2:
        parts += ["Not every file of this group exists yet; the reading is written once they do.", ""]
        return parts
    t_min = truth_by_filter["minwords1000"]
    t_his = truth_by_filter["history"]

    def short(name: str, k: int, key: str = "queries_short_of_k") -> Any:
        return r.get(name)["parameters"]["short_results"][str(k)][key]

    def trio(fn: Any, filter_name: str) -> str:
        return " / ".join(str(fn(f"strategy_{s}_{filter_name}")) for s, _, _ in STRATEGY_CASES)

    reading = (
        f"Reading: under min_words >= 1000 ({t_min['n_chunks_passing']} of {t_min['n_chunks']} "
        f"chunks pass; ceilings {t_min['article_recall_ceiling']}/{t_min['n_claims']} and "
        f"{t_min['evidence_recall_ceiling']}/{t_min['n_evidence_claims']}) inline / overfetch 10 / "
        f"overfetch 50 give article recall@10 {trio(lambda n: r.art(n, 10), 'minwords1000')} and "
        f"evidence recall@20 {trio(lambda n: r.ev(n, 20), 'minwords1000')} at SQL p50@10 "
        f"{trio(lambda n: r.p50(n, 10), 'minwords1000')} ms, with "
        f"{trio(lambda n: short(n, 10), 'minwords1000')} queries short of 10 rows. Under heading "
        f"LIKE '%History%' ({t_his['n_chunks_passing']} chunks pass; ceilings "
        f"{t_his['article_recall_ceiling']}/{t_his['n_claims']} and "
        f"{t_his['evidence_recall_ceiling']}/{t_his['n_evidence_claims']}) they give "
        f"{trio(lambda n: r.art(n, 10), 'history')} and {trio(lambda n: r.ev(n, 20), 'history')} at "
        f"{trio(lambda n: r.p50(n, 10), 'history')} ms, and overfetch 10 returned fewer than 10 rows "
        f"for {short('strategy_overfetch10_history', 10)} of {t_his['n_claims']} queries "
        f"({short('strategy_overfetch10_history', 10, 'queries_with_no_rows')} with no row), "
        f"overfetch 50 for {short('strategy_overfetch50_history', 10)}, inline for "
        f"{short('strategy_inline_history', 10)}. The inline statement's cost is the exact ranking "
        f"over the index plus the walk until k rows pass the predicates (p95@10 "
        f"{r.p95('strategy_inline_history', 10)} ms under the History filter against "
        f"{r.p95('strategy_inline_minwords1000', 10)} ms under min_words), so it grows with the "
        f"table and with the selectivity of the filter."
    )
    parts += [reading, ""]
    return parts


def summary_final(r: Results) -> list[str]:
    h = FINAL_HEADLINE
    parts = [
        f"## 7. Final configuration ({FINAL.label}; the fresh ingest of the defaults)",
        "",
        (f"Headline run `{h}`: strategy none (the bare index query joined back to `page` and "
        f"`section`), ef 100, {r.param(h, 'n_chunks')} chunks ({r.param(h, 'n_units_hatnote')} "
        "hatnote units excluded), M=16, 512 MB cache. Query embedding p50 is the time of one "
        "`embed_queries([claim])` call on the GPU and does not depend on k."),
        "",
        md_table(
            ("k", "article recall", "evidence recall", "unit coverage", "SQL p50 ms", "SQL p95 ms",
             "query embedding p50 ms"),
            [[k, f"{r.art(h, k)} ({(r.art_n(h, k) or 0) / (r.get(h) or {}).get('n_claims', 1):.3f})"
              if r.get(h) else "not found",
              f"{r.ev(h, k)} ({(r.ev_n(h, k) or 0) / (r.get(h) or {}).get('n_evidence_claims', 1):.3f})"
              if r.get(h) else "not found",
              r.cov(h, k), r.p50(h, k), r.p95(h, k), r.emb_p50(h)]
             for k in DEFAULT_KS],
        ),
        "",
        ("The other runs of the group on the same ingest (`final_inline` is the exact ranking and "
        "the reference of the last column; `final_oracle_titles` restricts the inline statement "
        "to the claim's gold page with the `titles` filter, so its evidence recall says at which "
        "rank the gold sentences surface once the page is known; the `*_after_restart` runs "
        "repeat the two approximate runs after `docker compose restart`):"),
        "",
    ]
    runs = [
        (h, "none, ef 100", True), (FINAL_EF_20, "none, ef 20", True),
        (FINAL_EXACT, "inline (exact), ef 100", True),
        (FINAL_RRF, f"rrf (overfetch {OVERFETCH_DEFAULT}), ef 100", False),
        (FINAL_ORACLE, "inline + gold page as titles filter, ef 100", False),
        (FINAL_AFTER[h], "none, ef 100, after restart", True),
        (FINAL_AFTER[FINAL_EF_20], "none, ef 20, after restart", True),
    ]
    parts += [
        md_table(
            ("run", "strategy", "article@1", "article@5", "article@10", "article@20", "evidence@1",
             "evidence@5", "evidence@10", "evidence@20", "coverage@10", "SQL p50 / p95 @10 ms",
             "identical to exact"),
            [[n, s, r.art(n, 1), r.art(n, 5), r.art(n, 10), r.art(n, 20), r.ev(n, 1), r.ev(n, 5),
              r.ev(n, 10), r.ev(n, 20), r.cov(n, 10), r.lat(n, 10),
              r.identical(FINAL_EXACT, n) if comparable else "- (other order by construction)"]
             for n, s, comparable in runs],
        ),
        "",
    ]
    comp = r.get("final_restart_comparison")
    if comp is not None:
        rows = []
        for key in (f"{h}_vs_{FINAL_AFTER[h]}", f"{FINAL_EF_20}_vs_{FINAL_AFTER[FINAL_EF_20]}"):
            c = comp["comparisons"].get(key)
            if c is None:
                continue
            before = c["reference"]
            per_k = comp["per_k"].get(before, {})
            for name, run in c["runs"].items():
                rows.append([
                    before, name, f"{run['identical_sequence']}/{c['n_claims']}",
                    f"{run['identical_set']}/{c['n_claims']}",
                    f"{r.art(before, 10)} -> {r.art(name, 10)}",
                    f"{r.ev(before, 20)} -> {r.ev(name, 20)}",
                    len(per_k.get("gold_page_lost_after_restart", [])),
                    len(per_k.get("gold_page_gained_after_restart", [])),
                    ", ".join(map(str, run["differing_claims"])) or "none",
                ])
        parts += [
            "Restart comparison (`final_restart_comparison.md`), 20 hit chunk ids per claim:",
            "",
            md_table(
                ("before", "after", "identical (sequence)", "identical (set)", "article@10",
                 "evidence@20", "gold page lost", "gold page gained", "claims that differ"),
                rows,
            ),
            "",
        ]
    if not r.has(*FINAL_RUNS, "final_restart_comparison"):
        parts += ["Not every file of this group exists yet; the reading is written once they do.", ""]
        return parts
    c100 = comp["comparisons"][f"{h}_vs_{FINAL_AFTER[h]}"]["runs"][FINAL_AFTER[h]]["identical_sequence"]
    c20 = comp["comparisons"][f"{FINAL_EF_20}_vs_{FINAL_AFTER[FINAL_EF_20]}"]["runs"][
        FINAL_AFTER[FINAL_EF_20]]["identical_sequence"]
    n = comp["comparisons"][f"{h}_vs_{FINAL_AFTER[h]}"]["n_claims"]
    reading = (
        f"Reading: the headline run reaches article recall {r.art(h, 1)} / {r.art(h, 5)} / "
        f"{r.art(h, 10)} / {r.art(h, 20)} at k 1 / 5 / 10 / 20 and evidence recall {r.ev(h, 1)} / "
        f"{r.ev(h, 5)} / {r.ev(h, 10)} / {r.ev(h, 20)}, at SQL p50 / p95 {r.lat(h, 10)} ms for k=10 "
        f"and query embedding p50 {r.emb_p50(h)} ms; its article recall@10 is "
        f"{cmp_word(r.art_n(h, 10), r.art_n(FINAL_EXACT, 10))} the exact ranking's "
        f"({r.art(FINAL_EXACT, 10)}) and its evidence recall@20 "
        f"{cmp_word(r.ev_n(h, 20), r.ev_n(FINAL_EXACT, 20))} it ({r.ev(FINAL_EXACT, 20)}), with hit "
        f"lists identical to the exact ones for {r.identical(FINAL_EXACT, h)} claims "
        f"({r.identical(FINAL_EXACT, FINAL_EF_20)} at ef 20) where the exact statement costs "
        f"{r.p50(FINAL_EXACT, 10)} ms p50. The rrf hybrid gives article recall@10 "
        f"{r.art(FINAL_RRF, 10)} and evidence recall@20 {r.ev(FINAL_RRF, 20)} at {r.p50(FINAL_RRF, 10)} "
        f"ms p50, {cmp_word(r.art_n(FINAL_RRF, 10), r.art_n(h, 10))} the vector ranking alone in "
        f"article recall@10 and {cmp_word(r.ev_n(FINAL_RRF, 20), r.ev_n(h, 20))} it in evidence "
        f"recall@20; with the gold page known (oracle) the gold sentences are inside the first chunk "
        f"for {r.ev(FINAL_ORACLE, 1)} claims, within 5 for {r.ev(FINAL_ORACLE, 5)} and within 20 for "
        f"{r.ev(FINAL_ORACLE, 20)}, at {r.lat(FINAL_ORACLE, 10)} ms. After the container restart the "
        f"hit lists at ef 100 are identical to the pre-restart ones for {c100} of {n} claims "
        f"(article recall@10 {r.art(h, 10)} -> {r.art(FINAL_AFTER[h], 10)}, evidence recall@20 "
        f"{r.ev(h, 20)} -> {r.ev(FINAL_AFTER[h], 20)}) and at ef 20 for {c20} of {n} "
        f"({r.art(FINAL_EF_20, 10)} -> {r.art(FINAL_AFTER[FINAL_EF_20], 10)}, {r.ev(FINAL_EF_20, 20)} "
        f"-> {r.ev(FINAL_AFTER[FINAL_EF_20], 20)}), against {r.identical(STABILITY_512_RUNS[0], AFTER_RESTART)} "
        f"identical hit lists across the restart of the M=6 index in section 1."
    )
    parts += [reading, ""]
    return parts


def summary_restore(r: Results) -> list[str]:
    state = r.get(RESTORE_STATE)
    parts = [f"## 8. Restored default state ({FINAL.label}; a fresh `run_ingest` with the defaults)", ""]
    if state is None:
        parts += ["`restore_state.json` not found.", ""]
        return parts
    parts += [
        md_table(("check", "expected", "found", "ok"),
                 [(c["item"], c["expected"], c["found"], "yes" if c["ok"] else "NO")
                  for c in state["checks"]]),
        "",
    ]
    info = state["ingest_run"]
    reading = (
        f"Reading: the last ingest ({info['counts']['n_chunks']} chunks, {info['n_units_hatnote']} "
        f"hatnote units excluded, {info['seconds']['total']} s of which {info['seconds']['embed']} s "
        f"embedding) {'passed every check' if state['all_ok'] else 'FAILED a check'}: `ingest_meta` "
        f"holds the defaults, the vector index is M={state['vector_index'].get('m')} "
        f"DISTANCE={state['vector_index'].get('distance')} and mhnsw_max_cache_size is "
        f"{state['mhnsw_max_cache_size']} bytes. This is the state `wikilense ingest` produces, so "
        f"`wikilense eval` on it reproduces section 7 up to the HNSW build (a fresh graph)."
    )
    parts += [reading, ""]
    return parts


def summary_defaults(r: Results) -> list[str]:
    parts = ["## Chosen defaults and evidence", ""]
    need = (*[f"m{m}_ef_{ef}" for m in INDEX_M_VALUES for ef in INDEX_M_EFS], EXACT_OLD,
            "ef_20", "ef_100", "ef_400", "chunk60_ef_100", "chunk120_ef_100", "chunk240_ef_100",
            "prefix_on_inline", "prefix_off_inline", "strategy_inline_history",
            "strategy_overfetch10_history", "strategy_overfetch50_history",
            "strategy_inline_minwords1000", "strategy_overfetch10_minwords1000",
            "stability_comparison", "final_restart_comparison", *FINAL_RUNS)
    missing = [n for n in need if r.get(n) is None]
    if missing:
        parts += [
            (f"Not all groups have run yet (missing: {', '.join(missing)}); this section is written "
             "once every result file exists."),
            "",
        ]
        return parts
    comp = r.get("stability_comparison")["comparisons"]
    fcomp = r.get("final_restart_comparison")["comparisons"]
    h = FINAL_HEADLINE
    add = {m: (r.rebuild(f"index_m_sweep_m{m}") or {}).get("add_seconds") for m in INDEX_M_VALUES}
    size = {m: fmt_bytes_mb(r.param(f"m{m}_ef_20", "vector_index_tablespace_bytes", None))
            for m in INDEX_M_VALUES}
    names = ["chunk60_ef_100", "chunk120_ef_100", "chunk240_ef_100"]
    c100 = fcomp[f"{h}_vs_{FINAL_AFTER[h]}"]["runs"][FINAL_AFTER[h]]["identical_sequence"]
    c20 = fcomp[f"{FINAL_EF_20}_vs_{FINAL_AFTER[FINAL_EF_20]}"]["runs"][FINAL_AFTER[FINAL_EF_20]][
        "identical_sequence"]
    short10 = r.get("strategy_overfetch10_history")["parameters"]["short_results"]["10"]["queries_short_of_k"]
    short50 = r.get("strategy_overfetch50_history")["parameters"]["short_results"]["10"]["queries_short_of_k"]
    reach_exact = ", ".join(
        str(m) for m in INDEX_M_VALUES
        if r.art_n(f"m{m}_ef_20", 10) == r.art_n(EXACT_OLD, 10)
        and r.ev_n(f"m{m}_ef_20", 20) == r.ev_n(EXACT_OLD, 20)
    )
    parts += [
        ("The defaults in the code are `sql/schema.sql` M=16, `Settings.ef_search` 100, "
        "`chunk_max_words` 240 with `chunk_overlap_units` 1, the embedding prefix on, "
        "`search()` strategy `inline` for filtered queries and `mhnsw_max_cache_size` 512 MB in "
        "docker-compose.yml. The evidence, section by section:"),
        "",
        (f"- **Index M = 16.** On the same vectors (section 3) the M=6 / 16 / 32 graphs give "
        f"article recall@10 {r.art('m6_ef_20', 10)} / {r.art('m16_ef_20', 10)} / "
        f"{r.art('m32_ef_20', 10)} and evidence recall@20 {r.ev('m6_ef_20', 20)} / "
        f"{r.ev('m16_ef_20', 20)} / {r.ev('m32_ef_20', 20)} at ef 20, against "
        f"{r.art(EXACT_OLD, 10)} / {r.ev(EXACT_OLD, 20)} for the exact ranking; at ef 100, the "
        f"default it is paired with, {r.art('m6_ef_100', 10)} / {r.art('m16_ef_100', 10)} / "
        f"{r.art('m32_ef_100', 10)} and {r.ev('m6_ef_100', 20)} / {r.ev('m16_ef_100', 20)} / "
        f"{r.ev('m32_ef_100', 20)}, with "
        f"{r.identical(EXACT_OLD, 'm6_ef_100')} / {r.identical(EXACT_OLD, 'm16_ef_100')} / "
        f"{r.identical(EXACT_OLD, 'm32_ef_100')} hit lists identical to the exact ranking; the builds "
        f"took {add[6]} / {add[16]} / {add[32]} s and the graphs are {size[6]} / {size[16]} / "
        f"{size[32]}. The M values whose article recall@10 and evidence recall@20 at ef 20 equal "
        f"the exact ranking's: {reach_exact or 'none'}; M=32 builds in {add[32]} s against "
        f"{add[16]} s for M=16 and moves article recall@10 by "
        f"{abs((r.art_n('m32_ef_100', 10) or 0) - (r.art_n('m16_ef_100', 10) or 0))} claims and "
        f"evidence recall@20 by "
        f"{abs((r.ev_n('m32_ef_100', 20) or 0) - (r.ev_n('m16_ef_100', 20) or 0))} at ef 100."),
        (f"- **mhnsw_ef_search = 100 per query.** On the M=6 graph (section 2) ef 20 / 100 / 400 "
        f"give article recall@10 {r.art('ef_20', 10)} / {r.art('ef_100', 10)} / {r.art('ef_400', 10)} "
        f"and evidence recall@20 {r.ev('ef_20', 20)} / {r.ev('ef_100', 20)} / {r.ev('ef_400', 20)} "
        f"for p50@10 {r.p50('ef_20', 10)} / {r.p50('ef_100', 10)} / {r.p50('ef_400', 10)} ms. On "
        f"the M=16 graph ef 100 makes {r.identical(EXACT_OLD, 'm16_ef_100')} hit lists identical "
        f"to the exact ranking against {r.identical(EXACT_OLD, 'm16_ef_20')} at ef 20 (section 3), "
        f"and on the final ingest (section 7) {r.identical(FINAL_EXACT, h)} against "
        f"{r.identical(FINAL_EXACT, FINAL_EF_20)}, for p50@10 {r.p50(h, 10)} against "
        f"{r.p50(FINAL_EF_20, 10)} ms; across the restart {c100} hit lists stayed identical at "
        f"ef 100 against {c20} at ef 20."),
        (f"- **chunk_max_words = 240, overlap 1.** At 1,200 nominal words (section 4) the 60 / 120 "
        f"/ 240-word chunkings give article recall {r.art(names[0], 20)} / {r.art(names[1], 10)} / "
        f"{r.art(names[2], 5)} and evidence recall {r.ev(names[0], 20)} / {r.ev(names[1], 10)} / "
        f"{r.ev(names[2], 5)}, at 2,400 words {r.art(names[0], 40)} / {r.art(names[1], 20)} / "
        f"{r.art(names[2], 10)} and {r.ev(names[0], 40)} / {r.ev(names[1], 20)} / "
        f"{r.ev(names[2], 10)}; 240 words store {r.param(names[2], 'n_chunks')} vectors against "
        f"{r.param(names[1], 'n_chunks')} and {r.param(names[0], 'n_chunks')}, a graph of "
        f"{fmt_bytes_mb(r.param(names[2], 'vector_index_tablespace_bytes', None))} against "
        f"{fmt_bytes_mb(r.param(names[1], 'vector_index_tablespace_bytes', None))} and "
        f"{fmt_bytes_mb(r.param(names[0], 'vector_index_tablespace_bytes', None))}, and SQL p50 "
        f"{r.p50(names[2], 5)} against {r.p50(names[1], 10)} and {r.p50(names[0], 20)} ms for the "
        f"1,200-word row."),
        (f"- **Embedding prefix on.** The exact rankings (section 5) give evidence recall@20 "
        f"{r.ev('prefix_on_inline', 20)} with the prefix and {r.ev('prefix_off_inline', 20)} "
        f"without, article recall@10 {r.art('prefix_on_inline', 10)} against "
        f"{r.art('prefix_off_inline', 10)}, unit coverage@10 {r.cov('prefix_on_inline', 10)} "
        f"against {r.cov('prefix_off_inline', 10)}: with the prefix, evidence recall@20 is "
        f"{cmp_word(r.ev_n('prefix_on_inline', 20), r.ev_n('prefix_off_inline', 20))} the value "
        f"without it."),
        (f"- **Filtered queries: strategy inline; overfetch when latency matters more than a full "
        f"result.** Under the selective History filter (section 6) overfetch 10 returned fewer "
        f"than 10 rows for {short10} of 75 queries and overfetch 50 for {short50}, inline for "
        f"{r.get('strategy_inline_history')['parameters']['short_results']['10']['queries_short_of_k']}, "
        f"at p50@10 {r.p50('strategy_inline_history', 10)} ms (p95 "
        f"{r.p95('strategy_inline_history', 10)} ms) against "
        f"{r.p50('strategy_overfetch10_history', 10)} / {r.p50('strategy_overfetch50_history', 10)} "
        f"ms; under the non-selective min_words filter the three give article recall@10 "
        f"{r.art('strategy_inline_minwords1000', 10)} / "
        f"{r.art('strategy_overfetch10_minwords1000', 10)} / "
        f"{r.art('strategy_overfetch50_minwords1000', 10)} at "
        f"{r.p50('strategy_inline_minwords1000', 10)} / "
        f"{r.p50('strategy_overfetch10_minwords1000', 10)} / "
        f"{r.p50('strategy_overfetch50_minwords1000', 10)} ms."),
        (f"- **rrf hybrid: available, not the default.** On the final ingest (section 7) rrf gives "
        f"article recall@10 {r.art(FINAL_RRF, 10)} and evidence recall@20 {r.ev(FINAL_RRF, 20)} "
        f"against {r.art(h, 10)} / {r.ev(h, 20)} for the vector ranking alone, at p50@10 "
        f"{r.p50(FINAL_RRF, 10)} against {r.p50(h, 10)} ms; on the OLD ingest (section 2) "
        f"{r.art(RRF_OLD, 10)} / {r.ev(RRF_OLD, 20)} against {r.art('ef_100', 10)} / "
        f"{r.ev('ef_100', 20)} at {r.p50(RRF_OLD, 10)} against {r.p50('ef_100', 10)} ms."),
        (f"- **mhnsw_max_cache_size = 512 MB, for capacity.** Within one server process the cache "
        f"size changed nothing (section 1: "
        f"{comp['after_restart_vs_16mb']['runs'][STABILITY_16_RUNS[0]]['identical_sequence']} of "
        f"{comp['after_restart_vs_16mb']['n_claims']} hit lists identical between 512 MB and 16 MB), "
        f"while the container restart changed "
        f"{comp['after_restart_vs_512mb']['n_claims'] - comp['after_restart_vs_512mb']['runs'][AFTER_RESTART]['identical_sequence']} "
        f"of {comp['after_restart_vs_512mb']['n_claims']} hit lists on the M=6 graph and "
        f"{fcomp[f'{h}_vs_{FINAL_AFTER[h]}']['n_claims'] - c100} of "
        f"{fcomp[f'{h}_vs_{FINAL_AFTER[h]}']['n_claims']} on the M=16 graph at ef 100. The graph of "
        f"the final ingest is {fmt_bytes_mb(r.param(h, 'vector_index_tablespace_bytes', None))} on "
        f"disk and the 120-word ingest's {fmt_bytes_mb(r.param('ef_100', 'vector_index_tablespace_bytes', None))}, "
        f"already the size of the 16 MB server default."),
        "",
    ]
    return parts


def group_summary(ctx: Context) -> None:
    path = ctx.out_dir / "SUMMARY.md"
    path.write_text(summary_markdown(ctx.out_dir), encoding="utf-8")
    log.info("wrote %s", path)


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------


GROUPS = {
    "stability": group_stability,
    "ef_search": group_ef_search,
    "index_m": group_index_m,
    "chunk_size": group_chunk_size,
    "prefix": group_prefix,
    "filters": group_filters,
    "final": group_final,
    "restore": group_restore,
    "summary": group_summary,
}
PROTOCOL_ORDER = tuple(GROUPS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the WikiLense experiment protocol (see the module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="groups: " + ", ".join(PROTOCOL_ORDER) + ", all",
    )
    parser.add_argument("groups", nargs="+", choices=[*PROTOCOL_ORDER, "all"], metavar="GROUP")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="results directory (default results/)")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"timed passes per claim after one warm-up (default {DEFAULT_REPEATS})")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", stream=sys.stderr
    )
    logging.getLogger("wikilense").setLevel(logging.INFO)
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    groups = list(PROTOCOL_ORDER) if "all" in args.groups else list(dict.fromkeys(args.groups))
    settings = load_settings()
    if settings.db_name == settings.test_db_name:
        raise SystemExit("refusing to run the experiments against the test database")
    ctx = Context(settings=settings, server=Server(settings), out_dir=args.out, repeats=args.repeats)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        started = time.perf_counter()
        log.info("=== group %s ===", group)
        try:
            GROUPS[group](ctx)
        except ExperimentError as exc:
            log.error("group %s failed: %s", group, exc)
            return 1
        log.info("=== group %s done in %.0f s ===", group, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
