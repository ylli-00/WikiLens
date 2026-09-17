#!/usr/bin/env python3
"""Run the WikiLense experiment protocol on the main database and write the files in results/.

Every evaluation goes through ``wikilense.evaluate.evaluate`` and ``write_results`` (one JSON and
one Markdown file per run, named as in the protocol); every re-ingest through
``wikilense.ingest.run_ingest`` with ``dataclasses.replace(settings, ...)``. The evaluations run
one after the other, never two at once, so the latency samples never overlap. Each result file
records, besides the harness's own ``ingest_meta``, the effective ``mhnsw_ef_search``, the global
``mhnsw_max_cache_size``, the vector index's M and DISTANCE as ``SHOW CREATE TABLE chunk`` reports
them, and the sizes of the chunk table and of the hidden InnoDB tablespace that holds the HNSW
graph (``parameters`` in the JSON, the "Parameters" table in the Markdown).

Groups (the full protocol runs them in this order; each can be rerun on its own):

  stability   three baseline runs with the 512 MB HNSW cache, one after a container restart,
              three with a 16 MB cache (SET GLOBAL, restored afterwards); per-claim hit
              comparison -> stability_*.json/.md and stability_comparison.json/.md
  ef_search   mhnsw_ef_search 20/50/100/200/400 with strategy none (ef_<n>), 20/100 with the
              inline statement and no filters (inline_ef_<n>)
  index_m     rebuild the vector index with M=16, M=32 and back to M=6 (timed ALTER TABLE, index
              sizes) -> m16_ef_20/100, m32_ef_20/100, m6_rebuilt_ef_20, index_m_rebuild.json/.md
  chunk_size  re-ingest with chunk_max_words 60 and 240 (overlap 1, prefix on), evaluate at
              ef_search 100 with ks that give the same amount of retrieved text
              -> chunk60_ef_100, chunk240_ef_100 (the 120-word case is ef_100)
  prefix      re-ingest 120 words without the "title > section: " prefix -> noprefix_ef_100/20
  filters     min_words=1000 and heading LIKE '%History%' with the inline, overfetch 10 and
              overfetch 50 strategies at ef_search 20, k 5 and 10 -> strategy_*
  restore     re-ingest the defaults (120 words, overlap 1, prefix on; the schema's M=6), run the
              baseline once more -> final_state_ef_20, and check ingest_meta against baseline.json
  summary     results/SUMMARY.md from the JSON files (no database access)
  all         every group above, in that order

Groups that need the default ingest (stability, ef_search, index_m, filters) check ``ingest_meta``
first and re-ingest the defaults when a previous group left another chunking in the database;
``index_m`` also rebuilds M=6 first when another M is in place. Every re-ingest is appended to
results/ingest_runs.json.

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
from dataclasses import dataclass, replace
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
    DEFAULT_ENV_FILE,
    DEFAULT_INDEX_M,
    Settings,
    load_settings,
)
from wikilense.embedding import Embedder
from wikilense.evaluate import (
    DEFAULT_RESULTS_DIR,
    EvalResult,
    evaluate,
    load_ground_truth,
    read_ingest_meta,
    write_results,
)
from wikilense.ingest import run_ingest
from wikilense.search import Filters, search

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
CHUNK60_KS: tuple[int, ...] = (2, 6, 10, 20, 40)
CHUNK240_KS: tuple[int, ...] = (1, 2, 3, 5, 10)
FILTER_KS: tuple[int, ...] = (5, 10)
EF_SWEEP: tuple[int, ...] = (20, 50, 100, 200, 400)
INDEX_M_SWEEP: tuple[int, ...] = (16, 32)
MIN_WORDS_FILTER = 1000
HEADING_FILTER = "%History%"
FILTER_EF_SEARCH = 20

INDEX_NAME = "embedding"
DROP_VECTOR_INDEX_SQL = "ALTER TABLE chunk DROP INDEX `embedding`"
#: ADD VECTOR INDEX statements as fixed literals, one per M value of the sweep.
ADD_VECTOR_INDEX_SQL: dict[int, str] = {
    6: "ALTER TABLE chunk ADD VECTOR INDEX `embedding` (embedding) M=6 DISTANCE=cosine",
    16: "ALTER TABLE chunk ADD VECTOR INDEX `embedding` (embedding) M=16 DISTANCE=cosine",
    32: "ALTER TABLE chunk ADD VECTOR INDEX `embedding` (embedding) M=32 DISTANCE=cosine",
}

SHOW_CREATE_CHUNK_SQL = "SHOW CREATE TABLE chunk"
GLOBALS_SQL = "SELECT @@GLOBAL.mhnsw_max_cache_size, @@GLOBAL.mhnsw_ef_search"
CHUNK_SIZES_SQL = (
    "SELECT data_length, index_length FROM information_schema.tables "
    "WHERE table_schema = DATABASE() AND table_name = 'chunk'"
)
CHUNK_WORDS_SQL = "SELECT n_words FROM chunk"
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
#: Hidden InnoDB tablespaces of the vector index (root: needs the PROCESS privilege).
HIDDEN_TABLESPACE_SQL = (
    "SELECT name, file_size FROM information_schema.innodb_sys_tablespaces WHERE name LIKE '{db}/chunk#i#%'"
)

INDEX_DEFINITION_RE = re.compile(r"VECTOR KEY `(?P<name>[^`]+)` \(`embedding`\)(?P<options>[^\n]*)")
INDEX_M_RE = re.compile(r"`M`='(\d+)'")
INDEX_DISTANCE_RE = re.compile(r"`DISTANCE`='(\w+)'")

STABILITY_512_RUNS = ("stability_512mb_run1", "stability_512mb_run2", "stability_512mb_run3")
STABILITY_16_RUNS = ("stability_16mb_run1", "stability_16mb_run2", "stability_16mb_run3")
AFTER_RESTART = "stability_512mb_after_restart"
COMMITTED_BASELINE = "baseline"


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
                "docker is not on PATH; the stability group and the tablespace sizes need it"
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
        try:
            rows = self.root_sql(HIDDEN_TABLESPACE_SQL.format(db=self.settings.db_name))
        except ExperimentError as exc:
            log.warning("hidden tablespace size not available: %s", exc)
            return None
        return sum(int(row[1]) for row in rows) if rows else None

    def restart(self, timeout_s: float = 300.0) -> float:
        """``docker compose restart``, wait until healthy and connectable; returns the seconds."""
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
                        cur.execute("SELECT 1")
                finally:
                    conn.close()
                break
            except pymysql.err.OperationalError as exc:
                if time.perf_counter() > deadline:
                    raise ExperimentError(f"server not connectable after restart: {exc}") from exc
                time.sleep(1.0)
        seconds = time.perf_counter() - start
        log.info("container healthy and connectable after %.1f s", seconds)
        return seconds


# ---------------------------------------------------------------------------------------------
# server state recorded in every result file
# ---------------------------------------------------------------------------------------------


def index_definition(conn: pymysql.Connection) -> dict[str, Any]:
    """Return ``{name, m, distance}`` of the vector index from SHOW CREATE TABLE chunk."""
    with conn.cursor() as cur:
        cur.execute(SHOW_CREATE_CHUNK_SQL)
        ddl = str(cur.fetchone()[1])
    match = INDEX_DEFINITION_RE.search(ddl)
    if match is None:
        return {"name": None, "m": None, "distance": None}
    options = match.group("options")
    m_match = INDEX_M_RE.search(options)
    d_match = INDEX_DISTANCE_RE.search(options)
    return {
        "name": match.group("name"),
        "m": int(m_match.group(1)) if m_match else None,
        "distance": d_match.group(1) if d_match else None,
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


# ---------------------------------------------------------------------------------------------
# context, evaluation and ingest wrappers
# ---------------------------------------------------------------------------------------------


@dataclass
class Context:
    """Settings, the docker helper, the output directory and the lazily loaded GPU embedder."""

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


def run_eval(
    ctx: Context,
    name: str,
    *,
    group: str,
    ks: Sequence[int] = DEFAULT_KS,
    ef_search: int | None = None,
    strategy: str = "none",
    filters: Filters | None = None,
    overfetch: int = 10,
    extra: dict[str, Any] | None = None,
) -> EvalResult:
    """Run one evaluation on a fresh connection, add the server state, write <name>.json/.md."""
    log.info(
        "eval %s: strategy=%s ef_search=%s ks=%s overfetch=%s filters=%s",
        name, strategy, ef_search, list(ks), overfetch if strategy == "overfetch" else "-",
        filters,
    )
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
        )
        result.parameters["experiment_group"] = group
        result.parameters["eval_seconds"] = round(time.perf_counter() - started, 1)
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


INGEST_RUNS = "ingest_runs"


def reingest(
    ctx: Context,
    *,
    label: str,
    chunk_max_words: int,
    chunk_overlap_units: int = DEFAULT_CHUNK_OVERLAP_UNITS,
    use_prefix: bool = True,
) -> dict[str, Any]:
    """Re-ingest the corpus with the given chunking and prefix; appends to ingest_runs.json."""
    settings = replace(
        ctx.settings, chunk_max_words=chunk_max_words, chunk_overlap_units=chunk_overlap_units
    )
    log.info(
        "re-ingest %s: chunk_max_words=%d overlap=%d prefix=%s",
        label, chunk_max_words, chunk_overlap_units, use_prefix,
    )
    report = run_ingest(settings, embedder=ctx.embedder, use_prefix=use_prefix, progress=False)
    conn = ctx.connect()
    try:
        snapshot = server_snapshot(conn, ctx.server)
        meta = read_ingest_meta(conn)
    finally:
        conn.close()
    info: dict[str, Any] = {
        "label": label,
        "chunk_max_words": chunk_max_words,
        "chunk_overlap_units": chunk_overlap_units,
        "use_prefix": use_prefix,
        "ingested_at": meta.get("ingested_at"),
        "counts": report.counts(),
        "seconds": {stage: round(value, 2) for stage, value in report.seconds.items()},
        "chunk_words_mean": snapshot["chunk_words_mean"],
        "chunk_words_median": snapshot["chunk_words_median"],
        "chunk_words_p95": snapshot["chunk_words_p95"],
        "chunk_words_max": snapshot["chunk_words_max"],
        "vector_index_m": snapshot["vector_index_m"],
        "chunk_data_length": snapshot["chunk_data_length"],
        "chunk_index_length": snapshot["chunk_index_length"],
        "vector_index_tablespace_bytes": snapshot["vector_index_tablespace_bytes"],
    }
    log.info(
        "  %d chunks (mean %.1f words) in %.1f s (embed %.1f s)",
        report.n_chunks, snapshot["chunk_words_mean"] or 0.0, report.seconds["total"],
        report.seconds["embed"],
    )
    runs = load_json(ctx.out_dir / f"{INGEST_RUNS}.json") or []
    runs.append(info)
    write_json(ctx.out_dir / f"{INGEST_RUNS}.json", runs)
    (ctx.out_dir / f"{INGEST_RUNS}.md").write_text(ingest_runs_markdown(runs), encoding="utf-8")
    return info


def ingest_runs_markdown(runs: list[dict[str, Any]]) -> str:
    rows = [
        (
            r["label"], r["chunk_max_words"], r["chunk_overlap_units"], r["use_prefix"],
            r["counts"]["n_chunks"], r["chunk_words_mean"], r["chunk_words_median"],
            r["seconds"]["total"], r["seconds"]["embed"], r["seconds"]["load"],
            r["vector_index_tablespace_bytes"], r["ingested_at"],
        )
        for r in runs
    ]
    return (
        "# Re-ingests run by scripts/run_experiments.py\n\n"
        "One row per `run_ingest` call, in order. Seconds are the stages of `IngestReport`.\n\n"
        + md_table(
            ("label", "max_words", "overlap", "prefix", "chunks", "words mean", "words median",
             "total s", "embed s", "load s", "vector index bytes", "ingested_at"),
            rows,
        )
        + "\n"
    )


def is_default_ingest(meta: dict[str, str]) -> bool:
    return (
        meta.get("chunk_max_words") == str(DEFAULT_CHUNK_MAX_WORDS)
        and meta.get("chunk_overlap_units") == str(DEFAULT_CHUNK_OVERLAP_UNITS)
        and meta.get("embedding_prefix") == "true"
    )


def ensure_default_ingest(ctx: Context, label: str) -> dict[str, Any] | None:
    """Re-ingest the defaults when ingest_meta shows another chunking; returns the ingest info."""
    conn = ctx.connect()
    try:
        meta = read_ingest_meta(conn)
    finally:
        conn.close()
    if is_default_ingest(meta):
        return None
    log.info("database holds a non-default ingest (%s); re-ingesting the defaults first", meta)
    return reingest(
        ctx,
        label=label,
        chunk_max_words=DEFAULT_CHUNK_MAX_WORDS,
        chunk_overlap_units=DEFAULT_CHUNK_OVERLAP_UNITS,
        use_prefix=True,
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


# ---------------------------------------------------------------------------------------------
# group 1: stability
# ---------------------------------------------------------------------------------------------


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
        "unstable_claims": data["unstable_claims"],
        "article_hits": {str(m["k"]): m["article_hits"] for m in data["per_k"]},
        "evidence_hits": {str(m["k"]): m["evidence_hits"] for m in data["per_k"]},
    }


def group_stability(ctx: Context) -> None:
    ensure_default_ingest(ctx, "stability_defaults")
    cache_before = ensure_cache_size(ctx, CACHE_512MB)
    log.info("mhnsw_max_cache_size in force: %d", cache_before)
    for name in STABILITY_512_RUNS:
        run_eval(ctx, name, group="stability", ef_search=20)
    restart_seconds = ctx.server.restart()
    run_eval(
        ctx, AFTER_RESTART, group="stability", ef_search=20,
        extra={"container_restart_seconds": round(restart_seconds, 1)},
    )
    ctx.server.set_cache_size(CACHE_16MB)
    try:
        for name in STABILITY_16_RUNS:
            run_eval(ctx, name, group="stability", ef_search=20)
    finally:
        ctx.server.set_cache_size(CACHE_512MB)
    write_stability_comparison(ctx.out_dir)


def write_stability_comparison(out_dir: Path) -> None:
    """Write stability_comparison.json/.md from the stability result files (and baseline.json)."""
    all_runs = [COMMITTED_BASELINE, *STABILITY_512_RUNS, AFTER_RESTART, *STABILITY_16_RUNS]
    comparisons = {
        "within_512mb": compare_hits(out_dir, STABILITY_512_RUNS[0], STABILITY_512_RUNS[1:]),
        "committed_baseline_vs_512mb": compare_hits(
            out_dir, COMMITTED_BASELINE, [*STABILITY_512_RUNS]
        ),
        "after_restart_vs_512mb": compare_hits(out_dir, STABILITY_512_RUNS[0], [AFTER_RESTART]),
        "after_restart_vs_16mb": compare_hits(out_dir, AFTER_RESTART, [STABILITY_16_RUNS[0]]),
        "within_16mb": compare_hits(out_dir, STABILITY_16_RUNS[0], STABILITY_16_RUNS[1:]),
        "512mb_vs_16mb": compare_hits(out_dir, STABILITY_512_RUNS[0], [*STABILITY_16_RUNS]),
        "committed_baseline_vs_16mb": compare_hits(
            out_dir, COMMITTED_BASELINE, [*STABILITY_16_RUNS]
        ),
    }
    data = {
        "created_at": utc_now(),
        "runs": [run_summary_fields(out_dir, name) for name in all_runs],
        "comparisons": comparisons,
    }
    write_json(out_dir / "stability_comparison.json", data)
    parts = [
        "# Stability of the HNSW results (per-claim hit chunk ids, LIMIT 20)",
        "",
        (f"Generated {data['created_at']}. A claim counts as identical when its 20 hit chunk ids "
        "are the same, in the same order; `identical_set` ignores the order."),
        "",
        "## Runs",
        "",
        md_table(
            ("run", "created_at", "index ingested_at", "cache bytes", "ef_search", "M",
             "unstable claims", "article hits @1/5/10/20", "evidence hits @5/10/20"),
            [
                (
                    r["name"], r.get("created_at"), r.get("ingested_at"),
                    r.get("mhnsw_max_cache_size"), r.get("ef_search_effective"),
                    r.get("vector_index_m"),
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
    for key, comparison in comparisons.items():
        parts.append(f"### {key}")
        parts.append("")
        parts.append(md_table(
            ("reference", "run", "identical (sequence)", "identical (set)", "of", "claims that differ"),
            comparison_rows(comparison),
        ))
        parts.append("")
    (out_dir / "stability_comparison.md").write_text("\n".join(parts), encoding="utf-8")
    log.info("wrote stability_comparison.json/.md")


# ---------------------------------------------------------------------------------------------
# group 2: ef_search sweep
# ---------------------------------------------------------------------------------------------


def ensure_index_m(ctx: Context, m: int) -> dict[str, Any] | None:
    conn = ctx.connect()
    try:
        definition = index_definition(conn)
    finally:
        conn.close()
    if definition["m"] == m and definition["name"] == INDEX_NAME:
        return None
    log.info("vector index is %s; rebuilding with M=%d first", definition, m)
    return rebuild_index(ctx, m)


def group_ef_search(ctx: Context) -> None:
    ensure_default_ingest(ctx, "ef_search_defaults")
    ensure_index_m(ctx, DEFAULT_INDEX_M)
    for ef in EF_SWEEP:
        run_eval(ctx, f"ef_{ef}", group="ef_search", ef_search=ef)
    for ef in (20, 100):
        run_eval(ctx, f"inline_ef_{ef}", group="ef_search", ef_search=ef, strategy="inline")


# ---------------------------------------------------------------------------------------------
# group 3: index M sweep
# ---------------------------------------------------------------------------------------------


def rebuild_index(ctx: Context, m: int) -> dict[str, Any]:
    """DROP and re-ADD the vector index with the given M (timed); returns the timings and sizes."""
    add_sql = ADD_VECTOR_INDEX_SQL.get(m)
    if add_sql is None:
        raise ValueError(f"M={m} is not one of {sorted(ADD_VECTOR_INDEX_SQL)}")
    conn = ctx.connect()
    try:
        before = index_definition(conn)
        if before["name"] != INDEX_NAME:
            raise ExperimentError(
                f"expected the vector index to be named {INDEX_NAME!r}, SHOW CREATE TABLE says "
                f"{before}"
            )
        log.info("rebuilding the vector index: M=%s -> M=%d", before["m"], m)
        with conn.cursor() as cur:
            start = time.perf_counter()
            cur.execute(DROP_VECTOR_INDEX_SQL)
            drop_seconds = time.perf_counter() - start
            start = time.perf_counter()
            cur.execute(add_sql)
            add_seconds = time.perf_counter() - start
        conn.commit()
        snapshot = server_snapshot(conn, ctx.server)
    finally:
        conn.close()
    info = {
        "m": m,
        "m_before": before["m"],
        "index_name": INDEX_NAME,
        "drop_seconds": round(drop_seconds, 2),
        "add_seconds": round(add_seconds, 2),
        "vector_index_m_after": snapshot["vector_index_m"],
        "vector_index_distance_after": snapshot["vector_index_distance"],
        "chunk_index_length": snapshot["chunk_index_length"],
        "chunk_data_length": snapshot["chunk_data_length"],
        "vector_index_tablespace_bytes": snapshot["vector_index_tablespace_bytes"],
        "rebuilt_at": utc_now(),
    }
    log.info(
        "  drop %.2f s, add %.2f s; index_length %s, vector tablespace %s bytes",
        drop_seconds, add_seconds, info["chunk_index_length"], info["vector_index_tablespace_bytes"],
    )
    return info


def group_index_m(ctx: Context) -> None:
    ensure_default_ingest(ctx, "index_m_defaults")
    ensure_index_m(ctx, DEFAULT_INDEX_M)
    rebuilds: list[dict[str, Any]] = []
    for m in INDEX_M_SWEEP:
        info = rebuild_index(ctx, m)
        rebuilds.append(info)
        for ef in (20, 100):
            run_eval(
                ctx, f"m{m}_ef_{ef}", group="index_m", ef_search=ef,
                extra={"index_rebuild": info},
            )
    info = rebuild_index(ctx, DEFAULT_INDEX_M)
    rebuilds.append(info)
    run_eval(
        ctx, "m6_rebuilt_ef_20", group="index_m", ef_search=20, extra={"index_rebuild": info}
    )
    comparisons: dict[str, Any] = {}
    for reference in (STABILITY_512_RUNS[0], "ef_20"):
        if (ctx.out_dir / f"{reference}.json").is_file():
            comparisons[f"m6_rebuilt_vs_{reference}"] = compare_hits(
                ctx.out_dir, reference, ["m6_rebuilt_ef_20"]
            )
    data = {"created_at": utc_now(), "rebuilds": rebuilds, "comparisons": comparisons}
    write_json(ctx.out_dir / "index_m_rebuild.json", data)
    parts = [
        "# Vector index rebuilds (ALTER TABLE chunk DROP INDEX / ADD VECTOR INDEX)",
        "",
        (f"Generated {data['created_at']}. `index_length` is information_schema.tables for `chunk` "
        "(all secondary indexes); `vector tablespace` is the file size of the hidden InnoDB table "
        "`chunk#i#NN` that holds the HNSW graph."),
        "",
        md_table(
            ("M before", "M", "drop s", "add s", "index_length", "vector tablespace bytes",
             "rebuilt_at"),
            [
                (r["m_before"], r["m"], r["drop_seconds"], r["add_seconds"],
                 r["chunk_index_length"], r["vector_index_tablespace_bytes"], r["rebuilt_at"])
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
# group 4: chunk size, group 5: prefix
# ---------------------------------------------------------------------------------------------


def group_chunk_size(ctx: Context) -> None:
    info = reingest(ctx, label="chunk60", chunk_max_words=60)
    run_eval(
        ctx, "chunk60_ef_100", group="chunk_size", ks=CHUNK60_KS, ef_search=100,
        extra={"ingest_report": info},
    )
    info = reingest(ctx, label="chunk240", chunk_max_words=240)
    run_eval(
        ctx, "chunk240_ef_100", group="chunk_size", ks=CHUNK240_KS, ef_search=100,
        extra={"ingest_report": info},
    )


def group_prefix(ctx: Context) -> None:
    info = reingest(
        ctx, label="noprefix", chunk_max_words=DEFAULT_CHUNK_MAX_WORDS, use_prefix=False
    )
    for ef in (100, 20):
        run_eval(
            ctx, f"noprefix_ef_{ef}", group="prefix", ef_search=ef,
            extra={"ingest_report": info},
        )


# ---------------------------------------------------------------------------------------------
# group 6: filter strategies
# ---------------------------------------------------------------------------------------------


def filter_ground_truth(conn: pymysql.Connection) -> dict[str, Any]:
    """Return, per filter, how many chunks pass it and how many claims' gold evidence it excludes."""
    truths = load_ground_truth(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunk")
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
                           overfetch=overfetch, ef_search=ef_search))
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
    ("inline", "inline", 10),
    ("overfetch10", "overfetch", 10),
    ("overfetch50", "overfetch", 50),
)


def group_filters(ctx: Context) -> None:
    ensure_default_ingest(ctx, "filters_defaults")
    ensure_index_m(ctx, DEFAULT_INDEX_M)
    conn = ctx.connect()
    try:
        truth = filter_ground_truth(conn)
    finally:
        conn.close()
    for filter_name, filters in FILTER_CASES:
        for strategy_name, strategy, overfetch in STRATEGY_CASES:
            name = f"strategy_{strategy_name}_{filter_name}"
            short = count_short_results(
                ctx, FILTER_KS, filters, strategy, overfetch, FILTER_EF_SEARCH
            )
            run_eval(
                ctx, name, group="filters", ks=FILTER_KS, ef_search=FILTER_EF_SEARCH,
                strategy=strategy, filters=filters, overfetch=overfetch,
                extra={
                    "filter_ground_truth": truth[filter_name],
                    "short_results": short,
                },
            )


# ---------------------------------------------------------------------------------------------
# group 7: restore the default state
# ---------------------------------------------------------------------------------------------


def group_restore(ctx: Context) -> None:
    info = reingest(
        ctx, label="restore_defaults", chunk_max_words=DEFAULT_CHUNK_MAX_WORDS,
        chunk_overlap_units=DEFAULT_CHUNK_OVERLAP_UNITS, use_prefix=True,
    )
    cache = ensure_cache_size(ctx, CACHE_512MB)
    baseline = load_json(ctx.out_dir / f"{COMMITTED_BASELINE}.json")
    conn = ctx.connect()
    try:
        meta = read_ingest_meta(conn)
    finally:
        conn.close()
    differences: dict[str, Any] = {}
    if baseline is not None:
        for key, value in baseline["ingest_meta"].items():
            if key != "ingested_at" and meta.get(key) != value:
                differences[key] = {"baseline": value, "now": meta.get(key)}
    run_eval(
        ctx, "final_state_ef_20", group="restore", ef_search=20,
        extra={
            "ingest_report": info,
            "mhnsw_max_cache_size_after_restore": cache,
            "ingest_meta_matches_baseline": baseline is not None and not differences,
            "ingest_meta_differences": differences,
        },
    )
    log.info("ingest_meta vs baseline.json (ingested_at excepted): %s", differences or "identical")


# ---------------------------------------------------------------------------------------------
# group 8: SUMMARY.md
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

    def p50_n(self, name: str, k: int) -> float | None:
        m = self.per_k(name).get(k)
        return float(m["sql_latency"]["p50_ms"]) if m else None

    def param(self, name: str, key: str, default: Any = "not found") -> Any:
        data = self.get(name)
        if data is None:
            return default
        return data["parameters"].get(key, default)

    def art_n(self, name: str, k: int) -> int | None:
        m = self.per_k(name).get(k)
        return int(m["article_hits"]) if m else None

    def ev_n(self, name: str, k: int) -> int | None:
        m = self.per_k(name).get(k)
        return int(m["evidence_hits"]) if m else None

    def identical(self, reference: str, other: str) -> str:
        """Return "n/75": claims whose max-k hit lists are identical between the two runs."""
        if not self.has(reference, other):
            return "not found"
        comparison = compare_hits(self.out_dir, reference, [other])
        return f"{comparison['runs'][other]['identical_sequence']}/{comparison['n_claims']}"

    def lost_gold(self, reference: str, other: str) -> list[int]:
        """Return the claims with a gold page in the top hits of ``reference`` but not of ``other``."""
        if not self.has(reference, other):
            return []
        ref = claim_ranks(self.out_dir / f"{reference}.json")
        oth = claim_ranks(self.out_dir / f"{other}.json")
        return sorted(cid for cid, rank in ref.items() if rank is not None and oth.get(cid) is None)

    def ingest_run(self, label: str) -> dict[str, Any] | None:
        runs = self.get(INGEST_RUNS) or []
        for run in reversed(runs):
            if run.get("label") == label:
                return run
        return None


def fmt_bytes_mb(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "not found"
    return f"{value / 1048576:.1f} MB"


def fmt_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "not found"
    return f"{int(value):,} B"


def summary_markdown(out_dir: Path) -> str:
    r = Results(out_dir)
    header = r.get("final_state_ef_20") or r.get("ef_20") or r.get("baseline")
    intro = (
        f"Generated {utc_now()} by `scripts/run_experiments.py summary`. Every number below is read "
        "from the JSON files in `results/` written by the same script (`write_results` of the "
        "evaluation harness); the per-run Markdown files hold the full tables. 75 FEVEROUS claims "
        "over a 100-page corpus (8,868 chunks of at most 120 words unless stated); article recall "
        "counts the claims with a gold page among the top-k chunks (of 75), evidence recall the "
        "claims whose gold sentences are all inside the top-k chunks (of the 66 claims with a "
        "sentence-only evidence set). Latencies are the SQL time of one `search()` call in "
        "milliseconds, p50 / p95 over 5 repeats x 75 claims, warm, query embedding excluded "
        "(about 4.6 ms on the GPU, see the per-run files). Unless stated, the vector index is "
        "the schema's HNSW index with M=6 and cosine distance, and `ef` is `mhnsw_ef_search`."
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
    parts += summary_restore(r)
    parts += summary_recommendations(r)
    return "\n".join(parts) + "\n"


def _cache_cell(r: Results, name: str) -> str:
    value = r.param(name, "mhnsw_max_cache_size", None)
    if isinstance(value, int):
        return fmt_bytes_mb(value)
    if r.get(name) is not None:
        return "not recorded (16 MB server default at the time)"
    return "not found"


def summary_stability(r: Results) -> list[str]:
    runs = [COMMITTED_BASELINE, *STABILITY_512_RUNS, AFTER_RESTART, *STABILITY_16_RUNS]
    rows = []
    for name in runs:
        data = r.get(name)
        if data is None:
            rows.append([name] + ["not found"] * 8)
            continue
        rows.append([
            name, _cache_cell(r, name), data["ingest_meta"].get("ingested_at"), r.art(name, 1),
            r.art(name, 10), r.ev(name, 10), r.ev(name, 20), r.p50(name, 10),
            len(data["unstable_claims"]),
        ])
    parts = [
        "## 1. Stability: the same configuration repeated (strategy none, ef 20, no filters)",
        "",
        ("`baseline` is the committed run of 2026-09-16 (16 MB cache, minutes after the ingest). "
        "The other runs use the same on-disk index; the container was restarted between "
        "`stability_512mb_run3` and `stability_512mb_after_restart`, and "
        "`SET GLOBAL mhnsw_max_cache_size = 16777216` was issued before `stability_16mb_run1` "
        "(restored to 536870912 afterwards). \"claims unstable within run\" counts the claims "
        "whose LIMIT-20 hits changed between the five repeats of one run."),
        "",
        md_table(
            ("run", "cache", "index built", "article@1", "article@10", "evidence@10",
             "evidence@20", "SQL p50@10 ms", "claims unstable within run"),
            rows,
        ),
        "",
    ]
    comp = r.get("stability_comparison")
    if comp is None:
        parts += ["`stability_comparison.json` not found.", ""]
        return parts
    c = comp["comparisons"]
    rows = []
    for key in ("within_512mb", "after_restart_vs_512mb", "after_restart_vs_16mb", "within_16mb",
                "512mb_vs_16mb", "committed_baseline_vs_512mb", "committed_baseline_vs_16mb"):
        if key in c:
            rows.extend([[key, *row] for row in comparison_rows(c[key])])
    parts += [
        ("Per-claim comparison of the 20 hit chunk ids (order-sensitive; `identical (set)` ignores "
        "the order):"),
        "",
        md_table(("comparison", "reference", "run", "identical (sequence)", "identical (set)", "of",
                  "claims that differ"), rows),
        "",
    ]
    n = c["within_512mb"]["n_claims"]
    within512 = c["within_512mb"]["identical_in_all"]
    within16 = c["within_16mb"]["identical_in_all"]
    restart = c["after_restart_vs_512mb"]["runs"][AFTER_RESTART]["identical_sequence"]
    set_global = r.identical(AFTER_RESTART, STABILITY_16_RUNS[0])
    vs_base512 = c["committed_baseline_vs_512mb"]["identical_in_all"]
    vs_base16 = c["committed_baseline_vs_16mb"]["identical_in_all"]
    lost = r.lost_gold(STABILITY_512_RUNS[0], AFTER_RESTART)
    gained = r.lost_gold(AFTER_RESTART, STABILITY_512_RUNS[0])
    reading = (
        f"Reading: within one server process the index search is deterministic. The three 512 MB "
        f"runs return the same 20 chunks in the same order for {within512} of {n} claims, the "
        f"three 16 MB runs for {within16} of {n}, and lowering the cache from 512 MB to 16 MB "
        f"with SET GLOBAL changed nothing ({set_global} identical between the run just before "
        f"and just after it). The container restart is what changed the results: the run after "
        f"it matches the runs before it for only {restart} of {n} claims, {len(lost)} claims lost "
        f"their gold page from the top 20 ({', '.join(map(str, lost)) or 'none'}) and "
        f"{len(gained)} gained one, and article recall@10 went from "
        f"{r.art(STABILITY_512_RUNS[0], 10)} to {r.art(AFTER_RESTART, 10)}. The committed "
        f"baseline is a third state of the same on-disk index: identical hit lists for "
        f"{vs_base512} of {n} claims with the pre-restart runs and {vs_base16} with the "
        f"post-restart runs. So the phase-2 suspicion, a thrashing 16 MB cache, is not supported "
        f"by these runs; what differs is the in-memory graph the server has after (re)loading "
        f"the index, and at ef 20 the effect on recall is large. Section 2 shows that a higher "
        f"ef_search narrows it."
    )
    parts += [reading, ""]
    return parts


def summary_ef_search(r: Results) -> list[str]:
    names = [f"ef_{ef}" for ef in EF_SWEEP] + ["inline_ef_20", "inline_ef_100"]
    ks = DEFAULT_KS
    parts = [
        ("## 2. mhnsw_ef_search sweep (strategy none; `inline_*` = the joined statement without "
        "filters, all on the post-restart index state)"),
        "",
        "Article recall (of 75):",
        "",
        md_table(("run", *[f"@{k}" for k in ks]), [[n, *[r.art(n, k) for k in ks]] for n in names]),
        "",
        "Evidence recall (of 66):",
        "",
        md_table(("run", *[f"@{k}" for k in ks]), [[n, *[r.ev(n, k) for k in ks]] for n in names]),
        "",
        "SQL latency p50 / p95 in ms:",
        "",
        md_table(
            ("run", *[f"@{k}" for k in ks]),
            [[n, *[f"{r.p50(n, k)} / {r.p95(n, k)}" for k in ks]] for n in names],
        ),
        "",
    ]
    if not r.has(*names):
        return parts
    a = {ef: r.art(f"ef_{ef}", 10) for ef in EF_SWEEP}
    e = {ef: r.ev(f"ef_{ef}", 20) for ef in EF_SWEEP}
    p = {ef: r.p50(f"ef_{ef}", 10) for ef in EF_SWEEP}
    reading = (
        f"Reading: raising ef_search from 20 to 100 lifts article recall@10 from {a[20]} to "
        f"{a[100]} and evidence recall@20 from {e[20]} to {e[100]} for p50@10 {p[20]} vs "
        f"{p[100]} ms; ef 200 reaches {a[200]} / {e[200]} and ef 400 {a[400]} / {e[400]} at "
        f"{p[200]} / {p[400]} ms. The inline statement walks the index until k joined rows have "
        f"passed and so returns the exact top-k: {r.art('inline_ef_20', 10)} / "
        f"{r.ev('inline_ef_20', 20)} at both ef values (hit lists identical for "
        f"{r.identical('inline_ef_20', 'inline_ef_100')} claims) for p50@10 "
        f"{r.p50('inline_ef_20', 10)} ms. ef 400 matches the exact counts at every k; its hit "
        f"lists are identical to the exact ones for {r.identical('inline_ef_20', 'ef_400')} "
        f"claims (ef 200: {r.identical('inline_ef_20', 'ef_200')}, ef 100: "
        f"{r.identical('inline_ef_20', 'ef_100')}, ef 20: {r.identical('inline_ef_20', 'ef_20')}). "
        f"Across the two index states of section 1 the LIMIT-20 hit lists at ef 100 "
        f"(`ef_search_100`, committed, vs `ef_100`) agree for "
        f"{r.identical('ef_search_100', 'ef_100')} claims, against "
        f"{r.identical('baseline', 'ef_20')} at ef 20: a higher ef_search also makes the "
        f"results less dependent on the index state."
    )
    parts += [reading, ""]
    return parts


def summary_index_m(r: Results) -> list[str]:
    rebuild = r.get("index_m_rebuild")
    parts = [
        ("## 3. Index M sweep (ALTER TABLE chunk DROP INDEX / ADD VECTOR INDEX ... M=n "
        "DISTANCE=cosine; ef 20 and 100)"),
        "",
    ]
    if rebuild is not None:
        parts += [
            ("`chunk index_length` is `information_schema.tables` for `chunk` (its B-tree "
            "secondary indexes only, recomputed by the ALTER); `vector tablespace` is the file "
            "size of the hidden InnoDB table `chunk#i#NN` that holds the HNSW graph "
            "(`information_schema.innodb_sys_tablespaces`)."),
            "",
            md_table(
                ("M before", "M", "DROP INDEX s", "ADD VECTOR INDEX s", "chunk index_length",
                 "vector tablespace"),
                [
                    (b["m_before"], b["m"], b["drop_seconds"], b["add_seconds"],
                     fmt_bytes(b["chunk_index_length"]),
                     fmt_bytes_mb(b["vector_index_tablespace_bytes"]))
                    for b in rebuild["rebuilds"]
                ],
            ),
            "",
        ]
    names = ["ef_20", "ef_100", "m16_ef_20", "m16_ef_100", "m32_ef_20", "m32_ef_100",
             "m6_rebuilt_ef_20", "inline_ef_20"]
    rows = []
    for n in names:
        rows.append([
            n, r.param(n, "vector_index_m"), r.param(n, "ef_search_effective"),
            fmt_bytes_mb(r.param(n, "vector_index_tablespace_bytes", None)),
            r.art(n, 1), r.art(n, 5), r.art(n, 10), r.ev(n, 5), r.ev(n, 10), r.ev(n, 20),
            f"{r.p50(n, 10)} / {r.p95(n, 10)}", r.identical("inline_ef_20", n),
        ])
    parts += [
        ("`ef_20` / `ef_100` are the M=6 index built at ingest (section 2); `inline_ef_20` is the "
        "exact ranking for reference; the last column counts the claims whose 20 hits are "
        "identical to that exact ranking."),
        "",
        md_table(
            ("run", "M", "ef", "vector tablespace", "article@1", "article@5", "article@10",
             "evidence@5", "evidence@10", "evidence@20", "SQL p50 / p95 @10 ms",
             "identical to exact"),
            rows,
        ),
        "",
    ]
    if rebuild is not None:
        for comp in rebuild.get("comparisons", {}).values():
            run = comp["runs"]["m6_rebuilt_ef_20"]
            parts += [
                (f"`m6_rebuilt_ef_20` vs `{comp['reference']}`: identical hit lists for "
                f"{run['identical_sequence']} of {comp['n_claims']} claims (same set: "
                f"{run['identical_set']})."),
                "",
            ]
    if not r.has(*names) or rebuild is None:
        return parts
    add = {b["m"]: b["add_seconds"] for b in rebuild["rebuilds"]}
    size = {b["m"]: fmt_bytes_mb(b["vector_index_tablespace_bytes"]) for b in rebuild["rebuilds"]}
    reading = (
        f"Reading: at ef 20 the M=16 and M=32 graphs already return the exact counts of the "
        f"inline statement (article@10 {r.art('m16_ef_20', 10)} and {r.art('m32_ef_20', 10)}, "
        f"evidence@20 {r.ev('m16_ef_20', 20)} and {r.ev('m32_ef_20', 20)}), where the M=6 graph "
        f"gives {r.art('ef_20', 10)} / {r.ev('ef_20', 20)} at ef 20 and {r.art('ef_100', 10)} / "
        f"{r.ev('ef_100', 20)} at ef 100. Hit lists identical to the exact ranking: M=16 "
        f"{r.identical('inline_ef_20', 'm16_ef_20')} at ef 20 and "
        f"{r.identical('inline_ef_20', 'm16_ef_100')} at ef 100, M=32 "
        f"{r.identical('inline_ef_20', 'm32_ef_20')} and "
        f"{r.identical('inline_ef_20', 'm32_ef_100')}. p50@10 at ef 20 is {r.p50('ef_20', 10)} "
        f"(M=6) / {r.p50('m16_ef_20', 10)} (M=16) / {r.p50('m32_ef_20', 10)} (M=32) ms; ADD "
        f"VECTOR INDEX took {add.get(6)} / {add.get(16)} / {add.get(32)} s and the graph "
        f"tablespace is {size.get(6)} / {size.get(16)} / {size.get(32)}. The M=6 index rebuilt "
        f"by ALTER TABLE does not reproduce `stability_512mb_run1`: "
        f"{r.identical(STABILITY_512_RUNS[0], 'm6_rebuilt_ef_20')} identical hit lists, article "
        f"recall@10 {r.art('m6_rebuilt_ef_20', 10)} against {r.art(STABILITY_512_RUNS[0], 10)}; "
        f"every build of an M=6 graph is a different approximation of the same vectors "
        f"(`final_state_ef_20` in section 7 is a third one, identical to the rebuilt one for "
        f"{r.identical('final_state_ef_20', 'm6_rebuilt_ef_20')} claims)."
    )
    parts += [reading, ""]
    return parts


def summary_chunk_size(r: Results) -> list[str]:
    parts = [
        "## 4. Chunk size at equal retrieved text (strategy none, ef 100, overlap 1, prefix on)",
        "",
        ("ef 100 for all three chunkings so that the comparison measures the chunking rather than "
        "the HNSW approximation at the default ef 20. Each chunking has its own ingest and its "
        "own freshly built M=6 index; the 120-word case is `ef_100` of section 2 (the committed "
        "ingest), its ingest time is that of the identical re-ingest `restore_defaults` in "
        "`ingest_runs.json`."),
        "",
    ]
    ingest_rows = []
    cases = (("chunk60_ef_100", 60, "chunk60"), ("ef_100", 120, "restore_defaults"),
             ("chunk240_ef_100", 240, "chunk240"))
    for name, words, label in cases:
        data = r.get(name)
        run = r.ingest_run(label)
        if data is None:
            ingest_rows.append([words, "not found", "", "", "", "", "", "", ""])
            continue
        p = data["parameters"]
        ingest_rows.append([
            words, p["n_chunks"], p.get("chunk_words_mean"), p.get("chunk_words_median"),
            p.get("chunk_words_p95"), p.get("chunk_words_max"),
            run["seconds"]["total"] if run else "not found",
            run["seconds"]["embed"] if run else "not found",
            fmt_bytes_mb(p.get("vector_index_tablespace_bytes")),
        ])
    parts += [
        md_table(("chunk_max_words", "chunks", "words mean", "median", "p95", "max",
                  "ingest total s", "embed s", "vector tablespace"), ingest_rows),
        "",
    ]
    align = [
        (1200, ("chunk60_ef_100", 20), ("ef_100", 10), ("chunk240_ef_100", 5)),
        (720, None, None, ("chunk240_ef_100", 3)),
        (600, ("chunk60_ef_100", 10), ("ef_100", 5), None),
        (480, None, None, ("chunk240_ef_100", 2)),
        (360, ("chunk60_ef_100", 6), ("ef_100", 3), None),
        (240, None, None, ("chunk240_ef_100", 1)),
        (120, ("chunk60_ef_100", 2), ("ef_100", 1), None),
    ]
    rows = []
    for words, *cells in align:
        row: list[Any] = [words]
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
    parts += [
        ("Aligned by nominal retrieved words (chunk_max_words x k); the words in brackets are "
        "k x the mean chunk length actually stored. Each cell: article recall / evidence recall / "
        "unit coverage / SQL p50."),
        "",
        md_table(("nominal words", "60-word chunks", "120-word chunks", "240-word chunks"), rows),
        "",
    ]
    if not r.has("chunk60_ef_100", "ef_100", "chunk240_ef_100"):
        return parts
    reading = (
        f"Reading: at 1,200 nominal words the three chunkings give article recall "
        f"{r.art('chunk60_ef_100', 20)} / {r.art('ef_100', 10)} / {r.art('chunk240_ef_100', 5)} "
        f"and evidence recall {r.ev('chunk60_ef_100', 20)} / {r.ev('ef_100', 10)} / "
        f"{r.ev('chunk240_ef_100', 5)} (unit coverage {r.cov('chunk60_ef_100', 20)} / "
        f"{r.cov('ef_100', 10)} / {r.cov('chunk240_ef_100', 5)}) for 60 / 120 / 240 words; at "
        f"about 600 words, 60x10 gives {r.art('chunk60_ef_100', 10)} / "
        f"{r.ev('chunk60_ef_100', 10)}, 120x5 {r.art('ef_100', 5)} / {r.ev('ef_100', 5)}, 240x2 "
        f"(480 words) {r.art('chunk240_ef_100', 2)} / {r.ev('chunk240_ef_100', 2)} and 240x3 "
        f"(720 words) {r.art('chunk240_ef_100', 3)} / {r.ev('chunk240_ef_100', 3)}. The 240-word "
        f"chunks are never below the others at equal nominal text although they actually "
        f"retrieve fewer words (mean chunk {r.param('chunk240_ef_100', 'chunk_words_mean')} "
        f"words, because sections are short), and the differences of two to three claims are "
        f"the size of the index-state noise at ef 100 (section 2: `ef_search_100` vs `ef_100` "
        f"differ by one claim in article recall@10 and three in evidence recall@20). So the chunk "
        f"size does not move recall at equal retrieved text on this corpus; it moves the cost: "
        f"{r.param('chunk60_ef_100', 'n_chunks')} / {r.param('ef_100', 'n_chunks')} / "
        f"{r.param('chunk240_ef_100', 'n_chunks')} vectors, a graph of "
        f"{fmt_bytes_mb(r.param('chunk60_ef_100', 'vector_index_tablespace_bytes', None))} / "
        f"{fmt_bytes_mb(r.param('ef_100', 'vector_index_tablespace_bytes', None))} / "
        f"{fmt_bytes_mb(r.param('chunk240_ef_100', 'vector_index_tablespace_bytes', None))}, "
        f"and p50 {r.p50('chunk60_ef_100', 20)} / {r.p50('ef_100', 10)} / "
        f"{r.p50('chunk240_ef_100', 5)} ms for the 1,200-word row."
    )
    parts += [reading, ""]
    return parts


def summary_prefix(r: Results) -> list[str]:
    ks = DEFAULT_KS
    names = [("ef_20", "on", 20), ("noprefix_ef_20", "off", 20), ("ef_100", "on", 100),
             ("noprefix_ef_100", "off", 100), ("inline_ef_20", "on (exact ranking)", 20)]
    parts = [
        ("## 5. Embedding prefix \"title > section path: \" on / off (120-word chunks, strategy "
        "none)"),
        "",
        ("The no-prefix runs use their own re-ingest and freshly built M=6 index; `inline_ef_20` "
        "is the exact ranking of the prefixed vectors, for reference."),
        "",
        md_table(
            ("run", "prefix", "ef", *[f"article@{k}" for k in ks], *[f"evidence@{k}" for k in ks],
             "coverage@10"),
            [[n, p, ef, *[r.art(n, k) for k in ks], *[r.ev(n, k) for k in ks], r.cov(n, 10)]
             for n, p, ef in names],
        ),
        "",
    ]
    if not r.has(*[n for n, _, _ in names]):
        return parts
    reading = (
        f"Reading: without the prefix the counts are equal or higher at every k. At ef 100, "
        f"article recall@1 is {r.art('noprefix_ef_100', 1)} against {r.art('ef_100', 1)} with "
        f"the prefix, @10 {r.art('noprefix_ef_100', 10)} against {r.art('ef_100', 10)}, "
        f"evidence recall@10 {r.ev('noprefix_ef_100', 10)} against {r.ev('ef_100', 10)}, @20 "
        f"{r.ev('noprefix_ef_100', 20)} against {r.ev('ef_100', 20)}, unit coverage@10 "
        f"{r.cov('noprefix_ef_100', 10)} against {r.cov('ef_100', 10)}; at ef 20 the gap is "
        f"wider ({r.art('noprefix_ef_20', 10)} vs {r.art('ef_20', 10)} article@10) but that "
        f"comparison is confounded by the index state of section 1. Against the exact ranking "
        f"of the prefixed vectors (`inline_ef_20`: article@10 {r.art('inline_ef_20', 10)}, "
        f"evidence@10 {r.ev('inline_ef_20', 10)}, @20 {r.ev('inline_ef_20', 20)}) the "
        f"approximate no-prefix ranking at ef 100 stands at {r.art('noprefix_ef_100', 10)}, "
        f"{r.ev('noprefix_ef_100', 10)} and {r.ev('noprefix_ef_100', 20)}. The title-and-section "
        f"prefix therefore buys nothing on these claims, which name their subject in the claim "
        f"text itself."
    )
    parts += [reading, ""]
    return parts


def summary_filters(r: Results) -> list[str]:
    parts = [
        "## 6. Filter strategies (ef 20, k 5 and 10, 120-word chunks, prefix on)",
        "",
        ("`queries < k rows` counts, over the 75 claims, the queries that returned fewer than k "
        "rows (a separate untimed pass of `search()` per claim and k); `no rows` those that "
        "returned nothing."),
        "",
    ]
    rows = []
    truth_by_filter: dict[str, Any] = {}
    for filter_name, _ in FILTER_CASES:
        for strategy_name, _, _ in STRATEGY_CASES:
            n = f"strategy_{strategy_name}_{filter_name}"
            data = r.get(n)
            if data is None:
                rows.append([n] + ["not found"] * 10)
                continue
            truth_by_filter[filter_name] = data["parameters"].get("filter_ground_truth")
            short = data["parameters"].get("short_results", {})
            rows.append([
                n, f"{r.p50(n, 5)} / {r.p95(n, 5)}", f"{r.p50(n, 10)} / {r.p95(n, 10)}",
                short.get("5", {}).get("queries_short_of_k", "-"),
                short.get("10", {}).get("queries_short_of_k", "-"),
                short.get("10", {}).get("queries_with_no_rows", "-"),
                short.get("10", {}).get("rows_mean", "-"),
                r.art(n, 5), r.art(n, 10), r.ev(n, 5), r.ev(n, 10),
            ])
    parts += [
        md_table(
            ("run", "SQL p50 / p95 @5 ms", "SQL p50 / p95 @10 ms", "queries < 5 rows",
             "queries < 10 rows", "no rows @10", "rows mean @10", "article@5", "article@10",
             "evidence@5", "evidence@10"),
            rows,
        ),
        "",
    ]
    for filter_name, truth in truth_by_filter.items():
        if not truth:
            continue
        parts.append(
            f"Filter `{filter_name}` ({truth['filter']}): {truth['n_chunks_passing']} of "
            f"{truth['n_chunks']} chunks pass, on {truth['n_pages_passing']} of 100 pages. The "
            f"filter itself excludes every gold page of {truth['claims_gold_page_excluded']} of "
            f"{truth['n_claims']} claims, so article recall can reach at most "
            f"{truth['article_recall_ceiling']}/{truth['n_claims']}; evidence recall at most "
            f"{truth['evidence_recall_ceiling']}/{truth['n_evidence_claims']} (a claim's gold "
            f"sentences must all lie in chunks that pass the filter)."
        )
        parts.append("")
    names = [f"strategy_{s}_{f}" for f, _ in FILTER_CASES for s, _, _ in STRATEGY_CASES]
    if not r.has(*names) or len(truth_by_filter) < 2:
        return parts
    t_min = truth_by_filter["minwords1000"]
    t_his = truth_by_filter["history"]

    def short(name: str, k: int, key: str = "queries_short_of_k") -> Any:
        return r.get(name)["parameters"]["short_results"][str(k)][key]

    reading = (
        f"Reading: with the non-selective filter (min_words 1000, {t_min['n_chunks_passing']} "
        f"of {t_min['n_chunks']} chunks pass) every strategy returns k rows for every query and "
        f"the same article recall ({r.art('strategy_inline_minwords1000', 10)} at k=10, of a "
        f"ceiling of {t_min['article_recall_ceiling']}); evidence recall@10 is "
        f"{r.ev('strategy_inline_minwords1000', 10)} inline against "
        f"{r.ev('strategy_overfetch10_minwords1000', 10)} / "
        f"{r.ev('strategy_overfetch50_minwords1000', 10)} for overfetch 10 / 50, whose inner "
        f"index query is approximate at ef 20 while the inline walk is exact. Inline costs "
        f"{r.p50('strategy_inline_minwords1000', 10)} ms p50 at k=10 against "
        f"{r.p50('strategy_overfetch10_minwords1000', 10)} ms (overfetch 10) and "
        f"{r.p50('strategy_overfetch50_minwords1000', 10)} ms (overfetch 50). With the selective "
        f"filter (heading LIKE '%History%', {t_his['n_chunks_passing']} chunks pass) overfetch "
        f"10 returns fewer than 10 rows for {short('strategy_overfetch10_history', 10)} of 75 "
        f"queries ({short('strategy_overfetch10_history', 10, 'queries_with_no_rows')} with no "
        f"row at all) and overfetch 50 for {short('strategy_overfetch50_history', 10)}, while "
        f"inline always fills k at {r.p50('strategy_inline_history', 10)} ms p50 / "
        f"{r.p95('strategy_inline_history', 10)} ms p95 (overfetch 10: "
        f"{r.p50('strategy_overfetch10_history', 10)} ms, overfetch 50: "
        f"{r.p50('strategy_overfetch50_history', 10)} ms). Article recall@10 under the History "
        f"filter is {r.art('strategy_inline_history', 10)} inline, "
        f"{r.art('strategy_overfetch10_history', 10)} overfetch 10 and "
        f"{r.art('strategy_overfetch50_history', 10)} overfetch 50, of a ceiling of "
        f"{t_his['article_recall_ceiling']}; evidence recall is 0 for all because no claim's "
        f"gold sentences lie in a History section."
    )
    parts += [reading, ""]
    return parts


def summary_restore(r: Results) -> list[str]:
    ks = DEFAULT_KS
    parts = ["## 7. Restored default state (120 words, overlap 1, prefix on, M=6, ef 20)", ""]
    names = ["baseline", "stability_512mb_run1", "ef_20", "m6_rebuilt_ef_20", "final_state_ef_20"]
    parts += [
        md_table(
            ("run", "index built", "cache", *[f"article@{k}" for k in ks],
             *[f"evidence@{k}" for k in ks], "SQL p50@10 ms"),
            [
                [n, (r.get(n) or {}).get("ingest_meta", {}).get("ingested_at", "not found"),
                 _cache_cell(r, n), *[r.art(n, k) for k in ks], *[r.ev(n, k) for k in ks],
                 r.p50(n, 10)]
                for n in names
            ],
        ),
        "",
    ]
    final = r.get("final_state_ef_20")
    if final is None:
        return parts
    p = final["parameters"]
    report = p.get("ingest_report") or {}
    differences = p.get("ingest_meta_differences")
    reading = (
        f"Reading: the final re-ingest ({report.get('counts', {}).get('n_chunks', '?')} chunks, "
        f"{report.get('seconds', {}).get('total', '?')} s) left `ingest_meta` "
        f"{'identical to' if p.get('ingest_meta_matches_baseline') else 'different from'} the "
        f"committed baseline's parameters (ingested_at excepted"
        + (f"; differences: {differences}" if differences else "")
        + f"), the vector index at M={p.get('vector_index_m')} and mhnsw_max_cache_size at "
        f"{p.get('mhnsw_max_cache_size')} bytes. The freshly built index gives article recall@10 "
        f"{r.art('final_state_ef_20', 10)} and evidence recall@20 {r.ev('final_state_ef_20', 20)} "
        f"at ef 20, against {r.art('baseline', 10)} / {r.ev('baseline', 20)} for the committed "
        f"baseline's index and {r.art('ef_20', 10)} / {r.ev('ef_20', 20)} for the same index "
        f"after the restart; its hit lists are identical to the committed baseline's for "
        f"{r.identical('final_state_ef_20', 'baseline')} claims. Same parameters, another HNSW "
        f"graph (section 3): at ef 20 and M=6 the numbers a reader reproduces will differ from "
        f"these by a few claims."
    )
    parts += [reading, ""]
    return parts


def summary_recommendations(r: Results) -> list[str]:
    parts = ["## Recommended defaults", ""]
    need = ("ef_20", "ef_100", "ef_200", "ef_400", "inline_ef_20", "m16_ef_20", "m16_ef_100",
            "m32_ef_20", "m32_ef_100", "chunk60_ef_100", "chunk240_ef_100", "noprefix_ef_100",
            "strategy_inline_history", "strategy_overfetch10_history",
            "strategy_overfetch50_history", "strategy_inline_minwords1000",
            "strategy_overfetch10_minwords1000", "stability_comparison", "index_m_rebuild")
    missing = [n for n in need if r.get(n) is None]
    if missing:
        parts += [
            (f"Not all groups have run yet (missing: {', '.join(missing)}); the recommendations "
            "are written once every result file exists."),
            "",
        ]
        return parts
    comp = r.get("stability_comparison")["comparisons"]
    rebuild = {b["m"]: b for b in r.get("index_m_rebuild")["rebuilds"]}
    parts += [
        ("The current defaults are the provisional ones of docs/DESIGN.md (M=6, ef_search 20, "
        "120-word chunks, prefix on, strategy inline, 512 MB cache). What the numbers say:"),
        "",
        (f"- **Index M = 16** (sql/schema.sql, currently M=6). At ef 20 the M=16 graph returns the "
        f"exact ranking's counts, article recall@10 {r.art('m16_ef_20', 10)} and evidence "
        f"recall@20 {r.ev('m16_ef_20', 20)}, at p50@10 {r.p50('m16_ef_20', 10)} ms, where M=6 "
        f"gives {r.art('ef_20', 10)} / {r.ev('ef_20', 20)} at {r.p50('ef_20', 10)} ms and needs "
        f"ef 200 to 400 ({r.art('ef_200', 10)} / {r.ev('ef_200', 20)} at {r.p50('ef_200', 10)} "
        f"ms, {r.art('ef_400', 10)} / {r.ev('ef_400', 20)} at {r.p50('ef_400', 10)} ms) to get "
        f"there. M=32 adds nothing ({r.art('m32_ef_20', 10)} / {r.ev('m32_ef_20', 20)} at "
        f"{r.p50('m32_ef_20', 10)} ms) and costs {rebuild[32]['add_seconds']} s to build against "
        f"{rebuild[16]['add_seconds']} s for M=16 and {rebuild[6]['add_seconds']} s for M=6; the "
        f"graph tablespace is {fmt_bytes_mb(rebuild[6]['vector_index_tablespace_bytes'])} / "
        f"{fmt_bytes_mb(rebuild[16]['vector_index_tablespace_bytes'])} / "
        f"{fmt_bytes_mb(rebuild[32]['vector_index_tablespace_bytes'])} for M 6 / 16 / 32 "
        f"(section 3)."),
        (f"- **mhnsw_ef_search = 100** per query (`search(ef_search=100)`), with M=16. At M=16 "
        f"ef 100 keeps the exact counts ({r.art('m16_ef_100', 10)} / {r.ev('m16_ef_100', 20)}) "
        f"and brings the hit lists closer to the exact ranking "
        f"({r.identical('inline_ef_20', 'm16_ef_100')} identical claims against "
        f"{r.identical('inline_ef_20', 'm16_ef_20')} at ef 20) for {r.p50('m16_ef_100', 10)} "
        f"instead of {r.p50('m16_ef_20', 10)} ms p50@10; on the M=6 index ef 100 is also where "
        f"recall stops depending strongly on the index state (section 1 and 2: "
        f"{r.identical('ef_search_100', 'ef_100')} identical hit lists between two states at "
        f"ef 100 against {r.identical('baseline', 'ef_20')} at ef 20). If M stays 6, use ef 200 "
        f"({r.art('ef_200', 10)} / {r.ev('ef_200', 20)}, {r.p50('ef_200', 10)} ms)."),
        (f"- **chunk_max_words = 240, overlap 1** (currently 120). At equal retrieved text the "
        f"240-word chunks are never below the others (1,200 nominal words: article "
        f"{r.art('chunk60_ef_100', 20)} / {r.art('ef_100', 10)} / {r.art('chunk240_ef_100', 5)}, "
        f"evidence {r.ev('chunk60_ef_100', 20)} / {r.ev('ef_100', 10)} / "
        f"{r.ev('chunk240_ef_100', 5)} for 60 / 120 / 240; 480 to 600 words: "
        f"{r.ev('chunk60_ef_100', 10)} / {r.ev('ef_100', 5)} / {r.ev('chunk240_ef_100', 2)}), "
        f"the differences are within the index-state noise, and 240 halves the vectors "
        f"({r.param('chunk240_ef_100', 'n_chunks')} against {r.param('ef_100', 'n_chunks')}), "
        f"the graph ({fmt_bytes_mb(r.param('chunk240_ef_100', 'vector_index_tablespace_bytes', None))} "
        f"against {fmt_bytes_mb(r.param('ef_100', 'vector_index_tablespace_bytes', None))}) and "
        f"the rows per query (section 4). 120 remains defensible when a shorter passage is "
        f"wanted for display: the recall difference is not measurable on this corpus."),
        (f"- **Prefix off** (currently on). The prefix gives no measurable gain: at ef 100 the "
        f"no-prefix vectors reach article recall@1 {r.art('noprefix_ef_100', 1)}, @10 "
        f"{r.art('noprefix_ef_100', 10)}, evidence recall@10 {r.ev('noprefix_ef_100', 10)}, @20 "
        f"{r.ev('noprefix_ef_100', 20)} against {r.art('ef_100', 1)}, {r.art('ef_100', 10)}, "
        f"{r.ev('ef_100', 10)}, {r.ev('ef_100', 20)} with the prefix, and they match the exact "
        f"ranking of the prefixed vectors ({r.art('inline_ef_20', 10)}, "
        f"{r.ev('inline_ef_20', 10)}, {r.ev('inline_ef_20', 20)}) although they are themselves "
        f"approximate (section 5). One re-ingest each; confirm with an inline (exact) run on a "
        f"no-prefix index before the README states it."),
        (f"- **Filtered queries: strategy inline** (the current default of `search()`). It always "
        f"fills k: under the selective History filter overfetch 10 returned fewer than 10 rows "
        f"for {r.get('strategy_overfetch10_history')['parameters']['short_results']['10']['queries_short_of_k']} "
        f"of 75 queries and overfetch 50 for "
        f"{r.get('strategy_overfetch50_history')['parameters']['short_results']['10']['queries_short_of_k']}, "
        f"inline for 0, at p50@10 {r.p50('strategy_inline_history', 10)} ms (p95 "
        f"{r.p95('strategy_inline_history', 10)} ms) against "
        f"{r.p50('strategy_overfetch10_history', 10)} / {r.p50('strategy_overfetch50_history', 10)} "
        f"ms. For a non-selective filter overfetch 10 gives the same article recall for "
        f"{r.p50('strategy_overfetch10_minwords1000', 10)} ms instead of "
        f"{r.p50('strategy_inline_minwords1000', 10)} ms, so it is the option when latency "
        f"matters more than a full result (section 6). The inline walk grows with the table "
        f"(docs/DESIGN.md: 24 ms on 12,000 random chunks), which is the number to watch when "
        f"the corpus is scaled."),
        (f"- **mhnsw_max_cache_size = 512 MB** (docker-compose.yml, kept). The cache size did not "
        f"change a single hit list ({comp['after_restart_vs_16mb']['runs'][STABILITY_16_RUNS[0]]['identical_sequence']} of {comp['after_restart_vs_16mb']['n_claims']} hit lists identical "
        f"between 512 MB and 16 MB within one server process); the container restart did "
        f"({comp['after_restart_vs_512mb']['runs'][AFTER_RESTART]['identical_sequence']} of 75 "
        f"identical, section 1). 512 MB is kept for capacity: the 8,868-chunk graph is already "
        f"{fmt_bytes_mb(r.param('ef_100', 'vector_index_tablespace_bytes', None))} on disk, the "
        f"16 MB default would not hold a larger corpus. The restart effect is an open point for "
        f"the README: results at ef 20 / M=6 are reproducible only within one server process."),
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
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed passes per claim after one warm-up (default 5)")
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
