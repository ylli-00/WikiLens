#!/usr/bin/env python3
"""Six behaviours of MariaDB's vector index, measured: five that its documentation does not
describe, and the ``LIMIT`` of a filtered search, which it does.

``forms``
    One nearest-neighbour statement in three shapes, on the ingested corpus (read only): the
    bare statement, the same with a predicate on the indexed table itself, and the same with a
    join to another table. Every hit list is compared with NumPy's exact ranking of the rows
    that pass. Result (MariaDB 11.8.9, 766 and 304 chunks passing, ``mhnsw_ef_search`` 20 and
    100): the bare statement is the HNSW search; a predicate on the table keeps that search
    going until ``k`` rows pass, so no result was short, but it stays approximate and gets
    slower the fewer rows pass; with a join the result was the exact top 10 for every claim at
    both values. Writes ``results/statement_forms.json`` and ``.md``.

``join_order``
    The filtered statement of ``wikilense query`` as written, with ``STRAIGHT_JOIN`` and
    ``chunk`` first, against the same text with plain ``JOIN``, where the optimizer picks the
    order (read only, the filters of the ``planner`` experiment). Every hit list is compared
    with the same statement under ``IGNORE INDEX (embedding)``, a scan of every chunk. Result
    (MariaDB 11.8.9): left free, the optimizer starts from the filter's own index or derived
    table under 8 of the 12 filters, and the vector index is not in those plans; that is much
    cheaper under a selective filter and much dearer under one unselective filter. Writes
    ``results/join_order.json`` and ``.md``.

``builds``
    How much the results move from one build of the HNSW graph to the next: the chunk vectors
    are copied into a scratch table in the *test* database ten times per ``M`` (16 and 6), and
    every build is searched with the claims at ``mhnsw_ef_search`` 20 and 100. Reports the
    spread of article recall, of the agreement with NumPy's exact ranking, and of the agreement
    between two builds. Writes ``results/build_variance.json`` and ``.md``.

``cache``
    The same searches with ``mhnsw_max_cache_size`` at the configured value, at the server
    default of 16 MB and below the size of the graph (8, 4, 2 and 1 MB), for the bare statement
    and the joined one, against NumPy's exact ranking. Changes a global server variable as
    root through ``docker exec`` and puts the configured value back at the end; reads the
    corpus only. Writes ``results/cache_size.json`` and ``.md``.

``isolation``
    What a vector search sees while another connection inserts or deletes near neighbours,
    under each transaction isolation level, and one sequence after which the index gives stale
    answers to every connection: a REPEATABLE READ transaction searches, another connection
    commits a change, the first searches again inside its old snapshot and then ends. Uses a
    scratch table of random vectors in the *test* database (``WIKILENSE_TEST_DB_NAME``), never
    the corpus. ``FLUSH TABLES`` on that table is run as root through ``docker exec`` (the
    route of ``scripts/run_experiments.py``; skipped when docker or the root password is
    missing). Writes ``results/isolation.json`` and ``.md``.

``hard_cap``
    What the ``LIMIT`` of an index search does to a filtered answer, on the ingested corpus
    (read only), at k = 10 (``--k``) and ``mhnsw_ef_search`` 100: a distance threshold in the
    statement's ``WHERE``, the documentation's form with the index search as a derived table
    (inner ``LIMIT`` 10 and 100) and the threshold outside it, and a predicate on ``chunk``
    itself whose word limits pass the counts nearest 0.5% and 0.05% of the chunks (the rule of
    ``scripts/scaled_filters.py``), alone and joined to ``page``. Each claim's threshold lies
    between two neighbours of NumPy's exact ranking, so that exactly 3, 10 or 30 chunks are
    nearer, and a scan of the table without the index counts them again. Reports the outcome
    of every execution, the agreement with the exact list, the chunk rows read and the plan of
    every statement. Writes ``results/hard_cap.json`` after the threshold check and after each
    case, and ``.md`` at the end.

    Runs only with ``--throwaway-server``, and running every part leaves it out without that
    flag: on MariaDB 11.8.9 a vector search that fewer rows pass than ``LIMIT`` can grow
    ``mariadbd`` until it is out of memory. The part is meant for a throwaway MariaDB container
    started with a ``--memory`` cap (and a restart policy, so that the server comes back after
    it is killed), never the project's database server. Every statement after the corpus is
    read has a time limit (``max_statement_time``, ``HARD_CAP_STATEMENT_SECONDS``), and every
    execution is recorded as complete, short, stopped at the limit or with the server lost.
    After a lost server the part waits for the server to accept connections again, reconnects
    and goes on with the next claim; it stops, and still writes its report, after
    ``HARD_CAP_LOST_IN_A_ROW`` lost servers in a row or when the server does not come back
    within ``HARD_CAP_RECONNECT_SECONDS``.

Usage (from the project root, with the database up and the corpus ingested):

    python scripts/measure_index_behaviour.py            # all but hard_cap
    python scripts/measure_index_behaviour.py forms
    python scripts/measure_index_behaviour.py join_order
    python scripts/measure_index_behaviour.py builds
    python scripts/measure_index_behaviour.py cache
    python scripts/measure_index_behaviour.py isolation
    # a throwaway server only (see hard_cap):
    python scripts/measure_index_behaviour.py hard_cap --throwaway-server --database wikilense_s1 \
        --sample 200 --out results/scaled/run1/s1

Do not run it while the test suite or another measurement runs: ``builds`` and ``isolation``
use the test database, and ``cache`` changes a server-wide setting for its duration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from functools import partial
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pymysql

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import compare_separate_index as separate  # the exact top k and the interleaved timing
import scaled_filters  # the rule that picks a filter value for a target share of the chunks

from wikilense import db, evaluate
from wikilense import search as searchmod
from wikilense.config import Settings, load_settings

K = 10
FORMS_REPEATS = 3
FORMS_EF: tuple[int, ...] = (20, 100)
WORD_LIMITS: tuple[int, ...] = (60, 30)

# --- forms: fixed statements on the corpus tables; the word limit and the vector are parameters
FORM_BARE = "SELECT chunk_id FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s"
FORM_BARE_JOINED = (
    "SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id "
    "ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s"
)
FORM_PREDICATE = (
    "SELECT chunk_id FROM chunk WHERE n_words < %s "
    "ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s"
)
FORM_PREDICATE_JOINED = (
    "SELECT chunk.chunk_id FROM chunk STRAIGHT_JOIN page ON page.page_id = chunk.page_id "
    "WHERE chunk.n_words < %s ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s"
)
JOIN_ORDER_REPEATS = 3
#: The seed of ``--sample``: a fixed draw, so a rerun measures the same claims.
SAMPLE_SEED = 20261119
JOIN_ORDER_EF = 100
#: The scan that every hit list is compared with: the statement as written, without the index.
SCAN_FROM = ("FROM chunk ", "FROM chunk IGNORE INDEX (embedding) ")
#: Position of ``distance`` in the inline statement's select list (search.INLINE_SELECT).
DISTANCE_COLUMN = 6
#: Two medians that differ by less than this factor are reported as the same time. Two runs of
#: one statement on this laptop differed by up to 1.24, and two forms with the same plan by 1.10.
NOISE_FACTOR = 1.5
PARTS: tuple[str, ...] = ("forms", "join_order", "builds", "cache", "isolation", "hard_cap")

CACHE_PASSES = 3
CACHE_EF = 100
MEBIBYTE = 1048576
#: The default cache sizes of the ``cache`` part, in MB: the server default first, then smaller.
CACHE_SIZES_MB: tuple[int, ...] = (16, 8, 4, 2, 1)


def cache_set_sql(size_mb: int) -> str:
    """The ``SET GLOBAL`` statement for a cache of ``size_mb`` megabytes.

    A literal statement: a server variable cannot be a bound parameter here, because the
    statement runs through the mariadb client inside the container. The size is an ``int``.
    """
    if isinstance(size_mb, bool) or not isinstance(size_mb, int) or size_mb < 1:
        raise ValueError(f"cache size must be a positive int of megabytes, got {size_mb!r}")
    return "SET GLOBAL mhnsw_max_cache_size = " + str(size_mb * MEBIBYTE)


CACHE_GET_SQL = "SELECT @@GLOBAL.mhnsw_max_cache_size"
CHUNK_FLUSH_SQL = "FLUSH TABLES chunk"

BUILDS = 10
BUILD_EF: tuple[int, ...] = (20, 100)
BUILD_BATCH = 500
BUILD_DROP_SQL = "DROP TABLE IF EXISTS build_probe"
#: The chunk table's vector column and index, once per M. M cannot be a bound parameter.
BUILD_CREATE_SQL: dict[int, str] = {
    16: (
        "CREATE TABLE build_probe (chunk_id INT UNSIGNED NOT NULL PRIMARY KEY, "
        "embedding VECTOR(384) NOT NULL, VECTOR INDEX (embedding) M=16 DISTANCE=cosine) "
        "ENGINE=InnoDB"
    ),
    6: (
        "CREATE TABLE build_probe (chunk_id INT UNSIGNED NOT NULL PRIMARY KEY, "
        "embedding VECTOR(384) NOT NULL, VECTOR INDEX (embedding) M=6 DISTANCE=cosine) "
        "ENGINE=InnoDB"
    ),
}
BUILD_INSERT_SQL = "INSERT INTO build_probe (chunk_id, embedding) VALUES (%s, %s)"
BUILD_KNN_SQL = (
    "SELECT chunk_id FROM build_probe ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s"
)
CHUNK_PAGE_EXPORT_SQL = "SELECT chunk_id, page_id, embedding FROM chunk ORDER BY chunk_id"

CHUNK_COUNT_SQL = "SELECT COUNT(*) FROM chunk"
CHUNK_EXPORT_SQL = "SELECT chunk_id, n_words, embedding FROM chunk ORDER BY chunk_id"

# --- isolation: a scratch table with the chunk table's vector column and index, in the test db
ISOLATION_ROWS = 1000
ISOLATION_EF = 100
ISOLATION_TRIALS = 10
ISOLATION_CONTROL_TRIALS = 5
ISOLATION_SEED = 7
NEW_ROW_IDS: tuple[int, ...] = (90001, 90002, 90003, 90004, 90005)
EXTRA_ROW_ID = 90011
OTHER_QUERIES = 20
PROBE_DROP_SQL = "DROP TABLE IF EXISTS isolation_probe"
PROBE_CREATE_SQL = (
    "CREATE TABLE isolation_probe (probe_id INT UNSIGNED NOT NULL PRIMARY KEY, "
    "embedding VECTOR(384) NOT NULL, VECTOR INDEX (embedding) M=16 DISTANCE=cosine) "
    "ENGINE=InnoDB"
)
PROBE_INSERT_SQL = "INSERT INTO isolation_probe (probe_id, embedding) VALUES (%s, %s)"
PROBE_DELETE3_SQL = "DELETE FROM isolation_probe WHERE probe_id IN (%s, %s, %s)"
PROBE_KNN_SQL = (
    "SELECT probe_id FROM isolation_probe ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s"
)
PROBE_SCAN_SQL = (
    "SELECT probe_id FROM isolation_probe IGNORE INDEX (embedding) "
    "ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s"
)
PROBE_COUNT_SQL = "SELECT COUNT(*) FROM isolation_probe"
PROBE_EF_SQL = "SET SESSION mhnsw_ef_search = %s"
#: How a connection that looks at a stale table may be set up: a level, or autocommit.
VICTIMS: tuple[str, ...] = ("REPEATABLE READ", "READ COMMITTED", "READ UNCOMMITTED", "autocommit")
DELETE_RECOVERY_TRIALS = 3
PROBE_FLUSH_SQL = "FLUSH TABLES isolation_probe"
#: One fixed statement per isolation level (the level is never interpolated from input).
ISOLATION_LEVELS: dict[str, str] = {
    "READ UNCOMMITTED": "SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED",
    "READ COMMITTED": "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED",
    "REPEATABLE READ": "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ",
    "SERIALIZABLE": "SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE",
}
LOCK_WAIT_SQL = "SET SESSION innodb_lock_wait_timeout = 1"
SERVER_STATE_SQL = (
    "SELECT VERSION(), @@GLOBAL.transaction_isolation, @@GLOBAL.innodb_snapshot_isolation"
)

# --- hard_cap: fixed statements on the corpus tables; the vector, the threshold, the limits
# and the word limit are parameters
HARD_CAP_EF = 100
HARD_CAP_REPEATS = 1
#: How many chunks a claim's distance threshold lets through: fewer than, as many as and more
#: than the default k.
HARD_CAP_NS: tuple[int, ...] = (3, 10, 30)
#: The LIMIT of the inner statement in the subquery form.
HARD_CAP_INNER_LIMITS: tuple[int, ...] = (10, 100)
#: The shares of the chunk table the two word limits aim at (two levels of scaled_filters.TARGETS).
HARD_CAP_SHARES: tuple[float, ...] = (0.005, 0.0005)
#: Two neighbouring exact distances at most this far apart give no threshold between them: a
#: tie, or a gap that rounding in the server's distance could cross. The report gives the
#: largest difference measured between a distance the server computed and NumPy's.
TIE_MARGIN = 1e-6
#: The distance threshold inside the statement. Parameters: vector, vector, threshold, vector, k.
HARD_CAP_THRESHOLD_SQL = (
    "SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance FROM chunk "
    "WHERE VEC_DISTANCE_COSINE(embedding, %s) < %s "
    "ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s"
)
#: The index search as a derived table (search.KNN_SQL), the threshold outside it.
#: Parameters: vector, vector, inner limit, threshold, k.
HARD_CAP_SUBQUERY_SQL = (
    "SELECT t.chunk_id, t.distance FROM ("
    "SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance FROM chunk "
    "ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s"
    ") AS t WHERE t.distance < %s ORDER BY t.distance LIMIT %s"
)
#: Every chunk under a threshold, without the index: the server's own count of the chunks that
#: qualify. Parameters: vector, vector, threshold.
HARD_CAP_SCAN_SQL = (
    "SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance "
    "FROM chunk IGNORE INDEX (embedding) WHERE VEC_DISTANCE_COSINE(embedding, %s) < %s"
)
#: The time limit of every statement on the part's connection once the corpus is read, in
#: seconds (``max_statement_time``); the session's own value is read first and written back.
HARD_CAP_STATEMENT_SECONDS = 10
HARD_CAP_TIME_GET_SQL = "SELECT @@SESSION.max_statement_time"
HARD_CAP_TIME_SET_SQL = "SET SESSION max_statement_time = %s"
#: MariaDB's error for a statement stopped at ``max_statement_time`` (ER_STATEMENT_TIMEOUT).
STATEMENT_TIMEOUT_ERROR = 1969
#: Client errors of a connection whose server went away: CR_SERVER_GONE_ERROR (2006),
#: CR_SERVER_LOST (2013) and CR_SERVER_LOST_EXTENDED (2055).
SERVER_LOST_ERRORS: frozenset[int] = frozenset({2006, 2013, 2055})
#: What one execution came to: the expected number of rows, fewer, stopped at the time limit,
#: or the connection to the server lost.
OUTCOMES: tuple[str, ...] = ("complete", "short", "stopped", "server lost")
COMPLETE, SHORT, STOPPED, SERVER_LOST = OUTCOMES
NOT_RUN = "not run"
#: The scan check of a claim whose thresholds were all left out (ties).
NO_THRESHOLD = "no threshold"
#: The part stops at this many lost servers in a row, or when the server does not accept a
#: connection within HARD_CAP_RECONNECT_SECONDS, tried every HARD_CAP_RECONNECT_INTERVAL seconds.
HARD_CAP_LOST_IN_A_ROW = 5
HARD_CAP_RECONNECT_SECONDS = 180.0
HARD_CAP_RECONNECT_INTERVAL = 2.0
#: Why hard_cap runs only with --throwaway-server: the message of the refusal.
HARD_CAP_REFUSAL = (
    "hard_cap runs only with --throwaway-server: on MariaDB 11.8.9 its vector searches that "
    "fewer rows pass than LIMIT can grow mariadbd until it is out of memory. Run it only against "
    "a throwaway MariaDB container started with a --memory cap, never against the project's "
    "database server."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sample_claims(truths: Sequence[Any], sample: int | None, seed: int = SAMPLE_SEED) -> list[Any]:
    """``sample`` of ``truths`` drawn with ``random.Random(seed)``, in their original order;
    all of them when ``sample`` is None or not smaller than their number."""
    if sample is None or sample >= len(truths):
        return list(truths)
    if sample < 1:
        raise ValueError(f"a sample needs at least one claim, got {sample}")
    picked = sorted(random.Random(seed).sample(range(len(truths)), sample))
    return [truths[i] for i in picked]


def median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


# ---------------------------------------------------------------------------------------------
# forms
# ---------------------------------------------------------------------------------------------


def measure_forms(settings: Settings, embedder: Any, repeats: int, k: int = K) -> dict[str, Any]:
    """Run the statement forms on the corpus database and return the report."""
    conn = db.connect(settings)
    try:
        truths = evaluate.load_ground_truth(conn)
        if not truths:
            raise SystemExit("the claim table is empty: run 'wikilense ingest' first")
        with conn.cursor() as cur:
            cur.execute(CHUNK_EXPORT_SQL)
            rows = cur.fetchall()
        ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        words = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
        vectors = separate.unit_rows(
            np.vstack([db.vec_from_bytes(row[2]) for row in rows]).astype(np.float64)
        )
        queries = separate.unit_rows(
            np.asarray(embedder.embed_queries([t.text for t in truths]), dtype=np.float32)
        ).astype(np.float32)
        separate._quiet_torch()
        n_queries = len(truths)

        cases: list[dict[str, Any]] = [
            {"key": "bare", "label": "the bare statement on chunk", "sql": FORM_BARE,
             "limit": None, "joined": False},
            {"key": "bare_joined", "label": "the same, joined to page", "sql": FORM_BARE_JOINED,
             "limit": None, "joined": True},
        ]
        for limit in WORD_LIMITS:
            cases.append({"key": f"predicate_{limit}", "label": f"WHERE chunk.n_words < {limit}",
                          "sql": FORM_PREDICATE, "limit": limit, "joined": False})
            cases.append({"key": f"predicate_{limit}_joined",
                          "label": f"WHERE chunk.n_words < {limit}, joined to page",
                          "sql": FORM_PREDICATE_JOINED, "limit": limit, "joined": True})
        for case in cases:
            mask = np.ones(ids.shape[0], dtype=bool) if case["limit"] is None else words < case["limit"]
            passing = np.flatnonzero(mask)
            case["passing_chunks"] = int(passing.shape[0])
            case["exact"] = [
                separate.top_k(vectors, queries[i].astype(np.float64), ids, k, passing)
                for i in range(n_queries)
            ]

        def answer_of(case: dict[str, Any]) -> Callable[[int], list[int]]:
            def answer(i: int) -> list[int]:
                vector = db.vec_param(queries[i])
                params = (vector, k) if case["limit"] is None else (case["limit"], vector, k)
                with conn.cursor() as cur:
                    cur.execute(case["sql"], params)
                    return [int(row[0]) for row in cur.fetchall()]

            return answer

        def rows_read(case: dict[str, Any]) -> float:
            """Median over the claims of the chunk rows ANALYZE reports as read (r_rows)."""
            read: list[float] = []
            with conn.cursor() as cur:
                for i in range(n_queries):
                    vector = db.vec_param(queries[i])
                    params = (vector, k) if case["limit"] is None else (case["limit"], vector, k)
                    cur.execute("ANALYZE " + case["sql"], params)
                    plan = cur.fetchall()
                    read.append(float(plan[0][9]))  # r_rows of the first table, chunk
            return median(read)

        def chunk_access(case: dict[str, Any]) -> str:
            """'<type> <key>' of the chunk row in EXPLAIN, for the first claim."""
            vector = db.vec_param(queries[0])
            params = (vector, k) if case["limit"] is None else (case["limit"], vector, k)
            row = next(r for r in db.explain(conn, case["sql"], params) if r["table"] == "chunk")
            return f"{row['type']} {row['key'] or 'no key'}"

        results: list[dict[str, Any]] = []
        for ef in FORMS_EF:
            with searchmod.ef_search_session(conn, ef):
                timed = separate.time_methods([answer_of(case) for case in cases], n_queries, repeats)
                for case, (samples, lists) in zip(cases, timed, strict=True):
                    stats = evaluate.latency_stats(samples)
                    pairs = list(zip(lists, case["exact"], strict=True))
                    results.append({
                        "form": case["key"], "label": case["label"], "joined": case["joined"],
                        "words_below": case["limit"], "passing_chunks": case["passing_chunks"],
                        "ef_search": ef,
                        "p50_ms": round(stats.p50_ms, 3), "p95_ms": round(stats.p95_ms, 3),
                        "equal_to_exact": sum(1 for got, want in pairs if got == want),
                        "same_set_as_exact": sum(1 for got, want in pairs if set(got) == set(want)),
                        "short": sum(1 for got, want in pairs if len(got) < len(want)),
                        "chunk_rows_read_p50": rows_read(case),
                        "chunk_access": chunk_access(case),
                    })  # fmt: skip
        return {
            "name": "statement_forms", "created_at": utc_now(), "k": k, "repeats": repeats,
            "n_queries": n_queries, "n_chunks": int(ids.shape[0]),
            **db.vector_index_info(conn),
            "results": results,
            "ingest_meta": evaluate.read_ingest_meta(conn),
            "machine": evaluate.machine_info(embedder), "versions": evaluate.versions_info(conn),
        }  # fmt: skip
    finally:
        conn.close()


def forms_markdown(report: dict[str, Any]) -> str:
    n = report["n_queries"]
    rows = report["results"]
    efs = sorted({row["ef_search"] for row in rows})

    def per_ef(selected: list[dict[str, Any]], key: str) -> str:
        """'a at ef x and b at ef y' for one form, or a list of values for several."""
        parts = []
        for ef in efs:
            values = [row[key] for row in selected if row["ef_search"] == ef]
            parts.append(f"{' and '.join(str(v) for v in values)} at `mhnsw_ef_search` {ef}")
        return ", ".join(parts)

    bare = [row for row in rows if row["form"] == "bare"]
    predicate = [row for row in rows if row["words_below"] is not None and not row["joined"]]
    joined = [row for row in rows if row["joined"]]
    passing = sorted({row["passing_chunks"] for row in predicate}, reverse=True)
    lines = [
        "# One nearest-neighbour statement in three shapes",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py forms` on "
        f"MariaDB {report['versions']['mariadb']}: {report['n_chunks']:,} chunks, index M = "
        f"{report['index_m']}, the {n} corpus claims as queries, k = {report['k']}, "
        f"{report['repeats']} timed passes after one untimed pass, the forms taking turns query "
        "by query. \"Equal to exact\" counts the claims whose hit list is, in order, NumPy's exact "
        f"top {report['k']} of the chunks that pass the predicate. \"Rows read\" is the median over the claims "
        "of `r_rows` for `chunk` in `ANALYZE`. \"Chunk access\" is the type and key of `chunk` in "
        "`EXPLAIN`. The `mhnsw_ef_search` values were run one after the other, so compare times "
        "inside one value."),
        "",
        ("| Statement | Chunks passing | ef_search | p50 ms | p95 ms | Equal to exact | Short "
        "| Rows read | Chunk access |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['passing_chunks']:,} | {row['ef_search']} | "
            f"{row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['equal_to_exact']}/{n} | "
            f"{row['short']}/{n} | {row['chunk_rows_read_p50']:.0f} | {row['chunk_access']} |"
        )
    lines += [
        "",
        "How to read it:",
        "",
        ("- The bare statement is the HNSW search: it reads `LIMIT` rows. On the graph now in the "
        f"database its list was the exact one for {per_ef(bare, 'equal_to_exact')} of {n} claims. "
        "Other builds of the graph differ (the `builds` part measures by how much)."),
        ("- A predicate on the indexed table itself does not stop that search at `LIMIT`: the "
        "server keeps taking rows from the index until `k` of them pass. "
        f"{sum(row['short'] for row in predicate)} of the {len(predicate) * n} answers were short "
        f"(with {' and '.join(f'{count:,}' for count in passing)} chunks passing; nothing more "
        "selective was run). The list was the exact one for "
        f"{per_ef(predicate, 'equal_to_exact')} of {n} claims, in the order of the table."),
        ("- With a join in the statement the list was the exact one for "
        f"{per_ef(joined, 'equal_to_exact')} of {n} claims, in the order of the table."),
        "",
        "The statements (the word limit, the query vector and `k` are bound parameters):",
        "",
        "```sql",
        FORM_BARE + ";",
        FORM_BARE_JOINED + ";",
        FORM_PREDICATE + ";",
        FORM_PREDICATE_JOINED + ";",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# join order
# ---------------------------------------------------------------------------------------------


def free_order(sql: str) -> str:
    """The inline statement with every ``STRAIGHT_JOIN`` turned into ``JOIN``."""
    keyword = searchmod.INLINE_JOIN_KEYWORD
    if keyword != "STRAIGHT_JOIN" or keyword not in sql:
        raise ValueError("the inline statement no longer fixes its join order")
    return sql.replace(keyword, "JOIN")


def scan_form(sql: str) -> str:
    """The inline statement without the vector index: every chunk is read and sorted."""
    old, new = SCAN_FROM
    if sql.count(old) != 1:
        raise ValueError("expected exactly one 'FROM chunk ' in the inline statement")
    return sql.replace(old, new)


def without_distinct(sql: str) -> str:
    """The inline statement with its derived tables written without ``DISTINCT``."""
    if "SELECT DISTINCT" not in sql:
        raise ValueError("the statement has no DISTINCT derived table")
    return sql.replace("SELECT DISTINCT", "SELECT")


def plan_label(rows: Sequence[dict[str, Any]]) -> str:
    """'<first table> first; chunk <access type> <key>' of one EXPLAIN result."""
    chunk = next(row for row in rows if row["table"] == "chunk")
    return f"{rows[0]['table']} first; chunk {chunk['type']} {chunk['key'] or 'no key'}"


def chunk_rows_read(node: Any) -> float:
    """Rows of ``chunk`` read according to ``ANALYZE FORMAT=JSON``: r_loops times r_rows."""
    if isinstance(node, dict):
        own = 0.0
        if node.get("table_name") == "chunk":
            own = float(node.get("r_loops") or 0) * float(node.get("r_rows") or 0)
        return own + sum(chunk_rows_read(value) for value in node.values())
    if isinstance(node, list):
        return sum(chunk_rows_read(value) for value in node)
    return 0.0


def agreement(
    lists: Sequence[Sequence[tuple[int, float]]], scans: Sequence[Sequence[tuple[int, float]]]
) -> dict[str, Any]:
    """How the hit lists (chunk_id, distance) compare with the scan's, over all queries.

    ``swap_gap_max`` is, over every position where the two lists name different chunks although
    they hold the same set, the largest difference between the two distances reported there:
    how close two neighbours were when the server returned them in the other order.
    ``not_ascending`` counts the lists whose rows are not in ascending order of the distance
    the statement itself reports for them.
    """
    same_rows = same_order = short = not_ascending = 0
    gap = 0.0
    for got, want in zip(lists, scans, strict=True):
        got_ids, want_ids = [hit[0] for hit in got], [hit[0] for hit in want]
        short += int(len(got_ids) < len(want_ids))
        distances = [hit[1] for hit in got]
        not_ascending += int(any(b < a for a, b in pairwise(distances)))
        if set(got_ids) != set(want_ids):
            continue
        same_rows += 1
        same_order += int(got_ids == want_ids)
        for mine, theirs in zip(got, want, strict=True):
            if mine[0] != theirs[0]:
                gap = max(gap, abs(mine[1] - theirs[1]))
    return {"same_rows_as_scan": same_rows, "same_order_as_scan": same_order, "short": short,
            "swap_gap_max": gap, "not_ascending": not_ascending}


def measure_join_order(
    settings: Settings,
    embedder: Any,
    repeats: int,
    k: int = K,
    cases: Sequence[tuple[str, searchmod.Filters, int | None]] | None = None,
    sample: int | None = None,
    seed: int = SAMPLE_SEED,
    taking_turns: bool = True,
) -> dict[str, Any]:
    """Run the filtered statement with its join order fixed and free; return the report.

    ``cases`` are ``(name, filters, expected passing chunks or None)``; the default is the
    filters of the ``planner`` experiment (results/planner_choices.md). A case whose statement
    keeps another number of chunks than expected stops the run. ``sample`` takes that many
    claims, drawn with ``random.Random(seed)`` and kept in claim order, instead of all of them.
    Without ``taking_turns`` each form is timed alone only.
    """
    if cases is None:
        from run_experiments import PLANNER_CASES

        cases = [(name, filters, None) for name, filters in PLANNER_CASES]

    conn = db.connect(settings)
    try:
        truths = evaluate.load_ground_truth(conn)
        if not truths:
            raise SystemExit("the claim table is empty: run 'wikilense ingest' first")
        n_claims_total = len(truths)
        truths = sample_claims(truths, sample, seed)
        queries = np.asarray(embedder.embed_queries([t.text for t in truths]), dtype=np.float32)
        separate._quiet_torch()
        n_queries = len(truths)
        with conn.cursor() as cur:
            cur.execute(CHUNK_COUNT_SQL)
            n_chunks = int(cur.fetchone()[0])

        def hits_of(sql: str, params: tuple[Any, ...]) -> list[tuple[int, float]]:
            """(chunk_id, distance) in the order the server returned the rows."""
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [(int(row[0]), float(row[DISTANCE_COLUMN])) for row in cur.fetchall()]

        results: list[dict[str, Any]] = []
        merged_tables: list[dict[str, Any]] = []
        with searchmod.ef_search_session(conn, JOIN_ORDER_EF):
            for name, filters, expected in cases:
                statements: list[tuple[str, tuple[Any, ...], str, str]] = []
                for query in queries:
                    sql, params = searchmod.search_statement(query, k, filters, "inline")
                    statements.append((sql, params, free_order(sql), scan_form(sql)))

                def answer_of(form: int, stmts: list = statements) -> Callable[[int], list[Any]]:
                    return lambda i: hits_of(stmts[i][form], stmts[i][1])

                scans = [hits_of(stmt[3], stmt[1]) for stmt in statements]
                sql0, params0 = searchmod.search_statement(queries[0], n_chunks, filters, "inline")
                passing = len(hits_of(scan_form(sql0), params0))
                if expected is not None and passing != expected:
                    raise SystemExit(f"filter {name!r}: the statement keeps {passing} chunks, "
                                     f"the filter file says {expected}")
                timed: list[Any] = (
                    separate.time_methods([answer_of(0), answer_of(2)], n_queries, repeats)
                    if taking_turns else [None, None]
                )
                # Each form alone as well: when the two take turns, a form that reads thousands
                # of rows slows the other one's next run (measured: 2.2 against 2.6 ms).
                alone = [
                    separate.time_methods([answer_of(form)], n_queries, repeats)[0]
                    for form in (0, 2)
                ]
                row: dict[str, Any] = {"filter": name, "passing_chunks": passing}
                for key, form, turn, (own, own_lists) in zip(
                    ("as_written", "free"), (0, 2), timed, alone, strict=True
                ):
                    lists = own_lists if turn is None else turn[1]
                    plans: Counter[str] = Counter()
                    read: list[float] = []
                    first_plan = [
                        {key_: plan_row[key_] for key_ in ("table", "type", "key", "Extra")}
                        for plan_row in db.explain(conn, statements[0][form], statements[0][1])
                    ]
                    with conn.cursor() as cur:
                        for stmt in statements:
                            plans[plan_label(db.explain(conn, stmt[form], stmt[1]))] += 1
                            cur.execute("ANALYZE FORMAT=JSON " + stmt[form], stmt[1])
                            # r_rows is an average per loop: the product is a whole number
                            read.append(float(round(chunk_rows_read(json.loads(cur.fetchone()[0])))))
                    stats = evaluate.latency_stats(own)
                    turns = None if turn is None else evaluate.latency_stats(turn[0])
                    plan, plan_count = plans.most_common(1)[0]
                    row[key] = {
                        "plan": plan, "plan_count": plan_count, "plans": dict(plans),
                        "explain_first_claim": first_plan,
                        "chunk_rows_read_p50": median(read), "chunk_rows_read_max": max(read),
                        "p50_ms": round(stats.p50_ms, 3), "p95_ms": round(stats.p95_ms, 3),
                        "p50_ms_taking_turns": None if turns is None else round(turns.p50_ms, 3),
                        **agreement(lists, scans),
                    }  # fmt: skip
                results.append(row)

                # The filters that join a derived table: the same statement without DISTINCT.
                if "SELECT DISTINCT" not in statements[0][0]:
                    continue
                merged_plans: Counter[str] = Counter()
                derived = duplicates = 0
                for stmt in statements:
                    merged_sql = without_distinct(stmt[0])
                    plan_rows = db.explain(conn, merged_sql, stmt[1])
                    merged_plans[plan_label(plan_rows)] += 1
                    derived += int(any(str(r["table"]).startswith("<derived") for r in plan_rows))
                    ids = [hit[0] for hit in hits_of(merged_sql, stmt[1])]
                    duplicates += int(len(set(ids)) < len(ids))
                merged_tables.append({
                    "filter": name, "plans": dict(merged_plans),
                    "plans_with_a_derived_table": derived,
                    "lists_with_a_chunk_twice": duplicates,
                })  # fmt: skip
        return {
            "name": "join_order", "created_at": utc_now(), "k": k, "repeats": repeats,
            "n_claims_total": n_claims_total, "sample": sample, "seed": seed if sample else None,
            "taking_turns": taking_turns,
            "ef_search": JOIN_ORDER_EF, "n_queries": n_queries, "n_chunks": n_chunks,
            **db.vector_index_info(conn),
            "results": results,
            "derived_without_distinct": merged_tables,
            "filters": {
                name: {key: (str(value) if isinstance(value, date) else value)
                       for key, value in asdict(filters).items() if value is not None}
                for name, filters, _ in cases
            },
            "ingest_meta": evaluate.read_ingest_meta(conn),
            "machine": evaluate.machine_info(embedder), "versions": evaluate.versions_info(conn),
        }  # fmt: skip
    finally:
        conn.close()


def join_order_markdown(report: dict[str, Any]) -> str:
    n = report["n_queries"]
    rows = report["results"]

    def cells(side: dict[str, Any]) -> str:
        plan = side["plan"] if side["plan_count"] == n else f"{side['plan']} ({side['plan_count']}/{n})"
        turns = side.get("p50_ms_taking_turns")
        return (f"{plan} | {side['chunk_rows_read_p50']:.0f} | {side['p50_ms']:.2f} "
                f"{'' if turns is None else f'({turns:.2f}) '}| "
                f"{side['same_rows_as_scan']} / {side['same_order_as_scan']} |")

    turns_on = report.get("taking_turns", True)
    if report.get("sample"):
        claims = (f"a random sample of {n} of the {report['n_claims_total']:,} corpus claims as "
                  f"queries (`random.Random({report['seed']})`)")
    else:
        claims = f"the {n} corpus claims as queries"
    turns_text = (
        "In brackets is the same median when the two forms take turns query by query. Taking "
        "turns guards against the machine drifting between two blocks, but a form that reads "
        "thousands of rows slows down the other form's next run, so the bracketed time of the "
        "faster form is the larger one."
        if turns_on else "Each form was timed alone only, the two forms one after the other."
    )
    p50_head = "p50 ms (taking turns)" if turns_on else "p50 ms"

    lines = [
        "# The filtered statement with its join order fixed, and left to the optimizer",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py join_order` on "
        f"MariaDB {report['versions']['mariadb']}: {report['n_chunks']:,} chunks, index M = "
        f"{report['index_m']}, {claims}, k = {report['k']}, "
        f"`mhnsw_ef_search` {report['ef_search']}. \"As written\" is the "
        "statement `wikilense query` runs: `STRAIGHT_JOIN`, `chunk` first. \"Free\" is the same "
        "text with every `STRAIGHT_JOIN` replaced by `JOIN`, so the optimizer picks the order. "
        "\"Plan\" is the first table of `EXPLAIN` and how `chunk` is read; `index embedding` is "
        "the vector index. \"Rows read\" is the median over the claims of the `chunk` rows read "
        "(`r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`). \"Same rows / order\" counts the "
        f"claims, of {n}, whose hit list holds the same chunks, and the same chunks in the same "
        "order, as the same statement under `IGNORE INDEX (embedding)`, which reads and sorts "
        "every chunk. \"p50 ms\" is the median time of the statement and the fetch of its rows "
        f"(without the Python sort that `wikilense` adds) when the form runs alone: one untimed "
        f"pass over the claims, then {report['repeats']} timed passes. " + turns_text),
        "",
        (f"| Filter | Chunks passing | As written: plan | Rows read | {p50_head} "
        f"| Same rows / order | Free: plan | Rows read | {p50_head} | Same rows / order |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['filter']} | {row['passing_chunks']:,} | "
                     f"{cells(row['as_written'])} {cells(row['free'])}")

    walked = [row for row in rows if row["as_written"]["plan"].endswith("index embedding")]
    moved = [row for row in rows if not row["free"]["plan"].startswith("chunk first")]

    def ratio(row: dict[str, Any]) -> float:
        return row["free"]["p50_ms"] / row["as_written"]["p50_ms"]

    cheaper = [row for row in moved if ratio(row) < 1 / NOISE_FACTOR]
    dearer = [row for row in moved if ratio(row) > NOISE_FACTOR]
    level = [row for row in moved if row not in cheaper and row not in dearer]

    def pairs(group: list[dict[str, Any]]) -> str:
        return ", ".join(
            f"`{row['filter']}` {row['as_written']['p50_ms']:.2f} to {row['free']['p50_ms']:.2f} ms"
            for row in group
        ) or "none"

    lines += [
        "",
        "How to read it:",
        "",
        (f"- As written, `chunk` is always the first table. Under {len(walked)} of the {len(rows)} "
        f"filters the server walks the vector index; under the other {len(rows) - len(walked)} it "
        "reads every chunk and sorts."),
        (f"- Left free, the optimizer starts from another table under {len(moved)} of the "
        f"{len(rows)} filters: `page`, or the derived table of the link or category filter. It "
        "then reads `chunk` through `uq_chunk_page_ordinal`, as many rows as pass the filter, "
        "and the vector index is not in that plan. The JSON file holds every `EXPLAIN` row of "
        "the first claim for both forms (`explain_first_claim`)."),
        (f"- The plan moved and the median got faster by more than a factor of {NOISE_FACTOR}, as "
        f"written to free: {pairs(cheaper)}."),
        f"- The plan moved and it got slower by more than that: {pairs(dearer)}.",
        f"- The plan moved and the time stayed within that factor: {pairs(level)}.",
    ]
    swapped = [
        (row["filter"], key, side) for row in rows
        for key, side in (("as written", row["as_written"]), ("free", row["free"]))
        if side["same_order_as_scan"] < side["same_rows_as_scan"]
    ]
    if swapped:
        notes = "; ".join(
            f"`{name}` {key}: {side['same_rows_as_scan'] - side['same_order_as_scan']} of {n} "
            f"claims, {side['not_ascending']} of them with rows not in ascending order of the "
            f"distance the statement reports, largest gap {side['swap_gap_max']:.1e}"
            for name, key, side in swapped
        )
        lines.append(
            "- Same chunks in another order than the scan. The gap is the largest difference "
            "between the distances the two lists report at a position where they name different "
            f"chunks ({notes}). `wikilense` sorts the rows by the reported distance and the chunk "
            "id afterwards."
        )
    else:
        lines.append("- No hit list had its chunks in another order than the scan.")
    merged = report.get("derived_without_distinct") or []
    if merged:
        lines += [
            "",
            "## The derived tables without DISTINCT",
            "",
            ("The link and category filters join a `SELECT DISTINCT` derived table. The same "
            f"statement as written, with every `SELECT DISTINCT` turned into `SELECT`, for the {n} "
            "claims: the plan, how many plans still hold a materialised derived table, and how "
            "many hit lists name the same chunk more than once."),
            "",
            "| Filter | Plan | Plans with a derived table | Lists with a chunk twice |",
            "|---|---|---|---|",
        ]
        for row in merged:
            plans = "; ".join(f"{plan} ({count}/{n})" for plan, count in row["plans"].items())
            lines.append(f"| {row['filter']} | {plans} | {row['plans_with_a_derived_table']}/{n} | "
                         f"{row['lists_with_a_chunk_twice']}/{n} |")
        lines += [
            "",
            ("Without `DISTINCT` MariaDB merges the derived table into the join. Whether the "
            "vector index survives is in the plan column; a list with a chunk twice means the "
            "merged join returned one chunk once per matching row of the filter table."),
        ]
    lines += ["", "Filters:", ""]
    for name, filters in report["filters"].items():
        lines.append(f"- `{name}`: {json.dumps(filters, ensure_ascii=False)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# builds
# ---------------------------------------------------------------------------------------------


def spread(values: Sequence[float]) -> dict[str, float]:
    """Smallest, median and largest of ``values``."""
    return {"min": float(min(values)), "median": median(values), "max": float(max(values))}


def identical_between_builds(builds: Sequence[Sequence[Sequence[int]]]) -> list[int]:
    """For every pair of builds, the number of queries whose hit lists are identical."""
    return [
        sum(1 for mine, theirs in zip(first, second, strict=True) if list(mine) == list(theirs))
        for first, second in combinations(builds, 2)
    ]


def measure_builds(
    settings: Settings, embedder: Any, builds: int, k: int = K, ms: Sequence[int] | None = None
) -> dict[str, Any]:
    """Build the index ``builds`` times per M on a scratch table and search each build.

    ``ms`` picks the values of ``M`` (each must be a key of ``BUILD_CREATE_SQL``); the default
    is all of them.
    """
    ms = tuple(BUILD_CREATE_SQL) if ms is None else tuple(ms)
    unknown = [m for m in ms if m not in BUILD_CREATE_SQL]
    if unknown:
        raise SystemExit(f"no scratch-table statement for M = {unknown}; choose from {sorted(BUILD_CREATE_SQL)}")
    if settings.test_db_name == settings.db_name:
        raise SystemExit("the builds experiment needs a test database other than the corpus")
    corpus = db.connect(settings)
    try:
        truths = evaluate.load_ground_truth(corpus)
        if not truths:
            raise SystemExit("the claim table is empty: run 'wikilense ingest' first")
        with corpus.cursor() as cur:
            cur.execute(CHUNK_PAGE_EXPORT_SQL)
            rows = cur.fetchall()
        meta = evaluate.read_ingest_meta(corpus)
        versions = evaluate.versions_info(corpus)
    finally:
        corpus.close()
    ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    page_of = {int(row[0]): int(row[1]) for row in rows}
    stored = [(int(row[0]), bytes(row[2])) for row in rows]
    vectors = separate.unit_rows(
        np.vstack([db.vec_from_bytes(row[2]) for row in rows]).astype(np.float64)
    )
    queries = separate.unit_rows(
        np.asarray(embedder.embed_queries([t.text for t in truths]), dtype=np.float32)
    ).astype(np.float32)
    separate._quiet_torch()
    everything = np.arange(ids.shape[0])
    exact = [
        separate.top_k(vectors, queries[i].astype(np.float64), ids, k, everything)
        for i in range(len(truths))
    ]

    def recalled(lists: Sequence[Sequence[int]]) -> int:
        return sum(
            1 for hits, truth in zip(lists, truths, strict=True)
            if evaluate.article_recall_at_k([page_of[c] for c in hits], truth.gold_pages, k)
        )

    conn = db.connect(settings, database=settings.test_db_name)
    results: list[dict[str, Any]] = []
    try:
        for m in ms:
            create_sql = BUILD_CREATE_SQL[m]
            lists: dict[int, list[list[list[int]]]] = {ef: [] for ef in BUILD_EF}
            seconds: list[float] = []
            for _ in range(builds):
                started = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(BUILD_DROP_SQL)
                    cur.execute(create_sql)
                    for start in range(0, len(stored), BUILD_BATCH):
                        cur.executemany(BUILD_INSERT_SQL, stored[start : start + BUILD_BATCH])
                conn.commit()
                seconds.append(time.perf_counter() - started)
                for ef in BUILD_EF:
                    with searchmod.ef_search_session(conn, ef), conn.cursor() as cur:
                        found = []
                        for query in queries:
                            cur.execute(BUILD_KNN_SQL, (db.vec_param(query), k))
                            found.append([int(row[0]) for row in cur.fetchall()])
                    lists[ef].append(found)
                conn.commit()
            for ef in BUILD_EF:
                per_build = lists[ef]
                results.append({
                    "m": m, "ef_search": ef, "builds": builds,
                    "article_recall": spread([recalled(found) for found in per_build]),
                    "equal_to_exact": spread([
                        sum(1 for got, want in zip(found, exact, strict=True) if got == want)
                        for found in per_build
                    ]),
                    "overlap_with_exact": spread([
                        float(np.mean([len(set(got) & set(want)) / len(want)
                                       for got, want in zip(found, exact, strict=True)]))
                        for found in per_build
                    ]),
                    # One build has no pair to compare.
                    "identical_between_two_builds": (
                        spread(identical_between_builds(per_build)) if builds > 1 else None
                    ),
                    "build_seconds": spread(seconds),
                })  # fmt: skip
        with conn.cursor() as cur:
            cur.execute(BUILD_DROP_SQL)
        conn.commit()
    finally:
        conn.close()
    return {
        "name": "build_variance", "created_at": utc_now(), "k": k, "ms": list(ms), "n_queries": len(truths),
        "n_chunks": int(ids.shape[0]), "builds": builds, "batch": BUILD_BATCH,
        "exact_article_recall": recalled(exact),
        "results": results, "ingest_meta": meta,
        "machine": evaluate.machine_info(embedder), "versions": versions,
    }  # fmt: skip


def builds_markdown(report: dict[str, Any]) -> str:
    n = report["n_queries"]

    def whole(values: dict[str, float] | None) -> str:
        if values is None:
            return "-"
        low, mid, high = values["min"], values["median"], values["max"]
        if low == high:
            return f"{low:g}"
        return f"{low:g} to {high:g} (median {mid:g})"

    lines = [
        "# The same vectors, built into the HNSW graph several times",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py builds` on "
        f"MariaDB {report['versions']['mariadb']}. The {report['n_chunks']:,} chunk vectors of "
        "the corpus are copied, in `chunk_id` order and in one transaction, into a scratch table "
        "with the chunk table's vector column and index, in the test database. That is done "
        f"{report['builds']} times for each `M`, and every build is searched with the {n} corpus "
        f"claims, k = {report['k']}, the bare statement. \"Article recall\" counts the claims "
        "with a gold page among the pages of the hits; NumPy's exact ranking of the same vectors "
        f"gives {report['exact_article_recall']} of {n}. \"Equal to exact\" counts the claims "
        f"whose hit list is NumPy's exact top {report['k']} in order. \"Two builds agree\" counts, for each "
        "pair of builds, the claims with identical hit lists. Each cell is the smallest to the "
        "largest value over the builds (or pairs), one number when they are all the same."),
        "",
        (f"| M | ef_search | Article recall@{report['k']}, of {n} | Equal to exact, of {n} "
        f"| Overlap with exact | Two builds agree, of {n} | Build, s |"),
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        overlap = row["overlap_with_exact"]
        seconds = row["build_seconds"]
        lines.append(
            f"| {row['m']} | {row['ef_search']} | {whole(row['article_recall'])} | "
            f"{whole(row['equal_to_exact'])} | {overlap['min']:.3f} to {overlap['max']:.3f} | "
            f"{whole(row['identical_between_two_builds'])} | "
            f"{seconds['min']:.1f} to {seconds['max']:.1f} |"
        )
    lines += [
        "",
        "The statements (`M` is part of the table definition; the vector and `k` are bound):",
        "",
        "```sql",
        *(BUILD_CREATE_SQL[m] + ";" for m in report.get("ms", BUILD_CREATE_SQL)),
        BUILD_INSERT_SQL + ";",
        BUILD_KNN_SQL + ";",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------------


def measure_cache(
    settings: Settings,
    embedder: Any,
    passes: int,
    k: int = K,
    sizes_mb: Sequence[int] = CACHE_SIZES_MB,
) -> dict[str, Any]:
    """Search with the index cache at several sizes; return the report.

    ``sizes_mb`` are the sizes set after the configured one, in the order given; the smallest
    is also measured once without reopening the table. The configured
    ``mhnsw_max_cache_size`` is read first and restored at the end, also when a step fails.
    """
    sizes_mb = tuple(int(size) for size in sizes_mb)
    if not sizes_mb or min(sizes_mb) < 1:
        raise SystemExit("the cache sizes must be whole megabytes, at least 1")
    import run_experiments  # the route to root inside the container

    try:
        server = run_experiments.Server(settings)
        configured = int(server.root_sql(CACHE_GET_SQL)[0][0])
    except run_experiments.ExperimentError as exc:
        raise SystemExit(f"the cache part needs root in the container: {exc}") from exc
    conn = db.connect(settings)
    try:
        truths = evaluate.load_ground_truth(conn)
        if not truths:
            raise SystemExit("the claim table is empty: run 'wikilense ingest' first")
        with conn.cursor() as cur:
            cur.execute(CHUNK_PAGE_EXPORT_SQL)
            rows = cur.fetchall()
        meta = evaluate.read_ingest_meta(conn)
        versions = evaluate.versions_info(conn)
        index = db.vector_index_info(conn)
    finally:
        conn.close()
    ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    page_of = {int(row[0]): int(row[1]) for row in rows}
    vectors = separate.unit_rows(
        np.vstack([db.vec_from_bytes(row[2]) for row in rows]).astype(np.float64)
    )
    queries = separate.unit_rows(
        np.asarray(embedder.embed_queries([t.text for t in truths]), dtype=np.float32)
    ).astype(np.float32)
    separate._quiet_torch()
    everything = np.arange(ids.shape[0])
    exact = [
        separate.top_k(vectors, queries[i].astype(np.float64), ids, k, everything)
        for i in range(len(truths))
    ]

    def step(label: str, cache_bytes: int, reopened: bool) -> list[dict[str, Any]]:
        """``passes`` passes over the claims with both statements at the current cache size."""
        found: dict[str, list[list[list[int]]]] = {"none": [], "inline": []}
        times: dict[str, list[float]] = {"none": [], "inline": []}
        reader = db.connect(settings)
        try:
            with searchmod.ef_search_session(reader, CACHE_EF):
                for _ in range(passes):
                    for strategy in ("none", "inline"):
                        lists = []
                        for query in queries:
                            started = time.perf_counter()
                            hits = searchmod.search(reader, query, k=k, strategy=strategy)
                            times[strategy].append((time.perf_counter() - started) * 1000.0)
                            lists.append([hit.chunk_id for hit in hits])
                        found[strategy].append(lists)
        finally:
            reader.close()
        out = []
        for strategy, statement in (("none", "bare"), ("inline", "joined")):
            per_pass = found[strategy]
            out.append({
                "cache": label, "cache_bytes": cache_bytes, "table_reopened": reopened,
                "statement": statement,
                "equal_to_exact": [
                    sum(1 for got, want in zip(lists, exact, strict=True) if got == want)
                    for lists in per_pass
                ],
                "article_recall": [
                    sum(1 for hits, truth in zip(lists, truths, strict=True)
                        if evaluate.article_recall_at_k([page_of[c] for c in hits],
                                                        truth.gold_pages, k))
                    for lists in per_pass
                ],
                "overlap_with_exact": [
                    round(float(np.mean([len(set(got) & set(want)) / len(want)
                                         for got, want in zip(lists, exact, strict=True)])), 4)
                    for lists in per_pass
                ],
                "short": [sum(1 for got in lists if len(got) < k) for lists in per_pass],
                "identical_between_two_passes": identical_between_builds(per_pass),
                "p50_ms": round(evaluate.latency_stats(times[strategy]).p50_ms, 3),
            })  # fmt: skip
        return out

    results: list[dict[str, Any]] = []
    restored = None
    try:
        results += step("as configured", configured, reopened=False)
        smallest = min(sizes_mb)
        server.root_sql(cache_set_sql(smallest))
        results += step(f"{smallest} MB, table not reopened", smallest * MEBIBYTE, reopened=False)
        for size in sizes_mb:
            server.root_sql(cache_set_sql(size))
            server.root_sql(CHUNK_FLUSH_SQL, database=settings.db_name)
            results += step(f"{size} MB", size * MEBIBYTE, reopened=True)
    finally:
        server.root_sql("SET GLOBAL mhnsw_max_cache_size = " + str(int(configured)))
        server.root_sql(CHUNK_FLUSH_SQL, database=settings.db_name)
        restored = int(server.root_sql(CACHE_GET_SQL)[0][0])
    results += step("as configured, again", restored, reopened=True)
    return {
        "name": "cache_size", "created_at": utc_now(), "k": k, "sizes_mb": list(sizes_mb), "passes": passes,
        "ef_search": CACHE_EF, "n_queries": len(truths), "n_chunks": int(ids.shape[0]),
        "configured_cache_bytes": configured, "restored_cache_bytes": restored,
        "exact_article_recall": sum(
            1 for hits, truth in zip(exact, truths, strict=True)
            if evaluate.article_recall_at_k([page_of[c] for c in hits], truth.gold_pages, k)
        ),
        **index, "results": results, "ingest_meta": meta,
        "machine": evaluate.machine_info(embedder), "versions": versions,
    }  # fmt: skip


def cache_markdown(report: dict[str, Any]) -> str:
    n = report["n_queries"]
    configured = report["configured_cache_bytes"]

    def size(row: dict[str, Any]) -> str:
        megabytes = row["cache_bytes"] / MEBIBYTE
        return f"{megabytes:g} MB" if row["cache"].startswith("as configured") else row["cache"]

    lines = [
        "# The index cache at, above and below the size of the graph",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py cache` on "
        f"MariaDB {report['versions']['mariadb']}: {report['n_chunks']:,} chunks, index M = "
        f"{report['index_m']}, the {n} corpus claims, k = {report['k']}, `mhnsw_ef_search` "
        f"{report['ef_search']}. `mhnsw_max_cache_size` was {configured:,} bytes "
        f"({configured / MEBIBYTE:g} MB) when the run began and is {report['restored_cache_bytes']:,} "
        "bytes again after it. Each size is set with `SET GLOBAL` as root, and, except in the "
        "row that says otherwise, the table is reopened with `FLUSH TABLES chunk`. Then the "
        f"claims are searched in {report['passes']} passes with the bare statement and with the "
        "joined one, through `wikilense.search.search`, which sorts the returned rows by "
        "distance and chunk id. \"Equal to exact\" counts, per pass, the claims whose hit list "
        f"is NumPy's exact top {report['k']} in order. \"Two passes agree\" counts, for each pair of passes, "
        "the claims with identical lists. The times are of the whole call."),
        "",
        (f"| Cache | Statement | Equal to exact, of {n}, per pass | Article recall@{report['k']}, of {n} "
        f"| Overlap with exact | Short | Two passes agree, of {n} | p50 ms |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {size(row)}{' (as configured)' if row['cache'] == 'as configured' else ''}"
            f"{' (as configured, after the run)' if row['cache'] == 'as configured, again' else ''} "
            f"| {row['statement']} | {row['equal_to_exact']} | {_span(row['article_recall'])} | "
            f"{min(row['overlap_with_exact']):.3f} to {max(row['overlap_with_exact']):.3f} | "
            f"{_span(row['short'])} | {row['identical_between_two_passes']} | "
            f"{row['p50_ms']:.2f} |"
        )
    lines += [
        "",
        (f"NumPy's exact ranking finds a gold page for {report['exact_article_recall']} of {n} "
        "claims. The statements are those of strategy `none` and strategy `inline` in "
        "`wikilense/search.py`; the settings statements are:"),
        "",
        "```sql",
        *(cache_set_sql(size) + ";" for size in report.get("sizes_mb", CACHE_SIZES_MB)),
        CHUNK_FLUSH_SQL + ";",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------------------------


class Probe:
    """The scratch table and the two fixed data sets of the isolation experiment."""

    def __init__(self, settings: Settings) -> None:
        if settings.test_db_name == settings.db_name:
            raise SystemExit("the isolation experiment needs a test database other than the corpus")
        self.settings = settings
        rng = np.random.default_rng(ISOLATION_SEED)
        self.base = separate.unit_rows(rng.standard_normal((ISOLATION_ROWS, db.VECTOR_DIM))).astype(np.float32)
        self.query = separate.unit_rows(rng.standard_normal((1, db.VECTOR_DIM)))[0].astype(np.float32)
        # Rows closer to the query than any base row: the query plus a little noise.
        noise = 0.02 * rng.standard_normal((len(NEW_ROW_IDS) + 1, db.VECTOR_DIM))
        self.near = separate.unit_rows(self.query + noise).astype(np.float32)
        self.others = separate.unit_rows(
            rng.standard_normal((OTHER_QUERIES, db.VECTOR_DIM))
        ).astype(np.float32)

    def connect(self, level: str | None = None) -> Any:
        conn = db.connect(self.settings, database=self.settings.test_db_name)
        with conn.cursor() as cur:
            if level is not None:
                cur.execute(ISOLATION_LEVELS[level])
            cur.execute(PROBE_EF_SQL, (ISOLATION_EF,))
        conn.commit()
        return conn

    def fresh_table(self) -> None:
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(PROBE_DROP_SQL)
                cur.execute(PROBE_CREATE_SQL)
                cur.executemany(
                    PROBE_INSERT_SQL,
                    [(i + 1, db.vec_param(self.base[i])) for i in range(ISOLATION_ROWS)],
                )
            conn.commit()
        finally:
            conn.close()

    def drop_table(self) -> None:
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(PROBE_DROP_SQL)
            conn.commit()
        finally:
            conn.close()

    def search(self, conn: Any, sql: str = PROBE_KNN_SQL, query: np.ndarray | None = None) -> list[int]:
        with conn.cursor() as cur:
            cur.execute(sql, (db.vec_param(self.query if query is None else query), K))
            return [int(row[0]) for row in cur.fetchall()]

    def insert_near(self, conn: Any, ids: Sequence[int]) -> None:
        with conn.cursor() as cur:
            cur.executemany(
                PROBE_INSERT_SQL,
                [(row_id, db.vec_param(self.near[i])) for i, row_id in enumerate(ids)],
            )

    def insert_and_commit(self, ids: Sequence[int], first_vector: int = 0) -> None:
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    PROBE_INSERT_SQL,
                    [(row_id, db.vec_param(self.near[first_vector + i])) for i, row_id in enumerate(ids)],
                )
            conn.commit()
        finally:
            conn.close()

    def fresh_view(
        self, new_ids: Sequence[int] = NEW_ROW_IDS, level: str | None = None,
        autocommit: bool = False,
    ) -> dict[str, int]:
        """What a new connection sees: new rows among the top k through the index and by a scan.

        ``level`` sets the connection's isolation level (default: the server's); with
        ``autocommit`` every statement of the connection is its own transaction.
        """
        conn = self.connect(level)
        try:
            if autocommit:
                conn.autocommit(True)
            index_hits = self.search(conn)
            scan_hits = self.search(conn, PROBE_SCAN_SQL)
            return {
                "index_rows": len(index_hits),
                "index_new": sum(1 for row_id in index_hits if row_id in new_ids),
                "scan_rows": len(scan_hits),
                "scan_new": sum(1 for row_id in scan_hits if row_id in new_ids),
            }
        finally:
            conn.close()


def new_among(hits: Sequence[int]) -> int:
    return sum(1 for row_id in hits if row_id in NEW_ROW_IDS)


def levels_table(probe: Probe, trials: int) -> list[dict[str, Any]]:
    """Per isolation level: what reader A sees of writer B's five near rows, at three moments.

    Every level is run ``trials`` times on a fresh table; each cell is the list of the trials.
    """
    moments = ("writer_uncommitted", "writer_committed_reader_open", "reader_next_transaction")
    rows = []
    for level in ISOLATION_LEVELS:
        seen: dict[str, dict[str, list[int]]] = {m: {"index": [], "scan": []} for m in moments}
        blocked: list[str | None] = []
        for _ in range(trials):
            probe.fresh_table()
            reader, writer = probe.connect(level), probe.connect()
            try:
                with writer.cursor() as cur:
                    cur.execute(LOCK_WAIT_SQL)
                reader.begin()
                probe.search(reader)
                try:
                    probe.insert_near(writer, NEW_ROW_IDS)
                    blocked.append(None)
                except db.pymysql.err.OperationalError as exc:
                    blocked.append(f"{exc.args[1]} (error {exc.args[0]})")
                    writer.rollback()
                for moment in moments:
                    if moment == "writer_committed_reader_open":
                        writer.commit()
                    elif moment == "reader_next_transaction":
                        reader.commit()
                    seen[moment]["index"].append(new_among(probe.search(reader)))
                    seen[moment]["scan"].append(new_among(probe.search(reader, PROBE_SCAN_SQL)))
                reader.commit()
            finally:
                for conn in (reader, writer):
                    conn.rollback()
                    conn.close()
        rows.append({"level": level, "writer_blocked": blocked, **seen})
    return rows


def stale_sequence(
    probe: Probe, level: str, search_again: bool, close_reader: bool, autocommit: bool = False,
    first: str = "index",
) -> dict[str, int]:
    """Reader searches, writer commits five near rows, reader (maybe) searches again and ends.

    With ``autocommit`` the reader opens no transaction: each of its searches is its own.
    ``first`` is the reader's statement before the writer's commit: ``index`` (the search),
    ``scan`` (the same without the index) or ``count`` (``SELECT COUNT(*)``). Returns what a
    new connection then sees (:meth:`Probe.fresh_view`), plus ``reader_second_search_new``:
    the new rows the reader's second search returned (None when it did not search again).
    """
    probe.fresh_table()
    reader = probe.connect(level)
    try:
        if autocommit:
            reader.autocommit(True)
        else:
            reader.begin()
        if first == "index":
            probe.search(reader)
        elif first == "scan":
            probe.search(reader, PROBE_SCAN_SQL)
        else:
            with reader.cursor() as cur:
                cur.execute(PROBE_COUNT_SQL)
                cur.fetchall()
        probe.insert_and_commit(NEW_ROW_IDS)
        second = new_among(probe.search(reader)) if search_again else None
        if not autocommit:
            reader.commit()
        if close_reader:
            reader.close()
        return {**probe.fresh_view(), "reader_second_search_new": second}
    finally:
        if reader.open:
            reader.close()


def delete_sequence(probe: Probe, search_again: bool) -> dict[str, int]:
    """The stale sequence with a DELETE: the writer removes the three rows nearest to the query.

    Returns what the reader's snapshot and then a new connection see.
    """
    probe.fresh_table()
    reader = probe.connect("REPEATABLE READ")
    try:
        reader.begin()
        victims = probe.search(reader, PROBE_SCAN_SQL)[:3]
        probe.search(reader)
        writer = probe.connect()
        try:
            with writer.cursor() as cur:
                cur.execute(PROBE_DELETE3_SQL, tuple(victims))
            writer.commit()
        finally:
            writer.close()
        in_snapshot = probe.search(reader) if search_again else []
        reader.commit()
    finally:
        reader.close()
    conn = probe.connect()
    try:
        index_hits = probe.search(conn)
        scan_hits = probe.search(conn, PROBE_SCAN_SQL)
        other_rows = [len(probe.search(conn, query=other)) for other in probe.others]
        with conn.cursor() as cur:
            cur.execute(PROBE_COUNT_SQL)
            table_rows = int(cur.fetchone()[0])
    finally:
        conn.close()
    return {
        "reader_snapshot_rows": len(in_snapshot),
        "reader_snapshot_still_sees_deleted": sum(1 for v in victims if v in in_snapshot),
        "new_connection_index_rows": len(index_hits),
        "new_connection_scan_rows": len(scan_hits),
        "other_queries_min_rows": min(other_rows),
        "other_queries_with_no_row": sum(1 for count in other_rows if count == 0),
        "table_rows": table_rows,
    }  # fmt: skip


def index_rows(probe: Probe) -> int:
    """Rows a new connection gets for the query through the index."""
    conn = probe.connect()
    try:
        return len(probe.search(conn))
    finally:
        conn.close()


def flush_as_root(settings: Settings) -> str | None:
    """Run FLUSH TABLES on the probe table as root through docker; returns why not, or None."""
    try:
        import run_experiments
    except ImportError as exc:  # pragma: no cover - the script sits next to this one
        return f"scripts/run_experiments.py could not be imported: {exc}"
    try:
        run_experiments.Server(settings).root_sql(PROBE_FLUSH_SQL, database=settings.test_db_name)
    except run_experiments.ExperimentError as exc:
        return str(exc)
    return None


def measure_isolation(settings: Settings, trials: int, control_trials: int) -> dict[str, Any]:
    """Run the isolation experiment on a scratch table in the test database; return the report."""
    probe = Probe(settings)
    try:
        conn = probe.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(SERVER_STATE_SQL)
                version, default_level, snapshot_isolation = cur.fetchone()
        finally:
            conn.close()
        levels = levels_table(probe, control_trials)

        # The sequence that leaves the index stale, and three controls.
        sequences = {
            "repeatable_read_searches_again_and_ends": [
                stale_sequence(probe, "REPEATABLE READ", search_again=True, close_reader=True)
                for _ in range(trials)
            ],
            "control_read_committed": [
                stale_sequence(probe, "READ COMMITTED", search_again=True, close_reader=True)
                for _ in range(control_trials)
            ],
            "control_no_second_search": [
                stale_sequence(probe, "REPEATABLE READ", search_again=False, close_reader=True)
                for _ in range(control_trials)
            ],
            "control_autocommit": [
                stale_sequence(probe, "REPEATABLE READ", search_again=True, close_reader=True,
                               autocommit=True)
                for _ in range(control_trials)
            ],
            "control_reader_stays_connected": [
                stale_sequence(probe, "REPEATABLE READ", search_again=True, close_reader=False)
                for _ in range(control_trials)
            ],
            "control_first_statement_count": [
                stale_sequence(probe, "REPEATABLE READ", search_again=True, close_reader=True,
                               first="count")
                for _ in range(control_trials)
            ],
            "control_first_statement_scan": [
                stale_sequence(probe, "REPEATABLE READ", search_again=True, close_reader=True,
                               first="scan")
                for _ in range(control_trials)
            ],
        }

        # Who gets the stale answer: new connections at each level and in autocommit mode.
        victims: list[dict[str, int]] = []
        for _ in range(control_trials):
            stale_sequence(probe, "REPEATABLE READ", search_again=True, close_reader=True)
            victims.append({
                name: probe.fresh_view(
                    level=None if name == "autocommit" else name, autocommit=name == "autocommit"
                )["index_new"]
                for name in VICTIMS
            })

        # What clears the stale state: the next committed write, or FLUSH TABLES.
        recovery: dict[str, Any] = {"by_write": [], "by_flush": [], "flush_skipped": None,
                                    "still_stale_for_new_connections": []}
        attempts = 0
        while (len(recovery["by_write"]) < 3 or len(recovery["by_flush"]) < 3) and attempts < 20:
            attempts += 1
            view = stale_sequence(probe, "REPEATABLE READ", search_again=True, close_reader=True)
            if view["index_new"] != 0:
                continue  # this build did not go stale; try again
            later = [probe.fresh_view()["index_new"] for _ in range(3)]
            recovery["still_stale_for_new_connections"].append(later)
            if len(recovery["by_write"]) <= len(recovery["by_flush"]) or recovery["flush_skipped"]:
                probe.insert_and_commit([EXTRA_ROW_ID], first_vector=len(NEW_ROW_IDS))
                seen = probe.fresh_view([*NEW_ROW_IDS, EXTRA_ROW_ID])
                recovery["by_write"].append(seen["index_new"])
            else:
                reason = flush_as_root(settings)
                if reason is not None:
                    recovery["flush_skipped"] = reason
                    continue
                recovery["by_flush"].append(probe.fresh_view()["index_new"])
        recovery["attempts"] = attempts

        # Deletes: the writer removes the three nearest rows.
        deletes = [delete_sequence(probe, search_again=True) for _ in range(trials)]
        delete_flush = None
        if deletes and deletes[-1]["new_connection_index_rows"] == 0 and not recovery["flush_skipped"]:
            reason = flush_as_root(settings)
            if reason is None:
                conn = probe.connect()
                try:
                    delete_flush = len(probe.search(conn))
                finally:
                    conn.close()
        delete_by_write: list[int] = []
        for _ in range(DELETE_RECOVERY_TRIALS):
            if delete_sequence(probe, search_again=True)["new_connection_index_rows"] != 0:
                continue  # this table did not go stale
            probe.insert_and_commit([EXTRA_ROW_ID], first_vector=len(NEW_ROW_IDS))
            delete_by_write.append(index_rows(probe))
        delete_control = [
            delete_sequence(probe, search_again=False)["new_connection_index_rows"]
            for _ in range(control_trials)
        ]
        return {
            "name": "isolation", "created_at": utc_now(), "k": K,
            "rows": ISOLATION_ROWS, "new_rows": len(NEW_ROW_IDS), "ef_search": ISOLATION_EF,
            "index": "VECTOR INDEX (embedding) M=16 DISTANCE=cosine", "seed": ISOLATION_SEED,
            "trials": trials, "control_trials": control_trials,
            "mariadb_version": str(version), "default_isolation_level": str(default_level),
            "innodb_snapshot_isolation": int(snapshot_isolation),
            "levels": levels, "sequences": sequences, "victims": victims, "recovery": recovery,
            "deletes": deletes, "delete_rows_after_flush": delete_flush,
            "delete_rows_after_write": delete_by_write,
            "delete_control_no_second_search": delete_control,
        }  # fmt: skip
    finally:
        probe.drop_table()


def _span(values: Sequence[int]) -> str:
    """One number when every trial agrees, else "low to high"."""
    return str(values[0]) if len(set(values)) == 1 else f"{min(values)} to {max(values)}"


def _stale_count(views: Sequence[dict[str, int]]) -> str:
    """Trials in which the index returned none of the new rows although the scan returned some."""
    stale = sum(1 for view in views if view["index_new"] == 0 < view["scan_new"])
    return f"{stale} of {len(views)}"


def _partial_count(sequences: dict[str, Sequence[dict[str, int]]]) -> int:
    """Trials, over all sequences, in which the index found some but not all of the new rows."""
    return sum(
        1 for views in sequences.values() for view in views
        if 0 < view["index_new"] < view["scan_new"]
    )


def isolation_markdown(report: dict[str, Any]) -> str:
    new = report["new_rows"]
    lines = [
        "# The vector index, isolation levels, and a reader that leaves the index stale",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py isolation` on "
        f"MariaDB {report['mariadb_version']} (default isolation level "
        f"{report['default_isolation_level']}, innodb_snapshot_isolation "
        f"{report['innodb_snapshot_isolation']}). A scratch table in the test database: "
        f"{report['rows']:,} random unit vectors (seed {report['seed']}), `{report['index']}`, "
        f"`mhnsw_ef_search` {report['ef_search']}, k = {report['k']}. A writer adds {new} rows that "
        "are nearer to the query than every other row, so a search that sees them returns all "
        f"{new} among its top {report['k']}. \"Index\" is the ordinary statement; \"scan\" is the "
        "same statement with `IGNORE INDEX (embedding)`, which reads the table."),
        "",
        "## What a reader sees of another connection's new rows",
        "",
        (f"Reader A opens a transaction at the given level and searches. Writer B inserts the {new} "
        "rows. Each cell: new rows A finds through the index / by a scan, over "
        f"{report['control_trials']} trials on fresh tables (one number when every trial agreed)."),
        "",
        "| A's isolation level | B has not committed | B has committed, A's transaction still open | A's next transaction | B's insert |",
        "|---|---|---|---|---|",
    ]
    for row in report["levels"]:
        cells = [
            f"{_span(row[key]['index'])} / {_span(row[key]['scan'])}"
            for key in ("writer_uncommitted", "writer_committed_reader_open", "reader_next_transaction")
        ]
        reasons = sorted({reason for reason in row["writer_blocked"] if reason})
        lines.append(
            f"| {row['level']} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{'blocked: ' + '; '.join(reasons) if reasons else 'went through'} |"
        )
    seq = report["sequences"]
    recovery = report["recovery"]
    lines += [
        "",
        "## The sequence after which the index is stale",
        "",
        "1. A, at REPEATABLE READ, opens a transaction and searches the index.",
        f"2. B inserts the {new} rows and commits.",
        ("3. A searches again, still inside its old snapshot (it found "
        f"{_span([v['reader_second_search_new'] for v in seq['repeatable_read_searches_again_and_ends']])} "
        "of the new rows, as its snapshot requires)."),
        "4. A commits and disconnects.",
        "5. A new connection searches.",
        "",
        ("At step 5 the table holds the new rows and a scan returns them. \"Stale\" below means the "
        "index search of the new connection returned none of them."),
        "",
        "| Sequence | Stale at step 5 | New rows found through the index, per trial |",
        "|---|---|---|",
        (f"| as above | {_stale_count(seq['repeatable_read_searches_again_and_ends'])} | "
        f"{[v['index_new'] for v in seq['repeatable_read_searches_again_and_ends']]} |"),
        (f"| control: A is at READ COMMITTED | {_stale_count(seq['control_read_committed'])} | "
        f"{[v['index_new'] for v in seq['control_read_committed']]} |"),
        (f"| control: A skips step 3 | {_stale_count(seq['control_no_second_search'])} | "
        f"{[v['index_new'] for v in seq['control_no_second_search']]} |"),
        (f"| control: A is in autocommit mode, so step 3 is a new transaction | "
        f"{_stale_count(seq['control_autocommit'])} | "
        f"{[v['index_new'] for v in seq['control_autocommit']]} |"),
        (f"| control: A commits but stays connected | {_stale_count(seq['control_reader_stays_connected'])} | "
        f"{[v['index_new'] for v in seq['control_reader_stays_connected']]} |"),
        ("| control: A's statement in step 1 is `SELECT COUNT(*)`, not a search | "
        f"{_stale_count(seq['control_first_statement_count'])} | "
        f"{[v['index_new'] for v in seq['control_first_statement_count']]} |"),
        ("| control: A's statement in step 1 is the scan, not the index search | "
        f"{_stale_count(seq['control_first_statement_scan'])} | "
        f"{[v['index_new'] for v in seq['control_first_statement_scan']]} |"),
        "",
        ("In every trial that went stale, a REPEATABLE READ transaction had searched the index "
        "before the other connection's commit and again after it; no control without one of "
        "the two went stale."
        + (f" In {_partial_count(seq)} trial(s) the index found some but not all of the new rows: "
           "that is the approximate search, not this state."
           if _partial_count(seq) else "")),
        "",
        ("Who gets the stale answer: after the sequence, new connections found these numbers of "
        f"the {new} new rows through the index, on {len(report['victims'])} stale tables: "
        + "; ".join(
            f"{name if name != 'autocommit' else 'in autocommit mode'} "
            f"{_span([v[name] for v in report['victims']])}"
            for name in VICTIMS
        )
        + ". The isolation level of the connection that asks does not protect it."),
        "",
        ("On each stale table, three more new connections then found these numbers of new rows "
        f"through the index: {recovery['still_stale_for_new_connections']}. What cleared it:"),
        "",
        (f"- the next committed write to the table: after one more row was inserted and committed, a "
        f"new connection found {recovery['by_write']} of {new + 1} new rows through the index;"),
    ]
    if recovery["flush_skipped"]:
        lines.append(f"- `FLUSH TABLES` was not tried: {recovery['flush_skipped']}.")
    else:
        lines.append(
            f"- `FLUSH TABLES isolation_probe` (as root): afterwards a new connection found "
            f"{recovery['by_flush']} of {new} new rows through the index."
        )
    deletes = report["deletes"]
    if recovery["by_flush"]:
        lines += [
            "",
            ("`FLUSH TABLES` makes the server open the table again, so the index as stored was "
            "right. That fits a copy of the graph in the server's memory that was loaded inside "
            "A's old snapshot. We did not inspect that cache or read the server source."),
        ]
    lines += [
        "",
        "## The same sequence with a DELETE",
        "",
        ("B deletes the three rows nearest to the query instead of inserting. In its second "
        "search A still found "
        f"{_span([trial['reader_snapshot_still_sees_deleted'] for trial in deletes])} of the 3 "
        "deleted rows, as its snapshot requires. Afterwards, for a new connection:"),
        "",
        "| Trial | Rows the index returns for that query | Rows a scan returns | Of 20 other queries, how many got no row (fewest rows any got) | Rows in the table |",
        "|---|---|---|---|---|",
    ]
    for i, trial in enumerate(deletes, start=1):
        lines.append(
            f"| {i} | {trial['new_connection_index_rows']} | {trial['new_connection_scan_rows']} | "
            f"{trial['other_queries_with_no_row']} ({trial['other_queries_min_rows']}) | "
            f"{trial['table_rows']:,} |"
        )
    if report["delete_rows_after_flush"] is not None:
        lines += ["", (f"After `FLUSH TABLES` the index returned {report['delete_rows_after_flush']} "
                      "rows for that query again.")]
    lines += [
        "",
        (f"After one more row was inserted and committed instead, the index returned "
        f"{report['delete_rows_after_write']} rows for that query "
        f"({len(report['delete_rows_after_write'])} stale tables). Control, A does not search "
        "again after B's commit: the index returned "
        f"{report['delete_control_no_second_search']} rows."),
        "",
        "## What follows for an application",
        "",
        ("- On this table, a reader did not cause the state when it searched at READ COMMITTED or "
        "in autocommit mode, when it ended its transaction before searching again, or when its "
        "transaction searched the index only after the other connection's commit. The last case "
        "is not safe in general: in a plain-SQL repro with 8-dimensional vectors on 11.8.9 and "
        "13.0.2, a REPEATABLE READ transaction whose first statement was `SELECT COUNT(*)`, a "
        "scan or `START TRANSACTION WITH CONSISTENT SNAPSHOT`, and that searched the index once "
        "after the other connection's commit, left the index stale in 3 of 3 trials. Once the "
        "state is there, connections at every level get the stale answer. So an application "
        "that writes while it searches is safe only if none of its connections searches the "
        "index inside a REPEATABLE READ transaction whose snapshot predates another "
        "connection's commit."),
        ("- PyMySQL opens connections with autocommit off, so a connection that has run one "
        "`SELECT` is inside a REPEATABLE READ transaction until it commits, rolls back or "
        "closes."),
        ("- WikiLense's measurements are not affected: no connection commits a write to `chunk` "
        "while a search runs, and `wikilense ingest` recreates the tables."),
        ("- The behaviour is MariaDB issue "
        "[MDEV-41321](https://jira.mariadb.org/browse/MDEV-41321). The tables above were measured "
        "on one server version, on one scratch table of "
        f"{report['rows']:,} random vectors and one query, rebuilt for every trial."),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# hard cap
# ---------------------------------------------------------------------------------------------


def exact_ranking(
    distances: np.ndarray, ids: np.ndarray, count: int
) -> tuple[list[int], list[float]]:
    """The ids and distances of the ``count`` rows nearest by ``distances``, nearest first.

    The same order as ``compare_separate_index.top_k``: ties are broken by the smaller id, and
    fewer rows than ``count`` returns all of them.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if ids.shape[0] > count:
        # Keep everything tied with the count-th distance, so the tie-break below is exact.
        kth = np.partition(distances, count - 1)[count - 1]
        keep = np.flatnonzero(distances <= kth)
    else:
        keep = np.arange(ids.shape[0])
    chosen = keep[np.lexsort((ids[keep], distances[keep]))[:count]]
    return [int(value) for value in ids[chosen]], [float(value) for value in distances[chosen]]


def distance_threshold(
    nearest: Sequence[float], n: int, margin: float = TIE_MARGIN
) -> float | None:
    """A distance that exactly ``n`` chunks are nearer than, from ``nearest`` (ascending).

    Halfway between the n-th and the (n+1)-th distance. None when there is no (n+1)-th one, or
    when the two are at most ``margin`` apart: a tie leaves no threshold with exactly ``n``
    below it, and a smaller gap than the server's rounding could put a chunk on the other side.
    """
    if n < 1 or len(nearest) <= n:
        return None
    below, above = float(nearest[n - 1]), float(nearest[n])
    if above - below <= margin:
        return None
    return (below + above) / 2.0


def word_limit_candidates(words: np.ndarray) -> list[tuple[int, int]]:
    """Every limit that changes which chunks pass ``n_words < limit``, with how many pass.

    For each distinct ``n_words`` value ``v`` the limit ``v + 1``, which passes the chunks with
    at most ``v`` words; in ascending order of the limit.
    """
    values, counts = np.unique(np.asarray(words, dtype=np.int64), return_counts=True)
    return [(int(value) + 1, int(passing))
            for value, passing in zip(values, np.cumsum(counts), strict=True)]


def pick_word_limits(
    words: np.ndarray, shares: Sequence[float] = HARD_CAP_SHARES
) -> list[dict[str, Any]]:
    """The word limit for each share of the chunks, by the rule of ``scripts/scaled_filters.py``.

    The limit whose passing chunks are nearest ``share`` times the chunk count on a log scale;
    ties go to the larger count, then to the smaller limit. A limit that two shares pick is
    kept once, under the first of them.
    """
    total = int(np.asarray(words).shape[0])
    candidates = word_limit_candidates(words)
    picked: list[dict[str, Any]] = []
    for share in shares:
        limit, passing = scaled_filters.nearest(candidates, share * total)
        if any(entry["words_below"] == limit for entry in picked):
            continue
        picked.append({"target_share": share, "words_below": limit, "passing_chunks": passing,
                       "passing_share": round(passing / total, 6)})
    return picked


def chunk_access_of(rows: Sequence[dict[str, Any]]) -> str:
    """'<type> <key>' of the ``chunk`` row of one EXPLAIN result."""
    row = next((r for r in rows if r["table"] == "chunk"), None)
    return "no chunk row" if row is None else f"{row['type']} {row['key'] or 'no key'}"


class HardCapStop(Exception):
    """The hard_cap part cannot go on: the server was lost too often in a row, or it did not
    accept a connection again within the wait."""


def error_outcome(error: BaseException) -> str | None:
    """'stopped' or 'server lost' for the error of one statement; None for any other error.

    'stopped' is MariaDB's error for a statement that ran past ``max_statement_time``
    (:data:`STATEMENT_TIMEOUT_ERROR`). 'server lost' is a client error of a connection whose
    server went away (:data:`SERVER_LOST_ERRORS`), or PyMySQL's ``InterfaceError(0, '')`` for a
    connection that is closed. The code is the exception's ``args[0]``.
    """
    if not isinstance(error, pymysql.err.Error) or not error.args:
        return None
    code = error.args[0]
    if code == STATEMENT_TIMEOUT_ERROR:
        return STOPPED
    if isinstance(error, pymysql.err.InterfaceError) and code == 0:
        return SERVER_LOST
    if (isinstance(error, (pymysql.err.OperationalError, pymysql.err.InterfaceError))
            and code in SERVER_LOST_ERRORS):
        return SERVER_LOST
    return None


def rows_outcome(rows: int, expected: int) -> str:
    """'complete' for a statement that returned ``expected`` rows, 'short' for one with fewer."""
    return SHORT if rows < expected else COMPLETE


def _drop(conn: Any) -> None:
    """Close ``conn`` if it is still open; an error while closing it is of no use here."""
    if getattr(conn, "open", False):
        with suppress(pymysql.err.Error, OSError):
            conn.close()


class GuardedConnection:
    """The hard_cap part's connection: a time limit on every statement, and a new connection
    after the server is lost.

    :meth:`start` reads the session's ``max_statement_time`` and ``mhnsw_ef_search`` and sets
    them to :data:`HARD_CAP_STATEMENT_SECONDS` and :data:`HARD_CAP_EF`; :meth:`close` writes the
    values it read back while the connection is open, then closes it. :meth:`attempt` runs one
    statement and turns a stop at the time limit or a lost server into an outcome. After a lost
    server it closes the connection, tries a new one at once and then every
    :data:`HARD_CAP_RECONNECT_INTERVAL` seconds for up to :data:`HARD_CAP_RECONNECT_SECONDS`, and
    sets the new session up the same way. ``connect``, ``sleep`` and ``clock`` default to
    ``db.connect``, ``time.sleep`` and ``time.monotonic``.
    """

    def __init__(
        self,
        settings: Settings,
        connect: Callable[[Settings], Any] | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._connect = db.connect if connect is None else connect
        self._sleep = time.sleep if sleep is None else sleep
        self._clock = time.monotonic if clock is None else clock
        self.conn: Any = self._connect(settings)
        self.previous: tuple[float, int] | None = None
        self.lost_in_a_row = 0
        self.losses = 0
        self.reconnects = 0

    @staticmethod
    def _set_up(conn: Any) -> None:
        with conn.cursor() as cur:
            cur.execute(HARD_CAP_TIME_SET_SQL, (HARD_CAP_STATEMENT_SECONDS,))
        db.set_session_var(conn, searchmod.EF_SEARCH_VARIABLE, HARD_CAP_EF)

    def start(self) -> None:
        """Read the session's two values, then set the time limit and ``mhnsw_ef_search``."""
        with self.conn.cursor() as cur:
            cur.execute(HARD_CAP_TIME_GET_SQL)
            limit = float(cur.fetchone()[0])
        self.previous = (limit, db.get_session_var(self.conn, searchmod.EF_SEARCH_VARIABLE))
        self._set_up(self.conn)

    def close(self) -> None:
        """Write back the values :meth:`start` read, while the connection is open; close it."""
        conn = self.conn
        if getattr(conn, "open", False) and self.previous is not None:
            limit, ef = self.previous
            with suppress(pymysql.err.Error, OSError):
                with conn.cursor() as cur:
                    cur.execute(HARD_CAP_TIME_SET_SQL, (limit,))
                db.set_session_var(conn, searchmod.EF_SEARCH_VARIABLE, ef)
        _drop(conn)

    def _reconnect(self) -> Any:
        deadline = self._clock() + HARD_CAP_RECONNECT_SECONDS
        while True:
            conn = None
            try:
                conn = self._connect(self._settings)
                self._set_up(conn)
                return conn
            except (pymysql.err.Error, OSError):
                if conn is not None:
                    _drop(conn)
            if self._clock() >= deadline:
                raise HardCapStop("the server did not accept a connection within "
                                  f"{HARD_CAP_RECONNECT_SECONDS:g} s")
            self._sleep(HARD_CAP_RECONNECT_INTERVAL)

    def attempt(
        self, work: Callable[..., Any], *args: Any, executes: bool = True
    ) -> tuple[str | None, Any]:
        """Run ``work(connection, *args)``; return ``(None, its value)``, ``('stopped', None)``
        for a statement stopped at the time limit, or ``('server lost', None)`` once a new
        connection is set up.

        A statement that ``executes`` and comes back, with rows or stopped, ends a run of lost
        servers; ``EXPLAIN`` (``executes=False``) does not, but a server lost during it counts.
        Raises :class:`HardCapStop` at the :data:`HARD_CAP_LOST_IN_A_ROW`-th lost server in a
        row and when no connection is accepted within the wait. Any other error is raised as it
        is.
        """
        try:
            value = work(self.conn, *args)
        except pymysql.err.Error as exc:
            outcome = error_outcome(exc)
            if outcome is None:
                raise
        else:
            if executes:
                self.lost_in_a_row = 0
            return None, value
        if outcome == STOPPED:
            if executes:
                self.lost_in_a_row = 0
            return outcome, None
        self.losses += 1
        self.lost_in_a_row += 1
        _drop(self.conn)
        if self.lost_in_a_row >= HARD_CAP_LOST_IN_A_ROW:
            raise HardCapStop(f"the server was lost {self.lost_in_a_row} times in a row")
        self.conn = self._reconnect()
        self.reconnects += 1
        return outcome, None


def _fetch(conn: Any, sql: str, params: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Every row of one statement."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _rows_read_of(conn: Any, sql: str, params: Sequence[Any]) -> float:
    """The ``chunk`` rows one statement reads, from ``ANALYZE FORMAT=JSON`` (which runs it)."""
    with conn.cursor() as cur:
        cur.execute("ANALYZE FORMAT=JSON " + sql, params)
        # r_rows is an average per loop: the product is a whole number
        return float(round(chunk_rows_read(json.loads(cur.fetchone()[0]))))


def run_claims(
    guard: Any,
    sql: str,
    params: Sequence[tuple[Any, ...] | None],
    expected: int,
    repeats: int,
    records: list[dict[str, Any]],
) -> None:
    """Run one case's statement for every claim with parameters (None leaves a claim out).

    Per claim, through ``guard.attempt``: one untimed execution and ``repeats`` timed ones, then
    ``EXPLAIN``, then ``ANALYZE FORMAT=JSON`` unless an execution of the claim was stopped or
    lost the server. A lost server ends the claim's executions and the next claim follows. Only
    the timed executions that came back complete or short are timed. Each claim's record is
    appended to ``records`` before its statements run, so the claims run so far are kept when
    :class:`HardCapStop` is raised.
    """
    for claim, claim_params in enumerate(params):
        if claim_params is None:
            continue
        record: dict[str, Any] = {"claim": claim, "outcomes": [], "ms": [], "rows": None,
                                  "plan": None, "access": None, "analyze": None,
                                  "rows_read": None}  # fmt: skip
        records.append(record)
        for pass_index in range(-1, repeats):  # -1 is the untimed execution
            start = time.perf_counter()
            outcome, found = guard.attempt(_fetch, sql, claim_params)
            elapsed = (time.perf_counter() - start) * 1000.0
            if outcome is None:
                got = [int(row[0]) for row in found]
                outcome = rows_outcome(len(got), expected)
                if record["rows"] is None:
                    record["rows"] = got
                if pass_index >= 0:
                    record["ms"].append(elapsed)
            record["outcomes"].append(outcome)
            if outcome == SERVER_LOST:
                break
        outcome, plan = guard.attempt(db.explain, sql, claim_params, executes=False)
        if outcome is None:
            record["plan"] = [{key: row[key] for key in ("table", "type", "key", "Extra")}
                              for row in plan]
            record["access"] = chunk_access_of(plan)
        if STOPPED in record["outcomes"] or SERVER_LOST in record["outcomes"]:
            record["analyze"] = "skipped"
            continue
        outcome, read = guard.attempt(_rows_read_of, sql, claim_params)
        record["analyze"] = outcome or COMPLETE
        record["rows_read"] = read


def summarise_hard_cap_case(
    case: dict[str, Any],
    records: Sequence[dict[str, Any]],
    exact: Callable[[int], list[int]],
    repeats: int,
) -> dict[str, Any]:
    """The report row of one case, from the records of :func:`run_claims`.

    The outcome counts are over executions, ``case['claims']`` times 1 + ``repeats`` of them;
    those not run (the rest of a claim after a lost server, the claims after a stop of the part)
    are ``not_run``. The agreement with ``exact(claim)`` is over the claims whose statement
    returned rows, from the first execution that did.
    """
    outcomes = Counter(outcome for record in records for outcome in record["outcomes"])
    planned = case["claims"] * (1 + repeats)
    answered = [record for record in records if record["rows"] is not None]
    pairs = [(record["rows"], exact(record["claim"])) for record in answered]
    times = [ms for record in records for ms in record["ms"]]
    read = [record["rows_read"] for record in records if record["rows_read"] is not None]
    plans = Counter(record["access"] for record in records if record["access"] is not None)
    analyze = Counter(record["analyze"] for record in records if record["analyze"] is not None)
    stats = evaluate.latency_stats(times) if times else None
    access, access_count = plans.most_common(1)[0] if plans else ("not run", 0)
    returned = Counter(len(got) for got, _ in pairs)
    return {
        "case": case["key"], "kind": case["kind"], "label": case["label"], "n": case["n"],
        "inner_limit": case["inner_limit"], "words_below": case["words_below"],
        "joined": case["joined"], "passing_chunks": case.get("passing_chunks"),
        "claims": case["claims"], "claims_run": len(records), "expected_rows": case["expected"],
        "executions": planned,
        "complete": outcomes[COMPLETE], "short": outcomes[SHORT],
        "stopped": outcomes[STOPPED], "server_lost": outcomes[SERVER_LOST],
        "not_run": planned - sum(outcomes.values()),
        "answered": len(answered),
        "rows_returned": {str(count): returned[count] for count in sorted(returned, reverse=True)},
        "equal_to_exact": sum(1 for got, want in pairs if got == want),
        "same_set_as_exact": sum(1 for got, want in pairs if set(got) == set(want)),
        "analyze": {key: analyze[key] for key in (COMPLETE, "skipped", STOPPED, SERVER_LOST)},
        "chunk_rows_read_p50": median(read) if read else None,
        "chunk_rows_read_max": max(read) if read else None,
        "chunk_access": access, "chunk_access_count": access_count, "plans": dict(plans),
        "timed_executions": len(times),
        "p50_ms": None if stats is None else round(stats.p50_ms, 3),
        "p95_ms": None if stats is None else round(stats.p95_ms, 3),
    }  # fmt: skip


def measure_hard_cap(
    settings: Settings,
    embedder: Any,
    repeats: int,
    k: int = K,
    sample: int | None = None,
    seed: int = SAMPLE_SEED,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the threshold, subquery and predicate statements on the corpus; return the report.

    ``sample`` takes that many claims, drawn with ``random.Random(seed)`` and kept in claim
    order, instead of all of them. Once the corpus is read, every statement runs through a
    :class:`GuardedConnection`, under its time limit, and each execution is recorded with its
    outcome. When the part stops (:class:`HardCapStop`), the report keeps what was measured and
    says why. ``checkpoint``, when given, receives the report so far after the threshold check
    and after each case. Reads the corpus only.
    """
    guard = GuardedConnection(settings)
    try:
        conn = guard.conn
        truths = evaluate.load_ground_truth(conn)
        if not truths:
            raise SystemExit("the claim table is empty: run 'wikilense ingest' first")
        n_claims_total = len(truths)
        truths = sample_claims(truths, sample, seed)
        with conn.cursor() as cur:
            cur.execute(CHUNK_EXPORT_SQL)
            rows = cur.fetchall()
        ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        words = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
        vectors = separate.unit_rows(
            np.vstack([db.vec_from_bytes(row[2]) for row in rows]).astype(np.float64)
        )
        del rows
        queries = separate.unit_rows(
            np.asarray(embedder.embed_queries([t.text for t in truths]), dtype=np.float32)
        ).astype(np.float32)
        separate._quiet_torch()
        n_queries = len(truths)
        vector_params = [db.vec_param(query) for query in queries]
        word_limits = pick_word_limits(words)
        passing_rows = {entry["words_below"]: np.flatnonzero(words < entry["words_below"])
                        for entry in word_limits}
        report: dict[str, Any] = {
            "name": "hard_cap", "created_at": utc_now(), "k": k, "ef_search": HARD_CAP_EF,
            "time_limit_s": HARD_CAP_STATEMENT_SECONDS,
            "server": {"version": conn.get_server_info(), "port": conn.port},
            "repeats": repeats, "n_claims_total": n_claims_total, "sample": sample,
            "seed": seed if sample else None, "n_queries": n_queries, "n_chunks": int(ids.shape[0]),
            **db.vector_index_info(conn),
            "threshold_ns": list(HARD_CAP_NS), "inner_limits": list(HARD_CAP_INNER_LIMITS),
            "tie_margin": TIE_MARGIN, "target_shares": list(HARD_CAP_SHARES),
            "word_limits": word_limits,
            "finished": False, "stopped_early": None, "cases_total": None, "cases_done": 0,
            "server_losses": 0, "reconnects": 0,
            "threshold_check": None, "claims": [], "results": [], "explain_first_claim": {},
            "ingest_meta": evaluate.read_ingest_meta(conn),
            "machine": evaluate.machine_info(embedder), "versions": evaluate.versions_info(conn),
        }  # fmt: skip
        print(f"hard_cap: MariaDB {report['server']['version']} at port "
              f"{report['server']['port']}, a time limit of {HARD_CAP_STATEMENT_SECONDS} s per "
              "statement", flush=True)
        guard.start()

        # NumPy's exact ranking of every chunk per claim: the thresholds and the exact lists.
        depth = max(HARD_CAP_NS) + 1
        rankings: list[list[int]] = []
        thresholds: list[dict[int, float | None]] = []
        predicate_exact: dict[int, list[list[int]]] = {limit: [] for limit in passing_rows}
        for i in range(n_queries):
            distances = 1.0 - vectors @ queries[i].astype(np.float64)
            ranked_ids, ranked = exact_ranking(distances, ids, depth)
            rankings.append(ranked_ids)
            thresholds.append({n: distance_threshold(ranked, n) for n in HARD_CAP_NS})
            for limit, selected in passing_rows.items():
                predicate_exact[limit].append(
                    exact_ranking(distances[selected], ids[selected], k)[0]
                )

        cases: list[dict[str, Any]] = []
        for n in HARD_CAP_NS:
            cases.append({"key": f"threshold_{n}", "kind": "threshold", "sql": HARD_CAP_THRESHOLD_SQL,
                          "n": n, "inner_limit": None, "words_below": None, "joined": False,
                          "label": f"WHERE distance < threshold, n = {n}",
                          "expected": min(k, n)})  # fmt: skip
        for inner in HARD_CAP_INNER_LIMITS:
            for n in HARD_CAP_NS:
                cases.append({"key": f"subquery_{inner}_{n}", "kind": "subquery",
                              "sql": HARD_CAP_SUBQUERY_SQL, "n": n, "inner_limit": inner,
                              "words_below": None, "joined": False,
                              "label": f"inner LIMIT {inner}, n = {n}",
                              "expected": min(inner, n, k)})  # fmt: skip
        for entry in word_limits:
            limit = entry["words_below"]
            for joined, sql in ((False, FORM_PREDICATE), (True, FORM_PREDICATE_JOINED)):
                cases.append({"key": f"predicate_{limit}{'_joined' if joined else ''}",
                              "kind": "predicate", "sql": sql, "n": None, "inner_limit": None,
                              "words_below": limit, "joined": joined,
                              "label": f"WHERE chunk.n_words < {limit}"
                                       f"{', joined to page' if joined else ''}",
                              "passing_chunks": entry["passing_chunks"],
                              "expected": min(k, entry["passing_chunks"])})  # fmt: skip

        def params_of(case: dict[str, Any], i: int) -> tuple[Any, ...] | None:
            """The parameters of ``case`` for claim ``i``; None when the claim has no threshold."""
            vector = vector_params[i]
            if case["kind"] == "predicate":
                return (case["words_below"], vector, k)
            threshold = thresholds[i][case["n"]]
            if threshold is None:
                return None
            if case["kind"] == "threshold":
                return (vector, vector, threshold, vector, k)
            return (vector, vector, case["inner_limit"], threshold, k)

        def exact_of(case: dict[str, Any], i: int) -> list[int]:
            if case["kind"] == "predicate":
                return predicate_exact[case["words_below"]][i]
            return rankings[i][: case["expected"]]

        for case in cases:
            case["claims"] = sum(1 for i in range(n_queries) if params_of(case, i) is not None)
        report["cases_total"] = len(cases)
        results: list[dict[str, Any]] = report["results"]

        def save() -> None:
            report.update(cases_done=len(results), server_losses=guard.losses,
                          reconnects=guard.reconnects)
            if checkpoint is not None:
                checkpoint(report)

        # The server's own count of the chunks under each threshold: a scan without the index.
        scans: list[str] = [NOT_RUN] * n_queries
        server_counts: list[dict[int, int] | None] = [None] * n_queries
        distance_gap = 0.0
        try:
            try:
                for i in range(n_queries):
                    widest = max((t for t in thresholds[i].values() if t is not None),
                                 default=None)
                    if widest is None:
                        scans[i] = NO_THRESHOLD
                        continue
                    outcome, found = guard.attempt(
                        _fetch, HARD_CAP_SCAN_SQL, (vector_params[i], vector_params[i], widest)
                    )
                    scans[i] = outcome or COMPLETE
                    if outcome is not None:
                        continue
                    server = np.asarray([float(row[1]) for row in found], dtype=np.float64)
                    found_ids = np.asarray([int(row[0]) for row in found], dtype=np.int64)
                    if found_ids.size:
                        query = queries[i].astype(np.float64)
                        # ids are in ascending order
                        mine = 1.0 - vectors[np.searchsorted(ids, found_ids)] @ query
                        distance_gap = max(distance_gap, float(np.max(np.abs(server - mine))))
                    server_counts[i] = {n: int(np.sum(server < threshold))
                                        for n, threshold in thresholds[i].items()
                                        if threshold is not None}
            finally:
                per_n = []
                for n in HARD_CAP_NS:
                    with_threshold = [i for i in range(n_queries) if thresholds[i][n] is not None]
                    counted = [server_counts[i][n] for i in with_threshold
                               if server_counts[i] is not None]
                    per_n.append({
                        "n": n, "claims_with_threshold": len(with_threshold),
                        "left_out_ties": n_queries - len(with_threshold),
                        "scan_finished": len(counted),
                        "server_count_equal_n": sum(1 for c in counted if c == n),
                        "server_counts_other": sorted({c for c in counted if c != n}),
                    })  # fmt: skip
                tally = Counter(scans)
                report["threshold_check"] = {
                    "per_n": per_n,
                    "scan": {key: tally[key]
                             for key in (COMPLETE, STOPPED, SERVER_LOST, NOT_RUN, NO_THRESHOLD)},
                    "server_numpy_distance_gap_max": distance_gap,
                }
                report["claims"] = [
                    {"claim_id": truth.claim_id,
                     "thresholds": {str(n): own[n] for n in HARD_CAP_NS}, "scan": scan}
                    for truth, own, scan in zip(truths, thresholds, scans, strict=True)
                ]
                save()

            for case in cases:
                records: list[dict[str, Any]] = []
                try:
                    run_claims(guard, case["sql"], [params_of(case, i) for i in range(n_queries)],
                               case["expected"], repeats, records)
                finally:
                    results.append(
                        summarise_hard_cap_case(case, records, partial(exact_of, case), repeats)
                    )
                    first = next((r["plan"] for r in records if r["plan"] is not None), None)
                    if first is not None:
                        report["explain_first_claim"][case["key"]] = first
                    save()
        except HardCapStop as stop:
            report["stopped_early"] = str(stop)
        report["finished"] = report["stopped_early"] is None and len(results) == len(cases)
        report.update(cases_done=len(results), server_losses=guard.losses,
                      reconnects=guard.reconnects)
        return report
    finally:
        guard.close()


def _rows_returned(row: dict[str, Any]) -> str:
    """'10 (195), 9 (5)': each number of rows returned, with the claims in brackets."""
    return ", ".join(f"{count} ({claims})" for count, claims in row["rows_returned"].items()) or "-"


def _rows_read(row: dict[str, Any]) -> str:
    if row["chunk_rows_read_p50"] is None:
        return "-"
    return f"{row['chunk_rows_read_p50']:.0f} ({row['chunk_rows_read_max']:.0f})"


def _access(row: dict[str, Any]) -> str:
    if row["chunk_access_count"] == row["claims"]:
        return row["chunk_access"]
    return f"{row['chunk_access']} ({row['chunk_access_count']}/{row['claims']})"


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _plural(count: int, noun: str) -> str:
    """'1 claim', '2 claims'."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _executed(row: dict[str, Any]) -> int:
    """The executions of a case that ran: complete, short, stopped or with the server lost."""
    return row["complete"] + row["short"] + row["stopped"] + row["server_lost"]


def hard_cap_markdown(report: dict[str, Any]) -> str:
    n = report["n_queries"]
    k = report["k"]
    rows = report["results"]
    threshold = [row for row in rows if row["kind"] == "threshold"]
    subquery = [row for row in rows if row["kind"] == "subquery"]
    predicate = [row for row in rows if row["kind"] == "predicate"]
    if report.get("sample"):
        claims = (f"a random sample of {n} of the {report['n_claims_total']:,} corpus claims as "
                  f"queries (`random.Random({report['seed']})`)")
    else:
        claims = f"the {n} corpus claims as queries"
    passes = report["repeats"]
    limit = report["time_limit_s"]
    server = report["server"]
    check = report["threshold_check"]

    def outcome(row: dict[str, Any]) -> str:
        return (f"| {row['claims']} | {row['expected_rows']} | {row['complete']} | {row['short']} "
                f"| {row['stopped']} | {row['server_lost']} | {_rows_returned(row)} | "
                f"{row['equal_to_exact']}/{row['answered']} | "
                f"{row['same_set_as_exact']}/{row['answered']} | {_rows_read(row)} | "
                f"{_access(row)} | {row['timed_executions']} | {_ms(row['p50_ms'])} | "
                f"{_ms(row['p95_ms'])} |")

    columns = ("| Claims | Expected | Complete | Short | Stopped | Server lost | Rows returned "
               "| Equal to exact | Same set | Rows read (largest) | Chunk access | Timed "
               "| p50 ms | p95 ms |")
    rule = "|---" * 14 + "|"

    def counts(
        selected: list[dict[str, Any]], key: str, name: Callable[[dict[str, Any]], str],
        of: Callable[[dict[str, Any]], int],
    ) -> str:
        """'a/x at n = 3, b/y at n = 10' for the rows given, x and y from ``of``."""
        return ", ".join(f"{row[key]}/{of(row)} at {name(row)}" for row in selected)

    def answered(row: dict[str, Any]) -> int:
        return row["answered"]

    lines = [
        "# The LIMIT of an index search, under a distance threshold and a selective predicate",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py hard_cap` on "
        f"MariaDB {server['version']} at port {server['port']}: {report['n_chunks']:,} chunks, "
        f"index M = {report['index_m']}, {claims}, k = {k}, `mhnsw_ef_search` "
        f"{report['ef_search']}, a time limit of {limit:g} s on every statement "
        f"(`max_statement_time`). The cases ran one after the other; for each claim the "
        f"statement ran once untimed and then {passes} {'time' if passes == 1 else 'times'} "
        "timed. MariaDB's documentation (\"Vector Overview\", \"Using the Vector Index "
        "Efficiently\") says: \"The LIMIT is a hard cap on how many rows the index returns before "
        "the threshold is applied.\" A claim's threshold lies halfway between its n-th and "
        "(n+1)-th nearest chunk in NumPy's exact cosine ranking of all chunk vectors, so exactly "
        "n chunks are nearer than it; \"Claims\" counts the claims that have one (see the last "
        "bullet). \"Expected\" is the number of rows of an exact answer: min(k, n), min(L, n, k) "
        "with an inner `LIMIT` L, min(k, chunks passing) for a predicate. \"Complete\", "
        f"\"Short\", \"Stopped\" and \"Server lost\" count the executions, {1 + passes} per "
        "claim: with the expected number of rows, with fewer, stopped at the time limit (error "
        "1969), and with the connection to the server lost, after which the part waited for the "
        "server, reconnected and went on with the next claim. \"Rows returned\" lists each number "
        "of rows with the claims that got it in brackets, from each claim's first execution that "
        "returned rows; of those claims, \"Equal to exact\" counts the ones whose list is, in "
        "order, the start of NumPy's exact ranking of the chunks that qualify, \"Same set\" those "
        "with the same chunks in any order. \"Rows read\" is the median (and the largest) of the "
        "`chunk` rows read, `r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`, over the claims "
        "whose executions were neither stopped nor lost (`ANALYZE` runs the statement once more). "
        "\"Chunk access\" is the type and key of `chunk` in `EXPLAIN`, which does not run the "
        "statement, over every claim; `index embedding` is the vector index. \"Timed\" is the "
        "number of timed executions that came back complete or short, the ones p50 and p95 are "
        "over; the times are of the statement and the fetch of its rows."),
    ]
    if report.get("stopped_early"):
        lines += ["", (f"The part stopped early: {report['stopped_early']}. "
                       f"{report['cases_done']} of the {report['cases_total']} cases were reached; "
                       "the tables hold what was measured until then.")]
    elif not report.get("finished"):
        lines += ["", (f"The run did not finish: {report['cases_done']} of the "
                       f"{report['cases_total']} cases were reached.")]
    lines += [
        "",
        "## The threshold inside the statement",
        "",
        "| n " + columns,
        "|---" + rule,
        *(f"| {row['n']} " + outcome(row) for row in threshold),
        "",
        "## The index search as a derived table, the threshold outside it",
        "",
        "| Inner LIMIT | n " + columns,
        "|---|---" + rule,
        *(f"| {row['inner_limit']} | {row['n']} " + outcome(row) for row in subquery),
        "",
        "## A selective predicate on `chunk`",
        "",
        "| Statement | Chunks passing " + columns,
        "|---|---" + rule,
        *(f"| {row['label']} | {row['passing_chunks']:,} " + outcome(row) for row in predicate),
        "",
        "How to read it:",
        "",
    ]

    def by_n(row: dict[str, Any]) -> str:
        return f"n = {row['n']}"

    def by_inner(row: dict[str, Any]) -> str:
        return f"L = {row['inner_limit']}, n = {row['n']}"

    def by_label(row: dict[str, Any]) -> str:
        return row["label"]

    def read_text(selected: list[dict[str, Any]], name: Callable[[dict[str, Any]], str]) -> str:
        return ", ".join(f"{_rows_read(row)} at {name(row)}" for row in selected)

    def access_text(selected: list[dict[str, Any]]) -> str:
        labels = {row["chunk_access"] for row in selected}
        every = all(row["chunk_access_count"] == row["claims"] for row in selected)
        if len(labels) == 1 and every:
            return f"`{labels.pop()}` for every claim"
        return "; ".join(f"`{_access(row)}` for {row['label']}" for row in selected)

    def measured(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in selected if row["chunk_rows_read_max"] is not None]

    if threshold:
        over = [row for row in measured(threshold) if row["chunk_rows_read_max"] > k]
        if over:
            cap = ("The server read more than `LIMIT` = "
                   f"{k} chunk rows from the index at {', '.join(by_n(row) for row in over)}: it "
                   "went on taking rows from the index after the first `LIMIT` of them.")
        elif measured(threshold):
            cap = ("Of the claims with their rows read measured, none read more than `LIMIT` = "
                   f"{k} chunk rows ({', '.join(by_n(row) for row in measured(threshold))}).")
            unmeasured = [row for row in threshold if row["chunk_rows_read_max"] is None]
            if unmeasured:
                cap += (f" At {', '.join(by_n(row) for row in unmeasured)} no claim has its rows "
                        "read measured: each had an execution stopped at the time limit or a lost "
                        "server, and `ANALYZE` was not run for it.")
        else:
            cap = "No claim had its rows read measured."
        lines.append(
            f"- Threshold inside the statement: `EXPLAIN` gives `chunk` {access_text(threshold)}. "
            f"Chunk rows read, median (largest): {read_text(threshold, by_n)}. {cap} Short "
            f"answers, of the executions: {counts(threshold, 'short', by_n, _executed)}; equal "
            "to the exact list, of the claims with rows: "
            f"{counts(threshold, 'equal_to_exact', by_n, answered)}."
        )
    if subquery:
        lines.append(
            "- Derived table: the outer `WHERE` sees the inner statement's `LIMIT` L rows only, "
            f"so an exact answer has min(L, n, k) rows. `EXPLAIN` gives `chunk` "
            f"{access_text(subquery)}. Chunk rows read, median (largest): "
            f"{read_text(subquery, by_inner)}. Short answers, of the executions: "
            f"{counts(subquery, 'short', by_inner, _executed)}; equal to the exact list, of the "
            f"claims with rows: {counts(subquery, 'equal_to_exact', by_inner, answered)}."
        )
    if predicate:
        shares = ", ".join(
            f"{entry['words_below']} words: {entry['passing_chunks']:,} chunks "
            f"({entry['passing_share']:.2%}, aimed at {entry['target_share']:.2%})"
            for entry in report["word_limits"]
        )
        total_short = sum(row["short"] for row in predicate)
        total = sum(_executed(row) for row in predicate)
        over = [row for row in measured(predicate) if row["chunk_rows_read_max"] > k]
        if over:
            reading = (f"The server read more than `LIMIT` = {k} chunk rows for "
                       f"{', '.join(row['label'] for row in over)}.")
        elif measured(predicate):
            reading = f"No claim read more than `LIMIT` = {k} chunk rows."
        else:
            reading = "No claim had its rows read measured."
        lines.append(
            "- Predicate on `chunk`, the word limits nearest 0.5% and 0.05% of the chunks on a log "
            f"scale (the rule of `scripts/scaled_filters.py`; {shares}): `EXPLAIN` gives `chunk` "
            f"{access_text(predicate)}. {total_short} of the {total} executions came back short. "
            f"Chunk rows read, median (largest): {read_text(predicate, by_label)}. {reading} "
            "Equal to the exact list of the chunks that pass, of the claims with rows: "
            f"{counts(predicate, 'equal_to_exact', by_label, answered)}."
        )

    executed = sum(_executed(row) for row in rows)
    scan = check["scan"]
    stopped = sum(row["stopped"] for row in rows)
    analyze_stopped = sum(row["analyze"][STOPPED] for row in rows)
    if stopped or analyze_stopped or scan[STOPPED]:
        per_case = ", ".join(f"{row['stopped']} of {_executed(row)} for {row['label']}"
                             for row in rows if row["stopped"])
        lines.append(
            f"- Stopped at the time limit of {limit:g} s: {stopped} of the {executed} executions"
            + (f" ({per_case})" if per_case else "")
            + f", `ANALYZE` for {_plural(analyze_stopped, 'claim')} and the scan check for "
            f"{_plural(scan[STOPPED], 'claim')}. A stopped execution has neither rows nor a time, "
            "and its claim has no rows read."
        )
    else:
        lines.append(f"- No statement reached the time limit of {limit:g} s.")
    lost = sum(row["server_lost"] for row in rows)
    if report["server_losses"]:
        per_case = ", ".join(f"{row['server_lost']} of {_executed(row)} for {row['label']}"
                             for row in rows if row["server_lost"])
        lines.append(
            f"- Server lost: {lost} of the {executed} executions"
            + (f" ({per_case})" if per_case else "")
            + ". With `EXPLAIN`, `ANALYZE` and the scan check, the connection was lost "
            f"{_plural(report['server_losses'], 'time')} and the part reconnected "
            f"{_plural(report['reconnects'], 'time')}; the timed executions after a reconnect "
            "count like the others."
        )
    else:
        lines.append("- The connection to the server was never lost.")
    not_run = sum(row["not_run"] for row in rows)
    if not_run:
        lines.append(
            f"- Executions not run: {not_run}, the rest of a claim after a lost server"
            + (" and the claims after the part stopped." if report.get("stopped_early") else ".")
        )

    finished = sum(entry["scan_finished"] for entry in check["per_n"])
    unfinished = scan[STOPPED] + scan[SERVER_LOST] + scan[NOT_RUN]
    lines.append(
        "- The thresholds: the server's own scan without the index (`IGNORE INDEX (embedding)`) "
        "found exactly n chunks under the threshold for "
        + ", ".join(f"{entry['server_count_equal_n']} of {entry['scan_finished']} claims "
                    f"at n = {entry['n']}" for entry in check["per_n"])
        + "."
        + (f" The scan did not finish for {_plural(unfinished, 'claim')} ({scan[STOPPED]} "
           f"stopped at the time limit, {scan[SERVER_LOST]} with the server lost, "
           f"{scan[NOT_RUN]} not run); "
           "their thresholds were used all the same." if unfinished else "")
        + " Left out, because the n-th and (n+1)-th exact distances were at most "
        f"{report['tie_margin']:g} apart: "
        + ", ".join(f"{entry['left_out_ties']} at n = {entry['n']}" for entry in check["per_n"])
        + "."
        + (" The largest difference between a distance the server computed and NumPy's was "
           f"{check['server_numpy_distance_gap_max']:.1e}." if finished
           else " No scan finished, so no distance of the server was compared with NumPy's.")
    )
    lines += [
        "",
        ("The statements (the vector, the threshold, the limits and the word limit are bound "
         "parameters), each after the session's time limit:"),
        "",
        "```sql",
        HARD_CAP_TIME_SET_SQL.replace("%s", f"{limit:g}") + ";",
        HARD_CAP_THRESHOLD_SQL + ";",
        HARD_CAP_SUBQUERY_SQL + ";",
        FORM_PREDICATE + ";",
        FORM_PREDICATE_JOINED + ";",
        HARD_CAP_SCAN_SQL + ";",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------


def _positive_ints(text: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive integers (an argparse type)."""
    try:
        values = tuple(int(part) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def selected_parts(named: Sequence[str], throwaway_server: bool) -> list[str]:
    """The parts to run: those ``named``, or every part in :data:`PARTS` order when none is
    named, hard_cap only with ``throwaway_server``."""
    if named:
        return list(named)
    return [part for part in PARTS if part != "hard_cap" or throwaway_server]


def _write_json(path: Path, report: dict[str, Any]) -> None:
    """Write ``report`` to ``path`` through a temporary file, so the file is never half written."""
    scratch = path.with_name(path.name + ".tmp")
    scratch.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    scratch.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parts", nargs="*", metavar="PART",
                        help="forms, join_order, builds, cache, isolation, hard_cap or several "
                             "(default: all; hard_cap only with --throwaway-server)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results",
                        help="output directory (default: results/ in the repository)")
    parser.add_argument("--database",
                        help="the corpus database to measure (default: WIKILENSE_DB_NAME); the "
                             "scratch tables of builds and isolation stay in the test database")
    parser.add_argument("--k", type=int, default=K,
                        help=f"hits per query for forms, join_order, builds, cache and hard_cap "
                             f"(default {K})")
    parser.add_argument("--filters", type=Path,
                        help="join_order: the filter cases of a scripts/scaled_filters.py file "
                             "instead of the planner experiment's")
    parser.add_argument("--cache-mb", type=_positive_ints, default=CACHE_SIZES_MB,
                        help="cache: the sizes in MB, in the order to set them (default "
                             + ",".join(str(size) for size in CACHE_SIZES_MB) + ")")
    parser.add_argument("--repeats", type=int,
                        help=f"timed passes per claim for forms ({FORMS_REPEATS}), join_order "
                             f"({JOIN_ORDER_REPEATS}) and hard_cap ({HARD_CAP_REPEATS}), after "
                             "one untimed pass")
    parser.add_argument("--passes", type=int, default=CACHE_PASSES,
                        help=f"cache: passes over the claims per size (default {CACHE_PASSES})")
    parser.add_argument("--sample", type=int,
                        help=f"join_order and hard_cap: a random sample of this many claims "
                             f"(seed {SAMPLE_SEED})")
    parser.add_argument("--no-turns", action="store_true",
                        help="join_order: time each form alone only, not also taking turns")
    parser.add_argument("--builds", type=int, default=BUILDS,
                        help=f"builds: graph builds per M (default {BUILDS})")
    parser.add_argument("--build-m", type=_positive_ints, default=tuple(BUILD_CREATE_SQL),
                        help="builds: the values of M (default "
                             + ",".join(str(m) for m in BUILD_CREATE_SQL) + ")")
    parser.add_argument("--throwaway-server", action="store_true",
                        help="hard_cap: the server is a throwaway MariaDB container with a "
                             "--memory cap; hard_cap runs only with this flag")
    args = parser.parse_args(argv)
    parts = selected_parts(args.parts, args.throwaway_server)
    unknown = [part for part in parts if part not in PARTS]
    if unknown:
        parser.error(f"unknown part(s) {unknown}; choose from {', '.join(PARTS)}")
    if "hard_cap" in parts and not args.throwaway_server:
        parser.error(HARD_CAP_REFUSAL)
    optional = [value for value in (args.repeats, args.sample) if value is not None]
    if min(args.k, args.builds, args.passes, *optional) < 1:
        parser.error("--k, --builds, --passes, --repeats and --sample must be at least 1")
    if args.passes < 2 and "cache" in (args.parts or PARTS):
        parser.error("--passes must be at least 2: the cache part compares two passes")
    settings = load_settings()
    if args.database:
        if args.database == settings.test_db_name:
            parser.error("--database names the test database; measure a corpus database")
        settings = replace(settings, db_name=args.database)
    if (settings.db_name != Settings.__dataclass_fields__["db_name"].default
            and args.out.resolve() == (REPO_ROOT / "results").resolve()):
        parser.error(f"{settings.db_name} is not the 100-page corpus: give --out a folder of "
                     "its own, so the files in results/ are not overwritten")
    cases = None
    if args.filters is not None:
        from scaled_filters import load_cases

        cases = load_cases(args.filters)
    args.out.mkdir(parents=True, exist_ok=True)
    for part in dict.fromkeys(parts):
        started = time.perf_counter()
        if part == "forms":
            from wikilense.cli import make_embedder

            report = measure_forms(settings, make_embedder(settings), args.repeats or FORMS_REPEATS,
                                   k=args.k)
            text = forms_markdown(report)
        elif part == "join_order":
            from wikilense.cli import make_embedder

            report = measure_join_order(settings, make_embedder(settings),
                                        args.repeats or JOIN_ORDER_REPEATS, k=args.k, cases=cases,
                                        sample=args.sample, taking_turns=not args.no_turns)
            text = join_order_markdown(report)
        elif part == "builds":
            from wikilense.cli import make_embedder

            report = measure_builds(settings, make_embedder(settings), args.builds, k=args.k,
                                    ms=args.build_m)
            text = builds_markdown(report)
        elif part == "cache":
            from wikilense.cli import make_embedder

            report = measure_cache(settings, make_embedder(settings), args.passes, k=args.k,
                                   sizes_mb=args.cache_mb)
            text = cache_markdown(report)
        elif part == "hard_cap":
            from wikilense.cli import make_embedder

            def checkpoint(partial_report: dict[str, Any]) -> None:
                _write_json(args.out / "hard_cap.json",
                            {**partial_report, "database": settings.db_name})

            report = measure_hard_cap(settings, make_embedder(settings),
                                      args.repeats or HARD_CAP_REPEATS, k=args.k,
                                      sample=args.sample, checkpoint=checkpoint)
            text = hard_cap_markdown(report)
        else:
            report = measure_isolation(settings, ISOLATION_TRIALS, ISOLATION_CONTROL_TRIALS)
            text = isolation_markdown(report)
        report["database"] = settings.db_name
        stem = report["name"]
        _write_json(args.out / f"{stem}.json", report)
        (args.out / f"{stem}.md").write_text(text, encoding="utf-8")
        print(f"{part}: wrote {args.out / (stem + '.md')} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
