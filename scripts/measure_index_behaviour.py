#!/usr/bin/env python3
"""Five behaviours of MariaDB's vector index that its documentation does not describe, measured.

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

Usage (from the project root, with the database up and the corpus ingested):

    python scripts/measure_index_behaviour.py            # all five
    python scripts/measure_index_behaviour.py forms
    python scripts/measure_index_behaviour.py join_order
    python scripts/measure_index_behaviour.py builds
    python scripts/measure_index_behaviour.py cache
    python scripts/measure_index_behaviour.py isolation

Do not run it while the test suite or another measurement runs: ``builds`` and ``isolation``
use the test database, and ``cache`` changes a server-wide setting for its duration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date, datetime, timezone
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import compare_separate_index as separate  # the exact top k and the interleaved timing

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
JOIN_ORDER_EF = 100
#: The scan that every hit list is compared with: the statement as written, without the index.
SCAN_FROM = ("FROM chunk ", "FROM chunk IGNORE INDEX (embedding) ")
#: Position of ``distance`` in the inline statement's select list (search.INLINE_SELECT).
DISTANCE_COLUMN = 6
#: Two medians that differ by less than this factor are reported as the same time. Two runs of
#: one statement on this laptop differed by up to 1.24, and two forms with the same plan by 1.10.
NOISE_FACTOR = 1.5
PARTS: tuple[str, ...] = ("forms", "join_order", "builds", "cache", "isolation")

CACHE_PASSES = 3
CACHE_EF = 100
MEBIBYTE = 1048576
#: Cache sizes below the configured one, in MB: the server default first, then smaller.
CACHE_SIZES_MB: tuple[int, ...] = (16, 8, 4, 2, 1)
#: One literal statement per size; a server variable cannot be a bound parameter here because
#: the statement runs through the mariadb client inside the container.
CACHE_SET_SQL: dict[int, str] = {
    size: "SET GLOBAL mhnsw_max_cache_size = " + str(size * MEBIBYTE) for size in CACHE_SIZES_MB
}
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


# ---------------------------------------------------------------------------------------------
# forms
# ---------------------------------------------------------------------------------------------


def measure_forms(settings: Settings, embedder: Any, repeats: int) -> dict[str, Any]:
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
                separate.top_k(vectors, queries[i].astype(np.float64), ids, K, passing)
                for i in range(n_queries)
            ]

        def answer_of(case: dict[str, Any]) -> Callable[[int], list[int]]:
            def answer(i: int) -> list[int]:
                vector = db.vec_param(queries[i])
                params = (vector, K) if case["limit"] is None else (case["limit"], vector, K)
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
                    params = (vector, K) if case["limit"] is None else (case["limit"], vector, K)
                    cur.execute("ANALYZE " + case["sql"], params)
                    plan = cur.fetchall()
                    read.append(float(plan[0][9]))  # r_rows of the first table, chunk
            return median(read)

        def chunk_access(case: dict[str, Any]) -> str:
            """'<type> <key>' of the chunk row in EXPLAIN, for the first claim."""
            vector = db.vec_param(queries[0])
            params = (vector, K) if case["limit"] is None else (case["limit"], vector, K)
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
            "name": "statement_forms", "created_at": utc_now(), "k": K, "repeats": repeats,
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
        "top 10 of the chunks that pass the predicate. \"Rows read\" is the median over the claims "
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
        "Other builds of the graph differ (`results/build_variance.md`)."),
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


def measure_join_order(settings: Settings, embedder: Any, repeats: int) -> dict[str, Any]:
    """Run the filtered statement with its join order fixed and free; return the report."""
    from run_experiments import PLANNER_CASES  # the same filters as results/planner_choices.md

    conn = db.connect(settings)
    try:
        truths = evaluate.load_ground_truth(conn)
        if not truths:
            raise SystemExit("the claim table is empty: run 'wikilense ingest' first")
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
            for name, filters in PLANNER_CASES:
                statements: list[tuple[str, tuple[Any, ...], str, str]] = []
                for query in queries:
                    sql, params = searchmod.search_statement(query, K, filters, "inline")
                    statements.append((sql, params, free_order(sql), scan_form(sql)))

                def answer_of(form: int, stmts: list = statements) -> Callable[[int], list[Any]]:
                    return lambda i: hits_of(stmts[i][form], stmts[i][1])

                scans = [hits_of(stmt[3], stmt[1]) for stmt in statements]
                sql0, params0 = searchmod.search_statement(queries[0], n_chunks, filters, "inline")
                passing = len(hits_of(scan_form(sql0), params0))
                timed = separate.time_methods([answer_of(0), answer_of(2)], n_queries, repeats)
                # Each form alone as well: when the two take turns, a form that reads thousands
                # of rows slows the other one's next run (measured: 2.2 against 2.6 ms).
                alone = [
                    separate.time_methods([answer_of(form)], n_queries, repeats)[0][0]
                    for form in (0, 2)
                ]
                row: dict[str, Any] = {"filter": name, "passing_chunks": passing}
                for key, form, (samples, lists), own in zip(
                    ("as_written", "free"), (0, 2), timed, alone, strict=True
                ):
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
                    turns = evaluate.latency_stats(samples)
                    plan, plan_count = plans.most_common(1)[0]
                    row[key] = {
                        "plan": plan, "plan_count": plan_count, "plans": dict(plans),
                        "explain_first_claim": first_plan,
                        "chunk_rows_read_p50": median(read), "chunk_rows_read_max": max(read),
                        "p50_ms": round(stats.p50_ms, 3), "p95_ms": round(stats.p95_ms, 3),
                        "p50_ms_taking_turns": round(turns.p50_ms, 3),
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
            "name": "join_order", "created_at": utc_now(), "k": K, "repeats": repeats,
            "ef_search": JOIN_ORDER_EF, "n_queries": n_queries, "n_chunks": n_chunks,
            **db.vector_index_info(conn),
            "results": results,
            "derived_without_distinct": merged_tables,
            "filters": {
                name: {key: (str(value) if isinstance(value, date) else value)
                       for key, value in asdict(filters).items() if value is not None}
                for name, filters in PLANNER_CASES
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
        return (f"{plan} | {side['chunk_rows_read_p50']:.0f} | {side['p50_ms']:.2f} "
                f"({side['p50_ms_taking_turns']:.2f}) | "
                f"{side['same_rows_as_scan']} / {side['same_order_as_scan']} |")

    lines = [
        "# The filtered statement with its join order fixed, and left to the optimizer",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py join_order` on "
        f"MariaDB {report['versions']['mariadb']}: {report['n_chunks']:,} chunks, index M = "
        f"{report['index_m']}, the {n} corpus claims as queries, k = {report['k']}, "
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
        f"pass over the claims, then {report['repeats']} timed passes. In brackets is the same "
        "median when the two forms take turns query by query. Taking turns guards against the "
        "machine drifting between two blocks, but a form that reads thousands of rows slows down "
        "the other form's next run, so the bracketed time of the faster form is the larger one."),
        "",
        ("| Filter | Chunks passing | As written: plan | Rows read | p50 ms (taking turns) "
        "| Same rows / order | Free: plan | Rows read | p50 ms (taking turns) | Same rows / order |"),
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


def measure_builds(settings: Settings, embedder: Any, builds: int) -> dict[str, Any]:
    """Build the index ``builds`` times per M on a scratch table and search each build."""
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
        separate.top_k(vectors, queries[i].astype(np.float64), ids, K, everything)
        for i in range(len(truths))
    ]

    def recalled(lists: Sequence[Sequence[int]]) -> int:
        return sum(
            1 for hits, truth in zip(lists, truths, strict=True)
            if evaluate.article_recall_at_k([page_of[c] for c in hits], truth.gold_pages, K)
        )

    conn = db.connect(settings, database=settings.test_db_name)
    results: list[dict[str, Any]] = []
    try:
        for m, create_sql in BUILD_CREATE_SQL.items():
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
                            cur.execute(BUILD_KNN_SQL, (db.vec_param(query), K))
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
                    "identical_between_two_builds": spread(identical_between_builds(per_build)),
                    "build_seconds": spread(seconds),
                })  # fmt: skip
        with conn.cursor() as cur:
            cur.execute(BUILD_DROP_SQL)
        conn.commit()
    finally:
        conn.close()
    return {
        "name": "build_variance", "created_at": utc_now(), "k": K, "n_queries": len(truths),
        "n_chunks": int(ids.shape[0]), "builds": builds, "batch": BUILD_BATCH,
        "exact_article_recall": recalled(exact),
        "results": results, "ingest_meta": meta,
        "machine": evaluate.machine_info(embedder), "versions": versions,
    }  # fmt: skip


def builds_markdown(report: dict[str, Any]) -> str:
    n = report["n_queries"]

    def whole(values: dict[str, float]) -> str:
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
        "whose hit list is NumPy's exact top 10 in order. \"Two builds agree\" counts, for each "
        "pair of builds, the claims with identical hit lists. Each cell is the smallest to the "
        "largest value over the builds (or pairs), one number when they are all the same."),
        "",
        (f"| M | ef_search | Article recall@10, of {n} | Equal to exact, of {n} "
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
        *(sql + ";" for sql in BUILD_CREATE_SQL.values()),
        BUILD_INSERT_SQL + ";",
        BUILD_KNN_SQL + ";",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------------


def measure_cache(settings: Settings, embedder: Any, passes: int) -> dict[str, Any]:
    """Search with the index cache at several sizes; return the report.

    The configured ``mhnsw_max_cache_size`` is read first and restored at the end, also when a
    step fails.
    """
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
        separate.top_k(vectors, queries[i].astype(np.float64), ids, K, everything)
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
                            hits = searchmod.search(reader, query, k=K, strategy=strategy)
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
                                                        truth.gold_pages, K))
                    for lists in per_pass
                ],
                "overlap_with_exact": [
                    round(float(np.mean([len(set(got) & set(want)) / len(want)
                                         for got, want in zip(lists, exact, strict=True)])), 4)
                    for lists in per_pass
                ],
                "short": [sum(1 for got in lists if len(got) < K) for lists in per_pass],
                "identical_between_two_passes": identical_between_builds(per_pass),
                "p50_ms": round(evaluate.latency_stats(times[strategy]).p50_ms, 3),
            })  # fmt: skip
        return out

    results: list[dict[str, Any]] = []
    restored = None
    try:
        results += step("as configured", configured, reopened=False)
        smallest = CACHE_SIZES_MB[-1]
        server.root_sql(CACHE_SET_SQL[smallest])
        results += step(f"{smallest} MB, table not reopened", smallest * MEBIBYTE, reopened=False)
        for size in CACHE_SIZES_MB:
            server.root_sql(CACHE_SET_SQL[size])
            server.root_sql(CHUNK_FLUSH_SQL, database=settings.db_name)
            results += step(f"{size} MB", size * MEBIBYTE, reopened=True)
    finally:
        server.root_sql("SET GLOBAL mhnsw_max_cache_size = " + str(int(configured)))
        server.root_sql(CHUNK_FLUSH_SQL, database=settings.db_name)
        restored = int(server.root_sql(CACHE_GET_SQL)[0][0])
    results += step("as configured, again", restored, reopened=True)
    return {
        "name": "cache_size", "created_at": utc_now(), "k": K, "passes": passes,
        "ef_search": CACHE_EF, "n_queries": len(truths), "n_chunks": int(ids.shape[0]),
        "configured_cache_bytes": configured, "restored_cache_bytes": restored,
        "exact_article_recall": sum(
            1 for hits, truth in zip(exact, truths, strict=True)
            if evaluate.article_recall_at_k([page_of[c] for c in hits], truth.gold_pages, K)
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
        "is NumPy's exact top 10 in order. \"Two passes agree\" counts, for each pair of passes, "
        "the claims with identical lists. The times are of the whole call."),
        "",
        (f"| Cache | Statement | Equal to exact, of {n}, per pass | Article recall@10, of {n} "
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
        *(sql + ";" for sql in CACHE_SET_SQL.values()),
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
        ("- A reader did not cause the state when it searched at READ COMMITTED or in autocommit "
        "mode, when it ended its transaction before searching again, or when its transaction "
        "searched the index only after the other connection's commit. Once the state is there, "
        "connections at every level get the stale answer. So an application that writes while "
        "it searches is safe only if none of its connections searches the index twice inside "
        "one REPEATABLE READ transaction."),
        ("- PyMySQL opens connections with autocommit off, so a connection that has run one "
        "`SELECT` is inside a REPEATABLE READ transaction until it commits, rolls back or "
        "closes."),
        ("- WikiLense's measurements are not affected: no connection commits a write to `chunk` "
        "while a search runs, and `wikilense ingest` recreates the tables."),
        ("- We found no MariaDB ticket that describes this (searched on 2026-11-11) and have not "
        "reported it. It was measured on one server version, on one scratch table of "
        f"{report['rows']:,} random vectors and one query, rebuilt for every trial."),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parts", nargs="*", metavar="PART",
                        help="forms, join_order, builds, cache, isolation or several "
                             "(default: all)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results",
                        help="output directory (default: results/ in the repository)")
    args = parser.parse_args(argv)
    parts = args.parts or list(PARTS)
    unknown = [part for part in parts if part not in PARTS]
    if unknown:
        parser.error(f"unknown part(s) {unknown}; choose from {', '.join(PARTS)}")
    settings = load_settings()
    args.out.mkdir(parents=True, exist_ok=True)
    for part in dict.fromkeys(parts):
        started = time.perf_counter()
        if part == "forms":
            from wikilense.cli import make_embedder

            report = measure_forms(settings, make_embedder(settings), FORMS_REPEATS)
            text = forms_markdown(report)
        elif part == "join_order":
            from wikilense.cli import make_embedder

            report = measure_join_order(settings, make_embedder(settings), JOIN_ORDER_REPEATS)
            text = join_order_markdown(report)
        elif part == "builds":
            from wikilense.cli import make_embedder

            report = measure_builds(settings, make_embedder(settings), BUILDS)
            text = builds_markdown(report)
        elif part == "cache":
            from wikilense.cli import make_embedder

            report = measure_cache(settings, make_embedder(settings), CACHE_PASSES)
            text = cache_markdown(report)
        else:
            report = measure_isolation(settings, ISOLATION_TRIALS, ISOLATION_CONTROL_TRIALS)
            text = isolation_markdown(report)
        stem = report["name"]
        (args.out / f"{stem}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        (args.out / f"{stem}.md").write_text(text, encoding="utf-8")
        print(f"{part}: wrote {args.out / (stem + '.md')} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
