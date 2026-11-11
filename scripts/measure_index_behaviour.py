#!/usr/bin/env python3
"""Three behaviours of MariaDB's vector index that its documentation does not describe, measured.

``forms``
    One nearest-neighbour statement in three shapes, on the ingested corpus (read only): the
    bare statement, the same with a predicate on the indexed table itself, and the same with a
    join to another table. Every hit list is compared with NumPy's exact ranking of the rows
    that pass. Result (MariaDB 11.8.9): the bare statement is the HNSW search; a predicate on
    the table keeps that search going until ``k`` rows pass, so it is never short but stays
    approximate and gets slower the fewer rows pass; a join makes the result exact and
    independent of ``mhnsw_ef_search``. Writes ``results/statement_forms.json`` and ``.md``.

``join_order``
    The filtered statement of ``wikilense query`` as written, with ``STRAIGHT_JOIN`` and
    ``chunk`` first, against the same text with plain ``JOIN``, where the optimizer picks the
    order (read only, the filters of the ``planner`` experiment). Every hit list is compared
    with the same statement under ``IGNORE INDEX (embedding)``, a scan of every chunk. Result
    (MariaDB 11.8.9): left free, the optimizer starts from the filter's own index whenever one
    exists and never walks the vector index then; that is much cheaper under a selective
    filter. Writes ``results/join_order.json`` and ``.md``.

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

    python scripts/measure_index_behaviour.py            # all three
    python scripts/measure_index_behaviour.py forms
    python scripts/measure_index_behaviour.py join_order
    python scripts/measure_index_behaviour.py isolation

Do not run it while the test suite runs: both use the test database.
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
#: Two medians of one run that differ by less than this factor are reported as the same time
#: (two runs of the same statement on this laptop differed by up to 1.24).
NOISE_FACTOR = 1.25
PARTS: tuple[str, ...] = ("forms", "join_order", "isolation")

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
    lines = [
        "# One nearest-neighbour statement in three shapes",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py forms` on "
        f"MariaDB {report['versions']['mariadb']}: {report['n_chunks']:,} chunks, index M = "
        f"{report['index_m']}, the {n} corpus claims as queries, k = {report['k']}, "
        f"{report['repeats']} timed passes after one untimed pass, the forms taking turns query "
        "by query. \"Equal to exact\" counts the claims whose hit list is, in order, NumPy's exact "
        "top 10 of the chunks that pass the predicate. \"Rows read\" is the median over the claims "
        "of `r_rows` for `chunk` in `ANALYZE`."),
        "",
        "| Statement | Chunks passing | ef_search | p50 ms | p95 ms | Equal to exact | Short | Rows read |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['label']} | {row['passing_chunks']:,} | {row['ef_search']} | "
            f"{row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['equal_to_exact']}/{n} | "
            f"{row['short']}/{n} | {row['chunk_rows_read_p50']:.0f} |"
        )
    lines += [
        "",
        "How to read it:",
        "",
        ("- The bare statement is the HNSW search: it reads `LIMIT` rows, and `mhnsw_ef_search` "
        "decides how often it equals the exact ranking."),
        ("- A predicate on the indexed table itself does not stop that search at `LIMIT`: the "
        "server keeps asking the index for more rows until `k` of them pass. No query comes back "
        "short. The result stays approximate, more so the fewer rows pass, and a larger "
        "`mhnsw_ef_search` helps only in part. It also gets slower than the joined form."),
        ("- With a join in the statement the same predicate gives the exact answer for every "
        "claim, at both `mhnsw_ef_search` values, at a cost that does not depend on them."),
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
    """
    same_rows = same_order = short = 0
    gap = 0.0
    for got, want in zip(lists, scans, strict=True):
        got_ids, want_ids = [hit[0] for hit in got], [hit[0] for hit in want]
        short += int(len(got_ids) < len(want_ids))
        if set(got_ids) != set(want_ids):
            continue
        same_rows += 1
        same_order += int(got_ids == want_ids)
        for mine, theirs in zip(got, want, strict=True):
            if mine[0] != theirs[0]:
                gap = max(gap, abs(mine[1] - theirs[1]))
    return {"same_rows_as_scan": same_rows, "same_order_as_scan": same_order, "short": short,
            "swap_gap_max": gap}


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
        results: list[dict[str, Any]] = []
        with searchmod.ef_search_session(conn, JOIN_ORDER_EF):
            for name, filters in PLANNER_CASES:
                statements: list[tuple[str, tuple[Any, ...], str, str]] = []
                for query in queries:
                    sql, params = searchmod.search_statement(query, K, filters, "inline")
                    statements.append((sql, params, free_order(sql), scan_form(sql)))

                def hits_of(sql: str, params: tuple[Any, ...]) -> list[tuple[int, float]]:
                    """(chunk_id, distance) in the order the server returned the rows."""
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        return [(int(row[0]), float(row[DISTANCE_COLUMN])) for row in cur.fetchall()]

                def answer_of(form: int, stmts: list = statements) -> Callable[[int], list[Any]]:
                    return lambda i: hits_of(stmts[i][form], stmts[i][1])

                scans = [hits_of(stmt[3], stmt[1]) for stmt in statements]
                sql0, params0 = searchmod.search_statement(queries[0], n_chunks, filters, "inline")
                passing = len(hits_of(scan_form(sql0), params0))
                timed = separate.time_methods([answer_of(0), answer_of(2)], n_queries, repeats)
                row: dict[str, Any] = {"filter": name, "passing_chunks": passing}
                for key, form, (samples, lists) in zip(
                    ("as_written", "free"), (0, 2), timed, strict=True
                ):
                    plans: Counter[str] = Counter()
                    read: list[float] = []
                    with conn.cursor() as cur:
                        for stmt in statements:
                            plans[plan_label(db.explain(conn, stmt[form], stmt[1]))] += 1
                            cur.execute("ANALYZE FORMAT=JSON " + stmt[form], stmt[1])
                            read.append(chunk_rows_read(json.loads(cur.fetchone()[0])))
                    stats = evaluate.latency_stats(samples)
                    plan, plan_count = plans.most_common(1)[0]
                    row[key] = {
                        "plan": plan, "plan_count": plan_count, "plans": dict(plans),
                        "chunk_rows_read_p50": median(read), "chunk_rows_read_max": max(read),
                        "p50_ms": round(stats.p50_ms, 3), "p95_ms": round(stats.p95_ms, 3),
                        **agreement(lists, scans),
                    }  # fmt: skip
                results.append(row)
        return {
            "name": "join_order", "created_at": utc_now(), "k": K, "repeats": repeats,
            "ef_search": JOIN_ORDER_EF, "n_queries": n_queries, "n_chunks": n_chunks,
            **db.vector_index_info(conn),
            "results": results,
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
        return (f"{plan} | {side['chunk_rows_read_p50']:.0f} | {side['p50_ms']:.2f} | "
                f"{side['same_rows_as_scan']} / {side['same_order_as_scan']} |")

    lines = [
        "# The filtered statement with its join order fixed, and left to the optimizer",
        "",
        (f"Generated {report['created_at']} by `scripts/measure_index_behaviour.py join_order` on "
        f"MariaDB {report['versions']['mariadb']}: {report['n_chunks']:,} chunks, index M = "
        f"{report['index_m']}, the {n} corpus claims as queries, k = {report['k']}, "
        f"`mhnsw_ef_search` {report['ef_search']}, {report['repeats']} timed passes after one "
        "untimed pass, the two forms taking turns query by query. \"As written\" is the "
        "statement `wikilense query` runs: `STRAIGHT_JOIN`, `chunk` first. \"Free\" is the same "
        "text with every `STRAIGHT_JOIN` replaced by `JOIN`, so the optimizer picks the order. "
        "\"Plan\" is the first table of `EXPLAIN` and how `chunk` is read; `index embedding` is "
        "the vector index. \"Rows read\" is the median over the claims of the `chunk` rows read "
        "(`r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`). \"Same rows / order\" counts the "
        f"claims, of {n}, whose hit list holds the same chunks, and the same chunks in the same "
        "order, as the same statement under `IGNORE INDEX (embedding)`, which reads and sorts "
        "every chunk. The times are the statement and the fetch of its rows, without the "
        "Python sort that `wikilense` adds."),
        "",
        ("| Filter | Chunks passing | As written: plan | Rows read | p50 ms | Same rows / order "
        "| Free: plan | Rows read | p50 ms | Same rows / order |"),
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
        f"{len(rows)} filters: the filter's own index or derived table. It then reads only the "
        "chunks that pass, through `uq_chunk_page_ordinal`, and sorts them. The vector index is "
        "not used in that plan."),
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
            f"claims, largest gap {side['swap_gap_max']:.1e}"
            for name, key, side in swapped
        )
        lines.append(
            "- Same chunks in another order: the server returned two neighbours the other way "
            "round than the distances it reports for them, which differ by the gap given here "
            f"({notes}). `wikilense` sorts the rows by the reported distance and the chunk id "
            "afterwards."
        )
    else:
        lines.append("- No hit list had its chunks in another order than the scan.")
    lines += ["", "Filters:", ""]
    for name, filters in report["filters"].items():
        lines.append(f"- `{name}`: {json.dumps(filters, ensure_ascii=False)}")
    lines.append("")
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
            cur.execute("SET SESSION mhnsw_ef_search = %s", (ISOLATION_EF,))
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

    def fresh_view(self, new_ids: Sequence[int] = NEW_ROW_IDS) -> dict[str, int]:
        """What a new connection sees: new rows among the top k through the index and by a scan."""
        conn = self.connect()
        try:
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
    probe: Probe, level: str, search_again: bool, close_reader: bool, autocommit: bool = False
) -> dict[str, int]:
    """Reader searches, writer commits five near rows, reader (maybe) searches again and ends.

    With ``autocommit`` the reader opens no transaction: each of its searches is its own.
    Returns what a new connection then sees (:meth:`Probe.fresh_view`).
    """
    probe.fresh_table()
    reader = probe.connect(level)
    try:
        if autocommit:
            reader.autocommit(True)
        else:
            reader.begin()
        probe.search(reader)
        probe.insert_and_commit(NEW_ROW_IDS)
        if search_again:
            probe.search(reader)
        if not autocommit:
            reader.commit()
        if close_reader:
            reader.close()
        return probe.fresh_view()
    finally:
        if reader.open:
            reader.close()


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
        }

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
        deletes = []
        for _ in range(trials):
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
                in_snapshot = probe.search(reader)
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
            deletes.append({
                "reader_snapshot_rows": len(in_snapshot),
                "reader_snapshot_still_sees_deleted": sum(1 for v in victims if v in in_snapshot),
                "new_connection_index_rows": len(index_hits),
                "new_connection_scan_rows": len(scan_hits),
                "other_queries_min_rows": min(other_rows),
                "other_queries_with_no_row": sum(1 for count in other_rows if count == 0),
                "table_rows": table_rows,
            })  # fmt: skip
        delete_flush = None
        if deletes and deletes[-1]["new_connection_index_rows"] == 0 and not recovery["flush_skipped"]:
            reason = flush_as_root(settings)
            if reason is None:
                conn = probe.connect()
                try:
                    delete_flush = len(probe.search(conn))
                finally:
                    conn.close()
        return {
            "name": "isolation", "created_at": utc_now(), "k": K,
            "rows": ISOLATION_ROWS, "new_rows": len(NEW_ROW_IDS), "ef_search": ISOLATION_EF,
            "index": "VECTOR INDEX (embedding) M=16 DISTANCE=cosine", "seed": ISOLATION_SEED,
            "trials": trials, "control_trials": control_trials,
            "mariadb_version": str(version), "default_isolation_level": str(default_level),
            "innodb_snapshot_isolation": int(snapshot_isolation),
            "levels": levels, "sequences": sequences, "recovery": recovery, "deletes": deletes,
            "delete_rows_after_flush": delete_flush,
        }  # fmt: skip
    finally:
        probe.drop_table()


def _span(values: Sequence[int]) -> str:
    """One number when every trial agrees, else "low to high"."""
    return str(values[0]) if len(set(values)) == 1 else f"{min(values)} to {max(values)}"


def _stale_count(views: Sequence[dict[str, int]]) -> str:
    stale = sum(1 for view in views if view["index_new"] < view["scan_new"])
    return f"{stale} of {len(views)}"


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
        "3. A searches again, still inside its old snapshot (correctly without the new rows).",
        "4. A commits and disconnects.",
        "5. A new connection searches.",
        "",
        ("At step 5 the table holds the new rows and a scan returns them. \"Stale\" below means the "
        "index search of the new connection returned fewer of them than the scan."),
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
        "",
        ("Once stale, it stays stale for later connections: three more new connections found "
        f"{recovery['still_stale_for_new_connections']} new rows (one list per stale table). "
        "What cleared it:"),
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
    lines += [
        "",
        "So the index on disk is right and the copy the server keeps in memory is not.",
        "",
        "## The same sequence with a DELETE",
        "",
        ("B deletes the three rows nearest to the query instead of inserting. A's snapshot keeps "
        "seeing them, as it should. Afterwards, for a new connection:"),
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
        "## What follows for an application",
        "",
        ("- Search at READ COMMITTED or in autocommit mode, or end the transaction before searching "
        "again, when other connections write to the table. PyMySQL opens connections with "
        "autocommit off, so a connection that has run one `SELECT` is inside a REPEATABLE READ "
        "transaction until it commits, rolls back or closes."),
        ("- WikiLense itself is not affected by this in its measurements: nothing writes to `chunk` "
        "while a search runs, and `wikilense ingest` recreates the tables."),
        ("- We found no MariaDB ticket that describes this (searched on 2026-11-11) and have not "
        "reported it. It was measured on one server version only."),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parts", nargs="*", metavar="PART",
                        help="forms, join_order, isolation or several (default: all three)")
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
