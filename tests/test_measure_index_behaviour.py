"""Tests for scripts/measure_index_behaviour.py: the parts that need no server and no model."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

from wikilense.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "measure_index_behaviour.py"


@pytest.fixture(scope="module")
def mib() -> Any:
    spec = importlib.util.spec_from_file_location("measure_index_behaviour", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_statement_is_a_fixed_literal_with_bound_parameters(mib: Any) -> None:
    statements = [
        mib.FORM_BARE, mib.FORM_BARE_JOINED, mib.FORM_PREDICATE, mib.FORM_PREDICATE_JOINED,
        mib.CHUNK_EXPORT_SQL, mib.CHUNK_COUNT_SQL, mib.PROBE_DROP_SQL, mib.PROBE_CREATE_SQL, mib.PROBE_INSERT_SQL,
        mib.PROBE_DELETE3_SQL, mib.PROBE_KNN_SQL, mib.PROBE_SCAN_SQL, mib.PROBE_COUNT_SQL,
        mib.PROBE_FLUSH_SQL, mib.LOCK_WAIT_SQL, mib.SERVER_STATE_SQL, *mib.ISOLATION_LEVELS.values(),
        mib.BUILD_DROP_SQL, mib.BUILD_INSERT_SQL, mib.BUILD_KNN_SQL, mib.CHUNK_PAGE_EXPORT_SQL,
        *mib.BUILD_CREATE_SQL.values(),
    ]
    for statement in statements:
        assert "{" not in statement and "%(" not in statement, statement
    # The word limit, the vector and k are placeholders, never text.
    assert mib.FORM_PREDICATE.count("%s") == 3 and mib.FORM_PREDICATE_JOINED.count("%s") == 3
    assert mib.FORM_BARE.count("%s") == 2 and mib.PROBE_KNN_SQL.count("%s") == 2
    # The probe table is the chunk table's vector column and index, nothing else.
    assert "VECTOR(384) NOT NULL" in mib.PROBE_CREATE_SQL
    assert "VECTOR INDEX (embedding) M=16 DISTANCE=cosine" in mib.PROBE_CREATE_SQL
    # The scan is the same statement without the index; the levels are the four of the server.
    assert mib.PROBE_SCAN_SQL == mib.PROBE_KNN_SQL.replace(
        "FROM isolation_probe ", "FROM isolation_probe IGNORE INDEX (embedding) "
    )
    assert list(mib.ISOLATION_LEVELS) == [
        "READ UNCOMMITTED", "READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE",
    ]
    for level, statement in mib.ISOLATION_LEVELS.items():
        assert statement == f"SET SESSION TRANSACTION ISOLATION LEVEL {level}"


def test_the_small_helpers(mib: Any) -> None:
    assert mib.new_among([1, 90001, 90005, 7]) == 2 and mib.new_among([]) == 0
    assert mib._span([5, 5, 5]) == "5" and mib._span([0, 5, 3]) == "0 to 5"
    views = [{"index_new": 0, "scan_new": 5}, {"index_new": 5, "scan_new": 5}]
    assert mib._stale_count(views) == "1 of 2"
    assert mib.median([3.0, 1.0, 2.0]) == 2.0


def test_an_unknown_part_is_refused_before_any_setting_is_loaded(
    mib: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_settings() -> Any:
        raise AssertionError("the settings must not be loaded for a bad argument")

    monkeypatch.setattr(mib, "load_settings", no_settings)
    with pytest.raises(SystemExit) as raised:
        mib.main(["forms", "everything"])
    assert raised.value.code == 2
    assert "unknown part(s) ['everything']" in capsys.readouterr().err


def _forms_report() -> dict[str, Any]:
    row = {
        "form": "predicate_30", "label": "WHERE chunk.n_words < 30", "joined": False,
        "words_below": 30, "passing_chunks": 304, "ef_search": 20, "p50_ms": 3.94, "p95_ms": 12.87,
        "equal_to_exact": 40, "same_set_as_exact": 41, "short": 0, "chunk_rows_read_p50": 276.0,
    }
    return {
        "name": "statement_forms", "created_at": "2026-11-11T00:00:00+00:00", "k": 10,
        "repeats": 3, "n_queries": 75, "n_chunks": 4598, "index_m": 16,
        "versions": {"mariadb": "11.8.9"}, "results": [row],
    }


def test_the_forms_report_shows_each_form_and_the_statements(mib: Any) -> None:
    text = mib.forms_markdown(_forms_report())
    assert text.startswith("# One nearest-neighbour statement in three shapes")
    assert "| WHERE chunk.n_words < 30 | 304 | 20 | 3.94 | 12.87 | 40/75 | 0/75 | 276 |" in text
    assert "4,598 chunks, index M = 16, the 75 corpus claims" in text
    for statement in (mib.FORM_BARE, mib.FORM_PREDICATE_JOINED):
        assert statement + ";" in text


def test_the_two_rewrites_of_the_inline_statement_change_one_thing_each(mib: Any) -> None:
    import numpy as np

    from wikilense import search as searchmod

    filters = searchmod.Filters(max_words=500, category="Rivers", linked_from="Asia")
    sql, params = searchmod.search_statement(np.ones(384, dtype=np.float32), 10, filters, "inline")
    free = mib.free_order(sql)
    # Four joins had their order fixed (page, section, the link and the category derived table);
    # the joins inside the derived tables were plain already, and nothing else changed.
    assert sql.count("STRAIGHT_JOIN") == 4 and "STRAIGHT_JOIN" not in free
    assert free.count("JOIN") == sql.count("JOIN")
    assert free.split("JOIN") == sql.replace("STRAIGHT_JOIN", "JOIN").split("JOIN")
    scan = mib.scan_form(sql)
    assert scan == sql.replace("FROM chunk STRAIGHT_JOIN", "FROM chunk IGNORE INDEX (embedding) STRAIGHT_JOIN")
    assert scan.count("%s") == sql.count("%s") == len(params)
    # The distance the report reads is the seventh selected column.
    before_distance = sql[: sql.index("VEC_DISTANCE_COSINE(chunk.embedding, %s) AS distance")]
    assert before_distance.count(",") == mib.DISTANCE_COLUMN
    with pytest.raises(ValueError, match="join order"):
        mib.free_order("SELECT 1")
    with pytest.raises(ValueError, match="FROM chunk"):
        mib.scan_form("SELECT 1")


def test_the_plan_label_the_rows_read_and_the_agreement_with_the_scan(mib: Any) -> None:
    explain = [
        {"table": "page", "type": "range", "key": "ix_page_n_words"},
        {"table": "chunk", "type": "ref", "key": "uq_chunk_page_ordinal"},
    ]
    assert mib.plan_label(explain) == "page first; chunk ref uq_chunk_page_ordinal"
    assert mib.plan_label([{"table": "chunk", "type": "ALL", "key": None}]) == "chunk first; chunk ALL no key"
    analyzed = {"query_block": {"filesort": {"temporary_table": {"nested_loop": [
        {"table": {"table_name": "page", "r_loops": 1, "r_rows": 6}},
        {"table": {"table_name": "chunk", "r_loops": 6, "r_rows": 4.5}},
        {"table": {"table_name": "section", "r_loops": 27, "r_rows": 1}},
    ]}}}}
    assert mib.chunk_rows_read(analyzed) == 27.0
    assert mib.chunk_rows_read({"query_block": {}}) == 0.0

    scan = [[(1, 0.10), (2, 0.20), (3, 0.30)], [(4, 0.10), (5, 0.20)], [(7, 0.1), (8, 0.2)]]
    same = mib.agreement(scan, scan)
    assert same == {"same_rows_as_scan": 3, "same_order_as_scan": 3, "short": 0, "swap_gap_max": 0.0}
    got = [[(1, 0.10), (3, 0.30), (2, 0.20)], [(4, 0.10)], [(7, 0.1), (9, 0.3)]]
    other = mib.agreement(got, scan)
    assert other["same_rows_as_scan"] == 1 and other["same_order_as_scan"] == 0
    assert other["short"] == 1
    assert other["swap_gap_max"] == pytest.approx(0.10)


def _join_order_report() -> dict[str, Any]:
    def side(plan: str, rows: float, ms: float, order: int = 75) -> dict[str, Any]:
        return {"plan": plan, "plan_count": 75, "plans": {plan: 75}, "chunk_rows_read_p50": rows,
                "chunk_rows_read_max": rows, "p50_ms": ms, "p95_ms": ms * 2,
                "same_rows_as_scan": 75, "same_order_as_scan": order, "short": 0,
                "swap_gap_max": 3.9e-06 if order < 75 else 0.0}

    walk, scan = "chunk first; chunk index embedding", "chunk first; chunk ALL no key"
    page = "page first; chunk ref uq_chunk_page_ordinal"
    return {
        "name": "join_order", "created_at": "2026-11-11T00:00:00+00:00", "k": 10, "repeats": 3,
        "ef_search": 100, "n_queries": 75, "n_chunks": 4598, "index_m": 16,
        "versions": {"mariadb": "11.8.9"},
        "results": [
            {"filter": "none", "passing_chunks": 4598,
             "as_written": side(walk, 10, 2.0), "free": side(walk, 10, 2.1)},
            {"filter": "max_words_500", "passing_chunks": 25,
             "as_written": side(scan, 4598, 4.8), "free": side(page, 25, 0.8)},
            {"filter": "min_words_5000", "passing_chunks": 3891,
             "as_written": side(walk, 17, 2.7, order=73), "free": side(page, 3891, 10.6)},
            {"filter": "edited_before", "passing_chunks": 778,
             "as_written": side(walk, 67, 2.75), "free": side(page, 778, 2.73)},
        ],
        "filters": {"none": {}, "max_words_500": {"max_words": 500}},
    }


def test_the_join_order_report_sorts_the_filters_by_what_the_free_plan_did(mib: Any) -> None:
    text = mib.join_order_markdown(_join_order_report())
    assert text.startswith("# The filtered statement with its join order fixed")
    assert ("| max_words_500 | 25 | chunk first; chunk ALL no key | 4598 | 4.80 | 75 / 75 | "
            "page first; chunk ref uq_chunk_page_ordinal | 25 | 0.80 | 75 / 75 |") in text
    assert "Under 3 of the 4 filters the server walks the vector index; under the other 1" in text
    assert "starts from another table under 3 of the 4 filters" in text
    assert "as written to free: `max_words_500` 4.80 to 0.80 ms." in text
    assert "got slower by more than that: `min_words_5000` 2.70 to 10.60 ms." in text
    assert "stayed within that factor: `edited_before` 2.75 to 2.73 ms." in text
    assert "(`min_words_5000` as written: 2 of 75 claims, largest gap 3.9e-06)" in text
    assert '- `max_words_500`: {"max_words": 500}' in text

    report = _join_order_report()
    report["results"][2]["as_written"]["same_order_as_scan"] = 75
    assert "No hit list had its chunks in another order than the scan." in mib.join_order_markdown(report)


def test_the_build_tables_differ_in_m_only_and_copy_the_chunk_index(mib: Any) -> None:
    assert list(mib.BUILD_CREATE_SQL) == [16, 6]
    for m, statement in mib.BUILD_CREATE_SQL.items():
        assert f"VECTOR INDEX (embedding) M={m} DISTANCE=cosine" in statement
        assert "VECTOR(384) NOT NULL" in statement
    assert mib.BUILD_CREATE_SQL[16].replace("M=16", "M=6") == mib.BUILD_CREATE_SQL[6]
    assert mib.BUILD_INSERT_SQL.count("%s") == 2 and mib.BUILD_KNN_SQL.count("%s") == 2
    assert mib.PARTS == ("forms", "join_order", "builds", "isolation")


def test_the_spread_and_the_agreement_between_builds(mib: Any) -> None:
    assert mib.spread([74, 73, 74, 70]) == {"min": 70.0, "median": 73.5, "max": 74.0}
    first = [[1, 2, 3], [4, 5, 6]]
    second = [[1, 2, 3], [4, 6, 5]]
    third = [[1, 2, 3], [4, 5, 6]]
    # pairs in order: (first, second), (first, third), (second, third)
    assert mib.identical_between_builds([first, second, third]) == [1, 2, 1]
    assert mib.identical_between_builds([first]) == []


def test_the_builds_report_gives_one_number_where_every_build_agreed(mib: Any) -> None:
    def cell(low: float, mid: float, high: float) -> dict[str, float]:
        return {"min": low, "median": mid, "max": high}

    report = {
        "name": "build_variance", "created_at": "2026-11-11T00:00:00+00:00", "k": 10,
        "n_queries": 75, "n_chunks": 4598, "builds": 10, "batch": 500, "exact_article_recall": 74,
        "versions": {"mariadb": "11.8.9"},
        "results": [
            {"m": 16, "ef_search": 100, "builds": 10, "article_recall": cell(74, 74, 74),
             "equal_to_exact": cell(74, 75, 75), "overlap_with_exact": cell(0.9987, 1.0, 1.0),
             "identical_between_two_builds": cell(74, 75, 75), "build_seconds": cell(1.42, 1.5, 2.31)},
            {"m": 6, "ef_search": 20, "builds": 10, "article_recall": cell(65, 69, 70),
             "equal_to_exact": cell(38, 42, 49), "overlap_with_exact": cell(0.775, 0.8, 0.869),
             "identical_between_two_builds": cell(26, 34, 44), "build_seconds": cell(0.41, 0.45, 0.52)},
        ],
    }
    text = mib.builds_markdown(report)
    assert text.startswith("# The same vectors, built into the HNSW graph several times")
    assert "NumPy's exact ranking of the same vectors gives 74 of 75" in text
    assert ("| 16 | 100 | 74 | 74 to 75 (median 75) | 0.999 to 1.000 | 74 to 75 (median 75) "
            "| 1.4 to 2.3 |") in text
    assert ("| 6 | 20 | 65 to 70 (median 69) | 38 to 49 (median 42) | 0.775 to 0.869 "
            "| 26 to 44 (median 34) | 0.4 to 0.5 |") in text
    for statement in (*mib.BUILD_CREATE_SQL.values(), mib.BUILD_KNN_SQL):
        assert statement + ";" in text


def _isolation_report(flush_skipped: str | None = None) -> dict[str, Any]:
    def moments(index: list[int], scan: list[int]) -> dict[str, Any]:
        return {"index": index, "scan": scan}

    stale = {"index_rows": 10, "index_new": 0, "scan_rows": 10, "scan_new": 5}
    fine = {"index_rows": 10, "index_new": 5, "scan_rows": 10, "scan_new": 5}
    return {
        "name": "isolation", "created_at": "2026-11-11T00:00:00+00:00", "k": 10, "rows": 1000,
        "new_rows": 5, "ef_search": 100, "index": "VECTOR INDEX (embedding) M=16 DISTANCE=cosine",
        "seed": 7, "trials": 2, "control_trials": 2, "mariadb_version": "11.8.9",
        "default_isolation_level": "REPEATABLE-READ", "innodb_snapshot_isolation": 1,
        "levels": [
            {"level": "REPEATABLE READ", "writer_blocked": [None, None],
             "writer_uncommitted": moments([0, 0], [0, 0]),
             "writer_committed_reader_open": moments([0, 0], [0, 0]),
             "reader_next_transaction": moments([0, 5], [5, 5])},
            {"level": "SERIALIZABLE", "writer_blocked": ["Lock wait timeout exceeded (error 1205)"] * 2,
             "writer_uncommitted": moments([0, 0], [0, 0]),
             "writer_committed_reader_open": moments([0, 0], [0, 0]),
             "reader_next_transaction": moments([0, 0], [0, 0])},
        ],
        "sequences": {
            "repeatable_read_searches_again_and_ends": [stale, stale],
            "control_read_committed": [fine, fine],
            "control_no_second_search": [fine, fine],
            "control_autocommit": [fine, fine],
            "control_reader_stays_connected": [stale, fine],
        },
        "recovery": {"by_write": [6], "by_flush": [] if flush_skipped else [5],
                     "flush_skipped": flush_skipped,
                     "still_stale_for_new_connections": [[0, 0, 0]], "attempts": 2},
        "deletes": [{
            "reader_snapshot_rows": 10, "reader_snapshot_still_sees_deleted": 3,
            "new_connection_index_rows": 0, "new_connection_scan_rows": 10,
            "other_queries_min_rows": 6, "other_queries_with_no_row": 0, "table_rows": 997,
        }],
        "delete_rows_after_flush": None if flush_skipped else 10,
    }


def test_the_isolation_report_states_what_was_seen_and_what_cleared_it(mib: Any) -> None:
    text = mib.isolation_markdown(_isolation_report())
    assert "| REPEATABLE READ | 0 / 0 | 0 / 0 | 0 to 5 / 5 | went through |" in text
    assert "| SERIALIZABLE | 0 / 0 | 0 / 0 | 0 / 0 | blocked: Lock wait timeout exceeded (error 1205) |" in text
    assert "| as above | 2 of 2 | [0, 0] |" in text
    assert "| control: A is at READ COMMITTED | 0 of 2 | [5, 5] |" in text
    assert "| control: A is in autocommit mode, so step 3 is a new transaction | 0 of 2 | [5, 5] |" in text
    assert "| control: A commits but stays connected | 1 of 2 | [0, 5] |" in text
    assert "a new connection found [6] of 6 new rows through the index" in text
    assert "`FLUSH TABLES isolation_probe` (as root): afterwards a new connection found [5] of 5" in text
    assert "| 1 | 0 | 10 | 0 (6) | 997 |" in text
    assert "After `FLUSH TABLES` the index returned 10 rows for that query again." in text

    without_docker = mib.isolation_markdown(_isolation_report("docker is not on PATH"))
    assert "`FLUSH TABLES` was not tried: docker is not on PATH." in without_docker
    assert "After `FLUSH TABLES`" not in without_docker
