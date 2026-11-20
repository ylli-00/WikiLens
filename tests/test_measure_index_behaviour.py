"""Tests for scripts/measure_index_behaviour.py: the parts that need no server and no model."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pymysql
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
        mib.PROBE_FLUSH_SQL, mib.PROBE_EF_SQL, mib.LOCK_WAIT_SQL, mib.SERVER_STATE_SQL, *mib.ISOLATION_LEVELS.values(),
        mib.BUILD_DROP_SQL, mib.BUILD_INSERT_SQL, mib.BUILD_KNN_SQL, mib.CHUNK_PAGE_EXPORT_SQL,
        *mib.BUILD_CREATE_SQL.values(),
        mib.CACHE_GET_SQL, mib.CHUNK_FLUSH_SQL, *(mib.cache_set_sql(size) for size in mib.CACHE_SIZES_MB),
        mib.HARD_CAP_THRESHOLD_SQL, mib.HARD_CAP_SUBQUERY_SQL, mib.HARD_CAP_SCAN_SQL,
        mib.HARD_CAP_TIME_GET_SQL, mib.HARD_CAP_TIME_SET_SQL,
    ]
    for statement in statements:
        assert "{" not in statement and "%(" not in statement, statement
    # The word limit, the vector and k are placeholders, never text.
    assert mib.FORM_PREDICATE.count("%s") == 3 and mib.FORM_PREDICATE_JOINED.count("%s") == 3
    assert mib.FORM_BARE.count("%s") == 2 and mib.PROBE_KNN_SQL.count("%s") == 2
    # hard_cap: vector, vector, threshold, vector, k / vector, vector, inner limit, threshold, k
    assert mib.HARD_CAP_THRESHOLD_SQL.count("%s") == 5 and mib.HARD_CAP_SUBQUERY_SQL.count("%s") == 5
    assert mib.HARD_CAP_SCAN_SQL.count("%s") == 3
    # hard_cap's time limit: the seconds are a parameter, the variable is the session's
    assert mib.HARD_CAP_TIME_SET_SQL == "SET SESSION max_statement_time = %s"
    assert mib.HARD_CAP_TIME_GET_SQL == "SELECT @@SESSION.max_statement_time"
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
    partial = [{"index_new": 4, "scan_new": 5}, {"index_new": 5, "scan_new": 5}]
    assert mib._stale_count(partial) == "0 of 2"  # an approximate miss is not the stale state
    assert mib._partial_count({"a": views, "b": partial}) == 1
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
        "chunk_access": "index embedding",
    }
    bare = {**row, "form": "bare", "label": "the bare statement on chunk", "words_below": None,
            "passing_chunks": 4598, "equal_to_exact": 75, "chunk_rows_read_p50": 10.0}
    joined = {**row, "form": "predicate_30_joined", "joined": True, "equal_to_exact": 75,
              "label": "WHERE chunk.n_words < 30, joined to page"}
    return {
        "name": "statement_forms", "created_at": "2026-11-11T00:00:00+00:00", "k": 10,
        "repeats": 3, "n_queries": 75, "n_chunks": 4598, "index_m": 16,
        "versions": {"mariadb": "11.8.9"}, "results": [bare, row, joined],
    }


def test_the_forms_report_shows_each_form_and_the_statements(mib: Any) -> None:
    text = mib.forms_markdown(_forms_report())
    assert text.startswith("# One nearest-neighbour statement in three shapes")
    assert ("| WHERE chunk.n_words < 30 | 304 | 20 | 3.94 | 12.87 | 40/75 | 0/75 | 276 "
            "| index embedding |") in text
    # the three summary sentences are computed from the rows
    assert "its list was the exact one for 75 at `mhnsw_ef_search` 20 of 75 claims" in text
    assert "0 of the 75 answers were short (with 304 chunks passing" in text
    assert "The list was the exact one for 40 at `mhnsw_ef_search` 20 of 75 claims" in text
    assert "With a join in the statement the list was the exact one for 75 at" in text
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
    merged = mib.without_distinct(sql)
    assert "DISTINCT" not in merged and merged.count("SELECT") == sql.count("SELECT") == 3
    assert merged.replace("(SELECT ", "(SELECT DISTINCT ") == sql
    with pytest.raises(ValueError, match="no DISTINCT"):
        mib.without_distinct("SELECT 1")
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
    assert same == {"same_rows_as_scan": 3, "same_order_as_scan": 3, "short": 0,
                    "swap_gap_max": 0.0, "not_ascending": 0}
    got = [[(1, 0.10), (3, 0.30), (2, 0.20)], [(4, 0.10)], [(7, 0.1), (9, 0.3)]]
    other = mib.agreement(got, scan)
    assert other["same_rows_as_scan"] == 1 and other["same_order_as_scan"] == 0
    assert other["short"] == 1
    assert other["swap_gap_max"] == pytest.approx(0.10)
    assert other["not_ascending"] == 1  # the first list: 0.10, 0.30, 0.20


def _join_order_report() -> dict[str, Any]:
    def side(plan: str, rows: float, ms: float, order: int = 75) -> dict[str, Any]:
        return {"plan": plan, "plan_count": 75, "plans": {plan: 75}, "chunk_rows_read_p50": rows,
                "chunk_rows_read_max": rows, "p50_ms": ms, "p95_ms": ms * 2,
                "p50_ms_taking_turns": ms + 0.5,
                "same_rows_as_scan": 75, "same_order_as_scan": order, "short": 0,
                "swap_gap_max": 3.9e-06 if order < 75 else 0.0,
                "not_ascending": 75 - order}

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
        "derived_without_distinct": [
            {"filter": "linked_from", "plans": {walk: 75}, "plans_with_a_derived_table": 0,
             "lists_with_a_chunk_twice": 75},
        ],
        "filters": {"none": {}, "max_words_500": {"max_words": 500}},
    }


def test_the_join_order_report_sorts_the_filters_by_what_the_free_plan_did(mib: Any) -> None:
    text = mib.join_order_markdown(_join_order_report())
    assert text.startswith("# The filtered statement with its join order fixed")
    assert ("| max_words_500 | 25 | chunk first; chunk ALL no key | 4598 | 4.80 (5.30) | 75 / 75 | "
            "page first; chunk ref uq_chunk_page_ordinal | 25 | 0.80 (1.30) | 75 / 75 |") in text
    assert "Under 3 of the 4 filters the server walks the vector index; under the other 1" in text
    assert "starts from another table under 3 of the 4 filters" in text
    assert "as written to free: `max_words_500` 4.80 to 0.80 ms." in text
    assert "got slower by more than that: `min_words_5000` 2.70 to 10.60 ms." in text
    assert "stayed within that factor: `edited_before` 2.75 to 2.73 ms." in text
    assert ("(`min_words_5000` as written: 2 of 75 claims, 2 of them with rows not in ascending "
            "order of the distance the statement reports, largest gap 3.9e-06)") in text
    assert '- `max_words_500`: {"max_words": 500}' in text
    assert "| linked_from | chunk first; chunk index embedding (75/75) | 0/75 | 75/75 |" in text

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
    assert mib.PARTS == ("forms", "join_order", "builds", "cache", "isolation", "hard_cap")


def test_the_cache_statements_set_whole_megabytes_from_the_default_down(mib: Any) -> None:
    assert mib.CACHE_SIZES_MB == (16, 8, 4, 2, 1)
    assert mib.cache_set_sql(16) == "SET GLOBAL mhnsw_max_cache_size = 16777216"  # the default
    assert mib.cache_set_sql(1) == "SET GLOBAL mhnsw_max_cache_size = 1048576"  # the minimum
    for size in (*mib.CACHE_SIZES_MB, 192, 512):
        assert mib.cache_set_sql(size).endswith(f" = {size * 1024 * 1024}")
    # The size is written into the statement, so only a positive int is accepted.
    for bad in (0, -1, 1.5, "16", True):
        with pytest.raises(ValueError):
            mib.cache_set_sql(bad)
    assert mib.CHUNK_FLUSH_SQL == "FLUSH TABLES chunk"


def test_the_cache_report_shows_every_pass_and_the_restored_size(mib: Any) -> None:
    def row(cache: str, megabytes: float, statement: str, equal: list[int], agree: list[int],
            ms: float) -> dict[str, Any]:
        return {"cache": cache, "cache_bytes": int(megabytes * 1048576), "table_reopened": True,
                "statement": statement, "equal_to_exact": equal, "article_recall": [74, 74, 74],
                "overlap_with_exact": [0.9551, 0.9634, 0.96], "short": [0, 0, 0],
                "identical_between_two_passes": agree, "p50_ms": ms}

    report = {
        "name": "cache_size", "created_at": "2026-11-11T00:00:00+00:00", "k": 10, "passes": 3,
        "ef_search": 100, "n_queries": 75, "n_chunks": 4598, "index_m": 16,
        "configured_cache_bytes": 536870912, "restored_cache_bytes": 536870912,
        "exact_article_recall": 74, "versions": {"mariadb": "11.8.9"},
        "results": [
            row("as configured", 512, "bare", [75, 75, 75], [75, 75, 75], 0.77),
            row("1 MB", 1, "joined", [57, 59, 57], [55, 53, 52], 2.64),
            row("as configured, again", 512, "joined", [75, 75, 75], [75, 75, 75], 2.63),
        ],
    }
    text = mib.cache_markdown(report)
    assert text.startswith("# The index cache at, above and below the size of the graph")
    assert "was 536,870,912 bytes (512 MB) when the run began and is 536,870,912 bytes again" in text
    assert "| 512 MB (as configured) | bare | [75, 75, 75] | 74 | 0.955 to 0.963 | 0 | [75, 75, 75] | 0.77 |" in text
    assert "| 1 MB | joined | [57, 59, 57] | 74 | 0.955 to 0.963 | 0 | [55, 53, 52] | 2.64 |" in text
    assert "| 512 MB (as configured, after the run) | joined | [75, 75, 75] |" in text
    assert "SET GLOBAL mhnsw_max_cache_size = 8388608;" in text and "FLUSH TABLES chunk;" in text


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

    stale = {"index_rows": 10, "index_new": 0, "scan_rows": 10, "scan_new": 5,
             "reader_second_search_new": 0}
    fine = {"index_rows": 10, "index_new": 5, "scan_rows": 10, "scan_new": 5,
            "reader_second_search_new": 0}
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
            "control_first_statement_count": [fine, fine],
            "control_first_statement_scan": [fine, fine],
        },
        "victims": [
            {"REPEATABLE READ": 0, "READ COMMITTED": 0, "READ UNCOMMITTED": 0, "autocommit": 0},
            {"REPEATABLE READ": 0, "READ COMMITTED": 0, "READ UNCOMMITTED": 0, "autocommit": 1},
        ],
        "recovery": {"by_write": [6], "by_flush": [] if flush_skipped else [5],
                     "flush_skipped": flush_skipped,
                     "still_stale_for_new_connections": [[0, 0, 0]], "attempts": 2},
        "deletes": [{
            "reader_snapshot_rows": 10, "reader_snapshot_still_sees_deleted": 3,
            "new_connection_index_rows": 0, "new_connection_scan_rows": 10,
            "other_queries_min_rows": 6, "other_queries_with_no_row": 0, "table_rows": 997,
        }],
        "delete_rows_after_flush": None if flush_skipped else 10,
        "delete_rows_after_write": [10, 10],
        "delete_control_no_second_search": [10, 10],
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
    assert "3. A searches again, still inside its old snapshot (it found 0 of the new rows" in text
    assert "| control: A's statement in step 1 is `SELECT COUNT(*)`, not a search | 0 of 2 | [5, 5] |" in text
    assert "| control: A's statement in step 1 is the scan, not the index search | 0 of 2 | [5, 5] |" in text
    assert ("on 2 stale tables: REPEATABLE READ 0; READ COMMITTED 0; READ UNCOMMITTED 0; "
            "in autocommit mode 0 to 1.") in text
    assert "In its second search A still found 3 of the 3 deleted rows" in text
    assert "the index returned [10, 10] rows for that query (2 stale tables)" in text
    assert "the index as stored was right" in text

    without_docker = mib.isolation_markdown(_isolation_report("docker is not on PATH"))
    assert "`FLUSH TABLES` was not tried: docker is not on PATH." in without_docker
    assert "After `FLUSH TABLES`" not in without_docker
    # no conclusion about the stored index without the measurement behind it
    assert "the index as stored was right" not in without_docker


def test_a_claim_sample_is_a_fixed_draw_in_claim_order(mib: Any) -> None:
    claims = list(range(100, 130))
    first = mib.sample_claims(claims, 10, seed=5)
    assert first == mib.sample_claims(claims, 10, seed=5)  # the same draw every run
    assert first == sorted(first) and len(set(first)) == 10 and set(first) <= set(claims)
    assert mib.sample_claims(claims, 10, seed=6) != first
    assert mib.sample_claims(claims, None) == claims
    assert mib.sample_claims(claims, 99) == claims  # not smaller than the population: all
    with pytest.raises(ValueError):
        mib.sample_claims(claims, 0)


def test_the_builds_report_of_a_single_build_has_no_agreement_between_builds(mib: Any) -> None:
    one = {"min": 74, "median": 74, "max": 74}
    report = {
        "name": "build_variance", "created_at": "2026-11-19T00:00:00+00:00", "k": 5,
        "n_queries": 75, "n_chunks": 4598, "builds": 1, "batch": 500, "exact_article_recall": 74,
        "ms": [16], "versions": {"mariadb": "11.8.9"},
        "results": [
            {"m": 16, "ef_search": 100, "builds": 1, "article_recall": one, "equal_to_exact": one,
             "overlap_with_exact": {"min": 0.99, "median": 0.99, "max": 0.99},
             "identical_between_two_builds": None,
             "build_seconds": {"min": 1.5, "median": 1.5, "max": 1.5}},
        ],
    }
    text = mib.builds_markdown(report)
    assert "| 16 | 100 | 74 | 74 | 0.990 to 0.990 | - | 1.5 to 1.5 |" in text
    assert mib.BUILD_CREATE_SQL[16] + ";" in text and mib.BUILD_CREATE_SQL[6] + ";" not in text


def test_the_hard_cap_statements_are_the_documented_shapes(mib: Any) -> None:
    from wikilense import search as searchmod

    # The derived table is the bare index search of wikilense.search, nothing else.
    inner = mib.HARD_CAP_SUBQUERY_SQL.split("FROM (", 1)[1].split(") AS t ", 1)[0]
    assert inner == searchmod.KNN_SQL
    assert mib.HARD_CAP_SUBQUERY_SQL.endswith("WHERE t.distance < %s ORDER BY t.distance LIMIT %s")
    # The threshold statement is the same search with the threshold in its own WHERE.
    assert mib.HARD_CAP_THRESHOLD_SQL == searchmod.KNN_SQL.replace(
        "FROM chunk ", "FROM chunk WHERE VEC_DISTANCE_COSINE(embedding, %s) < %s "
    )
    # The check counts the same condition without the index, over every chunk.
    assert "IGNORE INDEX (embedding)" in mib.HARD_CAP_SCAN_SQL
    assert "LIMIT" not in mib.HARD_CAP_SCAN_SQL
    assert mib.HARD_CAP_SCAN_SQL.endswith("WHERE VEC_DISTANCE_COSINE(embedding, %s) < %s")
    assert mib.HARD_CAP_NS == (3, 10, 30) and mib.HARD_CAP_INNER_LIMITS == (10, 100)
    assert mib.HARD_CAP_EF == 100 and mib.K == 10


def test_the_exact_ranking_and_the_threshold_let_exactly_n_through(mib: Any) -> None:
    import numpy as np

    separate = mib.separate  # scripts/compare_separate_index.py, as the module imports it
    rng = np.random.default_rng(3)
    vectors = separate.unit_rows(rng.normal(size=(200, 8)))
    ids = np.arange(1000, 1200, dtype=np.int64)
    query = separate.unit_rows(rng.normal(size=(1, 8)))[0]
    distances = 1.0 - vectors @ query
    ranked_ids, ranked = mib.exact_ranking(distances, ids, 31)
    # The same order as compare_separate_index.top_k, with the distances alongside.
    assert ranked_ids == separate.top_k(vectors, query, ids, 31)
    assert ranked == sorted(ranked) and len(ranked) == 31
    for n in (3, 10, 30):
        threshold = mib.distance_threshold(ranked, n)
        assert threshold is not None
        assert int(np.sum(distances < threshold)) == n
        assert ranked[n - 1] < threshold < ranked[n]
    # Fewer rows than asked for: all of them; ties go to the smaller id.
    tied = np.asarray([0.3, 0.1, 0.1, 0.2])
    assert mib.exact_ranking(tied, np.asarray([7, 9, 4, 1]), 3) == ([4, 9, 1], [0.1, 0.1, 0.2])
    assert mib.exact_ranking(tied, np.asarray([7, 9, 4, 1]), 9)[0] == [4, 9, 1, 7]
    with pytest.raises(ValueError):
        mib.exact_ranking(tied, np.asarray([7, 9, 4, 1]), 0)


def test_a_tie_at_the_nth_distance_gives_no_threshold(mib: Any) -> None:
    nearest = [0.10, 0.20, 0.20, 0.30]
    assert mib.distance_threshold(nearest, 1) == pytest.approx(0.15)
    assert mib.distance_threshold(nearest, 2) is None  # the 2nd and the 3rd are tied
    assert mib.distance_threshold(nearest, 3) == pytest.approx(0.25)
    assert mib.distance_threshold(nearest, 4) is None  # no 5th chunk
    assert mib.distance_threshold(nearest, 0) is None
    # Closer than the margin counts as a tie; just above it does not.
    margin = mib.TIE_MARGIN
    assert mib.distance_threshold([0.1, 0.1 + margin / 2], 1) is None
    assert mib.distance_threshold([0.1, 0.1 + 3 * margin], 1) == pytest.approx(0.1 + 1.5 * margin)


def test_the_word_limits_follow_the_rule_of_scaled_filters(mib: Any) -> None:
    import numpy as np

    assert set(mib.HARD_CAP_SHARES) <= set(mib.scaled_filters.TARGETS)
    # 10,000 chunks: 3 with 1 word, 20 with 2, 40 with 3, 100 with 5, the rest with 240.
    words = np.asarray([1] * 3 + [2] * 20 + [3] * 40 + [5] * 100 + [240] * 9837)
    assert mib.word_limit_candidates(words) == [(2, 3), (3, 23), (4, 63), (6, 163), (241, 10000)]
    # Targets 50 and 5 chunks: 63 is nearest 50 on a log scale, 3 is nearest 5 (|log 3/5| is
    # 0.51, |log 23/5| is 1.53).
    assert mib.pick_word_limits(words) == [
        {"target_share": 0.005, "words_below": 4, "passing_chunks": 63, "passing_share": 0.0063},
        {"target_share": 0.0005, "words_below": 2, "passing_chunks": 3, "passing_share": 0.0003},
    ]
    # A limit that both shares pick is kept once, under the first.
    few = np.asarray([1] * 3 + [240] * 9997)
    assert [entry["words_below"] for entry in mib.pick_word_limits(few)] == [2]
    # A log-scale tie goes to the larger count: 25 and 100 are equally far from 50.
    even = np.asarray([1] * 25 + [2] * 75 + [240] * 9900)
    assert mib.pick_word_limits(even, shares=(0.005,))[0]["passing_chunks"] == 100


def test_the_chunk_access_of_a_plan(mib: Any) -> None:
    assert mib.chunk_access_of([
        {"table": "<derived2>", "type": "ALL", "key": None},
        {"table": "chunk", "type": "index", "key": "embedding"},
    ]) == "index embedding"
    assert mib.chunk_access_of([{"table": "chunk", "type": "ALL", "key": None}]) == "ALL no key"
    assert mib.chunk_access_of([{"table": "page", "type": "ALL", "key": None}]) == "no chunk row"


def _hard_cap_report() -> dict[str, Any]:
    def row(kind: str, label: str, claims: int, expected: int, returned: dict[str, int],
            complete: int, short: int, equal: int, read: float, read_max: float,
            **extra: Any) -> dict[str, Any]:
        base = {"case": label, "kind": kind, "label": label, "n": None, "inner_limit": None,
                "words_below": None, "joined": False, "passing_chunks": None, "claims": claims,
                "claims_run": claims, "expected_rows": expected, "executions": 2 * claims,
                "complete": complete, "short": short, "stopped": 0, "server_lost": 0,
                "not_run": 0, "answered": sum(returned.values()), "rows_returned": returned,
                "equal_to_exact": equal, "same_set_as_exact": equal,
                "analyze": {"complete": claims, "skipped": 0, "stopped": 0, "server lost": 0},
                "chunk_rows_read_p50": read, "chunk_rows_read_max": read_max,
                "chunk_access": "index embedding", "chunk_access_count": claims,
                "plans": {"index embedding": claims}, "timed_executions": claims,
                "p50_ms": 1.5, "p95_ms": 4.25}
        return {**base, **extra}

    return {
        "name": "hard_cap", "created_at": "2026-11-20T00:00:00+00:00", "k": 10, "ef_search": 100,
        "time_limit_s": 10, "server": {"version": "11.8.9-MariaDB", "port": 3307},
        "repeats": 1, "n_claims_total": 3000, "sample": 4, "seed": 20261119, "n_queries": 4,
        "n_chunks": 132359, "index_m": 16, "tie_margin": 1e-06,
        "finished": True, "stopped_early": None, "cases_total": 4, "cases_done": 4,
        "server_losses": 0, "reconnects": 0,
        "versions": {"mariadb": "11.8.9-MariaDB"},
        "word_limits": [
            {"target_share": 0.005, "words_below": 9, "passing_chunks": 700, "passing_share": 0.005289},
            {"target_share": 0.0005, "words_below": 3, "passing_chunks": 70, "passing_share": 0.000529},
        ],
        "threshold_check": {
            "per_n": [{"n": 3, "claims_with_threshold": 4, "left_out_ties": 0, "scan_finished": 4,
                       "server_count_equal_n": 4, "server_counts_other": []},
                      {"n": 10, "claims_with_threshold": 3, "left_out_ties": 1, "scan_finished": 3,
                       "server_count_equal_n": 3, "server_counts_other": []}],
            "scan": {"complete": 4, "stopped": 0, "server lost": 0, "not run": 0,
                     "no threshold": 0},
            "server_numpy_distance_gap_max": 2.5e-08,
        },
        "results": [
            row("threshold", "WHERE distance < threshold, n = 3", 4, 3, {"3": 4}, 8, 0, 4, 10, 10,
                n=3),
            row("threshold", "WHERE distance < threshold, n = 10", 3, 10, {"10": 2, "9": 1}, 4, 2,
                2, 10, 10, n=10),
            row("subquery", "inner LIMIT 10, n = 3", 4, 3, {"3": 4}, 8, 0, 4, 10, 10, n=3,
                inner_limit=10),
            row("predicate", "WHERE chunk.n_words < 3", 4, 10, {"10": 4}, 8, 0, 3, 1800, 4100,
                words_below=3, passing_chunks=70),
        ],
    }


def test_the_hard_cap_report_is_written_from_the_numbers(mib: Any) -> None:
    text = mib.hard_cap_markdown(_hard_cap_report())
    assert text.startswith("# The LIMIT of an index search")
    assert ("on MariaDB 11.8.9-MariaDB at port 3307: 132,359 chunks, index M = 16, a random "
            "sample of 4 of the 3,000 corpus claims as queries (`random.Random(20261119)`), k = "
            "10, `mhnsw_ef_search` 100, a time limit of 10 s on every statement "
            "(`max_statement_time`). The cases ran one after the other; for each claim the "
            "statement ran once untimed and then 1 time timed.") in text
    assert "count the executions, 2 per claim" in text
    assert ("| n | Claims | Expected | Complete | Short | Stopped | Server lost | Rows returned "
            "| Equal to exact | Same set | Rows read (largest) | Chunk access | Timed | p50 ms "
            "| p95 ms |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n") in text
    assert ("| 10 | 3 | 10 | 4 | 2 | 0 | 0 | 10 (2), 9 (1) | 2/3 | 2/3 | 10 (10) "
            "| index embedding | 3 | 1.50 | 4.25 |") in text
    assert ("| 10 | 3 | 4 | 3 | 8 | 0 | 0 | 0 | 3 (4) | 4/4 | 4/4 | 10 (10) | index embedding "
            "| 4 |") in text
    assert ("| WHERE chunk.n_words < 3 | 70 | 4 | 10 | 8 | 0 | 0 | 0 | 10 (4) | 3/4 | 3/4 "
            "| 1800 (4100) | index embedding | 4 |") in text
    # The measured threshold rows never read more than LIMIT rows; the predicate row did. The
    # text states only that, for the rows measured, and draws no conclusion beyond them.
    assert ("Of the claims with their rows read measured, none read more than `LIMIT` = 10 chunk "
            "rows (") in text
    assert "the threshold was applied" not in text
    assert "Short answers, of the executions: 0/8 at n = 3, 2/6 at n = 10" in text
    assert "equal to the exact list, of the claims with rows: 4/4 at n = 3, 2/3 at n = 10." in text
    assert "`EXPLAIN` gives `chunk` `index embedding` for every claim" in text
    assert "0 of the 8 executions came back short" in text
    assert "The server read more than `LIMIT` = 10 chunk rows for WHERE chunk.n_words < 3." in text
    assert "9 words: 700 chunks (0.53%, aimed at 0.50%)" in text
    assert ("found exactly n chunks under the threshold for 4 of 4 claims at n = 3, 3 of 3 claims "
            "at n = 10.") in text
    assert "at most 1e-06 apart: 0 at n = 3, 1 at n = 10" in text
    assert "was 2.5e-08." in text
    # Nothing was stopped or lost: one line each, and no early stop.
    assert "- No statement reached the time limit of 10 s." in text
    assert "- The connection to the server was never lost." in text
    assert "stopped early" not in text and "Executions not run" not in text
    assert "The scan did not finish" not in text
    assert "SET SESSION max_statement_time = 10;" in text
    for statement in (mib.HARD_CAP_THRESHOLD_SQL, mib.HARD_CAP_SUBQUERY_SQL, mib.FORM_PREDICATE,
                      mib.FORM_PREDICATE_JOINED, mib.HARD_CAP_SCAN_SQL):
        assert statement + ";" in text

    report = _hard_cap_report()
    report["results"][1]["chunk_rows_read_max"] = 4100
    report["results"][1]["chunk_access_count"] = 2
    text = mib.hard_cap_markdown(report)
    assert ("The server read more than `LIMIT` = 10 chunk rows from the index at n = 10: it went "
            "on taking rows") in text
    assert "| index embedding (2/3) |" in text


def test_the_hard_cap_report_counts_the_stopped_and_the_lost_executions(mib: Any) -> None:
    report = _hard_cap_report()
    # n = 10: one claim complete twice, one stopped then short, one lost on its first execution
    report["results"][1].update(
        complete=2, short=1, stopped=1, server_lost=1, not_run=1, answered=2,
        rows_returned={"10": 1, "9": 1}, equal_to_exact=1, same_set_as_exact=1,
        analyze={"complete": 1, "skipped": 2, "stopped": 0, "server lost": 0},
        chunk_rows_read_p50=10, chunk_rows_read_max=10, timed_executions=2,
    )
    report["results"][3]["analyze"] = {"complete": 3, "skipped": 0, "stopped": 1,
                                       "server lost": 0}
    report["threshold_check"]["scan"].update(complete=3, stopped=1)
    report["threshold_check"]["per_n"][0]["scan_finished"] = 3
    report["threshold_check"]["per_n"][0]["server_count_equal_n"] = 3
    report.update(server_losses=2, reconnects=2, finished=False, cases_done=4, cases_total=10,
                  stopped_early="the server was lost 5 times in a row")
    text = mib.hard_cap_markdown(report)
    assert ("| 10 | 3 | 10 | 2 | 1 | 1 | 1 | 10 (1), 9 (1) | 1/2 | 1/2 | 10 (10) "
            "| index embedding | 2 | 1.50 | 4.25 |") in text
    assert ("The part stopped early: the server was lost 5 times in a row. 4 of the 10 cases "
            "were reached; the tables hold what was measured until then.") in text
    # 8 + 5 + 8 + 8 executions ran
    assert ("- Stopped at the time limit of 10 s: 1 of the 29 executions (1 of 5 for WHERE "
            "distance < threshold, n = 10), `ANALYZE` for 1 claim and the scan check for "
            "1 claim.") in text
    assert ("- Server lost: 1 of the 29 executions (1 of 5 for WHERE distance < threshold, "
            "n = 10). With `EXPLAIN`, `ANALYZE` and the scan check, the connection was lost "
            "2 times and the part reconnected 2 times;") in text
    assert ("- Executions not run: 1, the rest of a claim after a lost server and the claims "
            "after the part stopped.") in text
    assert "Short answers, of the executions: 0/8 at n = 3, 1/5 at n = 10" in text
    assert ("for 3 of 3 claims at n = 3, 3 of 3 claims at n = 10. The scan did not finish for "
            "1 claim (1 stopped at the time limit, 0 with the server lost, 0 not run); their "
            "thresholds were used all the same.") in text
    assert "No statement reached" not in text and "never lost" not in text


def test_hard_cap_is_refused_without_the_throwaway_flag(
    mib: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_connection(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may connect without --throwaway-server")

    monkeypatch.setattr(mib, "load_settings", no_connection)
    monkeypatch.setattr(mib.db, "connect", no_connection)
    monkeypatch.setattr(mib, "measure_hard_cap", no_connection)
    for argv in (["hard_cap"], ["forms", "hard_cap"], ["hard_cap", "--sample", "5"]):
        with pytest.raises(SystemExit) as raised:
            mib.main(argv)
        assert raised.value.code == 2
        err = capsys.readouterr().err
        assert "hard_cap runs only with --throwaway-server" in err
        assert "out of memory" in err and "--memory cap" in err
        assert "never against the project's database server" in err


def test_every_part_but_hard_cap_runs_by_default(mib: Any) -> None:
    assert mib.selected_parts([], False) == ["forms", "join_order", "builds", "cache", "isolation"]
    assert mib.selected_parts([], True) == list(mib.PARTS)
    assert mib.selected_parts(["hard_cap", "forms"], True) == ["hard_cap", "forms"]
    # A named part is kept as named: main refuses hard_cap without the flag.
    assert mib.selected_parts(["hard_cap"], False) == ["hard_cap"]


def test_an_execution_is_complete_short_stopped_or_lost(mib: Any) -> None:
    err = pymysql.err
    stopped = err.OperationalError(1969, "Query execution was interrupted (max_statement_time "
                                         "exceeded)")
    assert mib.error_outcome(stopped) == "stopped"
    assert mib.error_outcome(err.OperationalError(2013, "Lost connection to server")) == (
        "server lost"
    )
    assert mib.error_outcome(err.OperationalError(2006, "Server has gone away")) == "server lost"
    assert mib.error_outcome(err.OperationalError(2055, "Lost connection")) == "server lost"
    assert mib.error_outcome(err.InterfaceError(0, "")) == "server lost"  # a closed connection
    # Anything else is not an outcome: the caller raises it.
    assert mib.error_outcome(err.ProgrammingError(1064, "syntax")) is None
    assert mib.error_outcome(err.OperationalError(1045, "Access denied")) is None
    assert mib.error_outcome(ValueError("no")) is None
    # A result: the expected number of rows, or fewer.
    assert mib.rows_outcome(10, 10) == "complete"
    assert mib.rows_outcome(9, 10) == "short" and mib.rows_outcome(0, 3) == "short"
    assert mib.OUTCOMES == ("complete", "short", "stopped", "server lost")
    assert mib.HARD_CAP_STATEMENT_SECONDS == 10 and mib.HARD_CAP_LOST_IN_A_ROW == 5
    assert mib.HARD_CAP_RECONNECT_SECONDS == 180 and mib.HARD_CAP_RECONNECT_INTERVAL == 2


class _FakeCursor:
    def __init__(self, conn: _FakeConnection) -> None:
        self.conn = conn
        self.rows: list[Any] = []

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.conn.statements.append((sql, params))
        self.rows = {"SELECT @@SESSION.max_statement_time": [(0.0,)],
                     "SELECT @@SESSION.mhnsw_ef_search": [(20,)]}.get(sql, [])

    def fetchone(self) -> Any:
        return self.rows[0]


class _FakeConnection:
    """Answers the session statements of GuardedConnection; nothing else."""

    def __init__(self) -> None:
        self.open = True
        self.statements: list[tuple[str, Any]] = []

    def cursor(self, *args: Any) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.open = False


def _lose(conn: Any) -> None:
    conn.open = False  # as PyMySQL closes a connection whose server went away
    raise pymysql.err.OperationalError(2013, "Lost connection to server")


def _stop(conn: Any) -> None:
    raise pymysql.err.OperationalError(1969, "max_statement_time exceeded")


def _ok(conn: Any) -> int:
    return 1


def test_the_part_stops_after_five_lost_servers_in_a_row(mib: Any) -> None:
    connections: list[_FakeConnection] = []

    def connect(settings: Any) -> _FakeConnection:
        connections.append(_FakeConnection())
        return connections[-1]

    guard = mib.GuardedConnection(None, connect=connect, sleep=lambda s: None, clock=lambda: 0.0)
    guard.start()
    set_up = [("SET SESSION max_statement_time = %s", (10,)),
              ("SET SESSION mhnsw_ef_search = %s", (100,))]
    assert connections[0].statements == [("SELECT @@SESSION.max_statement_time", None),
                                          ("SELECT @@SESSION.mhnsw_ef_search", None), *set_up]
    for _ in range(4):
        assert guard.attempt(_lose) == ("server lost", None)
    # Each lost server: the connection closed, a new one with the same session settings.
    assert len(connections) == 5 and guard.conn is connections[-1]
    assert all(conn.statements == set_up for conn in connections[1:])
    assert not any(conn.open for conn in connections[:-1])
    assert guard.attempt(_ok) == (None, 1)  # a statement that came back ends the run
    for _ in range(4):
        guard.attempt(_lose)
    assert guard.attempt(_stop) == ("stopped", None)  # stopped: the server is still there
    for _ in range(4):
        guard.attempt(_lose)
    # EXPLAIN does not run the statement: it does not end the run of lost servers.
    assert guard.attempt(_ok, executes=False) == (None, 1)
    with pytest.raises(mib.HardCapStop, match="the server was lost 5 times in a row"):
        guard.attempt(_lose)
    assert guard.losses == 13 and guard.reconnects == 12
    guard.close()  # the last connection is closed: nothing to write back
    assert connections[-1].statements == set_up


def test_the_part_stops_when_the_server_does_not_come_back(mib: Any) -> None:
    now = [0.0]
    sleeps: list[float] = []
    tries: list[float] = []

    def connect(settings: Any) -> _FakeConnection:
        tries.append(now[0])
        if len(tries) > 1:
            raise pymysql.err.OperationalError(2003, "Can't connect to server")
        return _FakeConnection()

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    guard = mib.GuardedConnection(None, connect=connect, sleep=sleep, clock=lambda: now[0])
    with pytest.raises(mib.HardCapStop, match="did not accept a connection within 180 s"):
        guard.attempt(_lose)
    # One try at once, then one every 2 s up to 180 s.
    assert set(sleeps) == {2.0} and tries[1:] == [2.0 * i for i in range(91)]


def test_closing_the_part_writes_the_session_values_back(mib: Any) -> None:
    conn = _FakeConnection()
    guard = mib.GuardedConnection(None, connect=lambda settings: conn)
    guard.start()
    guard.close()
    assert conn.statements[-2:] == [("SET SESSION max_statement_time = %s", (0.0,)),
                                    ("SET SESSION mhnsw_ef_search = %s", (20,))]
    assert not conn.open


class _ScriptedGuard:
    """Answers run_claims' statements from a script: per claim, one entry per execution, the
    number of rows, 'stopped', 'server lost' or 'end' (the part stops)."""

    def __init__(self, mib: Any, script: dict[int, list[Any]]) -> None:
        self.mib = mib
        self.script = script
        self.calls: list[tuple[str, int]] = []

    def attempt(self, work: Any, sql: str, params: Any, executes: bool = True) -> Any:
        claim = params[0]
        self.calls.append((work.__name__, claim))
        if work is self.mib.db.explain:
            return None, [{"table": "chunk", "type": "index", "key": "embedding", "Extra": ""}]
        if work is not self.mib._fetch:
            return None, 12.0  # rows read
        step = self.script[claim].pop(0)
        if step == "end":
            raise self.mib.HardCapStop("the server was lost 5 times in a row")
        if isinstance(step, str):
            return step, None
        return None, [(chunk,) for chunk in range(step)]


def test_the_claims_of_a_case_record_every_execution(mib: Any) -> None:
    guard = _ScriptedGuard(mib, {0: [10, 10], 2: ["server lost"], 3: ["stopped", 9]})
    records: list[dict[str, Any]] = []
    mib.run_claims(guard, "SQL", [(0,), None, (2,), (3,)], 10, 1, records)
    assert [record["claim"] for record in records] == [0, 2, 3]  # claim 1 has no threshold
    first, lost, stopped = records
    assert first["outcomes"] == ["complete", "complete"] and len(first["ms"]) == 1
    assert first["rows"] == list(range(10)) and first["analyze"] == "complete"
    assert first["rows_read"] == 12.0 and first["access"] == "index embedding"
    # A lost server ends the claim; EXPLAIN still runs, ANALYZE does not.
    assert lost["outcomes"] == ["server lost"] and lost["ms"] == [] and lost["rows"] is None
    assert lost["analyze"] == "skipped" and lost["access"] == "index embedding"
    # Stopped, then short: only the short timed execution has a time.
    assert stopped["outcomes"] == ["stopped", "short"] and len(stopped["ms"]) == 1
    assert stopped["rows"] == list(range(9)) and stopped["analyze"] == "skipped"
    assert ("_rows_read_of", 2) not in guard.calls and ("_rows_read_of", 3) not in guard.calls

    case = {"key": "threshold_10", "kind": "threshold", "label": "n = 10", "n": 10,
            "inner_limit": None, "words_below": None, "joined": False, "claims": 3,
            "expected": 10}
    row = mib.summarise_hard_cap_case(case, records, lambda claim: list(range(10)), 1)
    assert (row["complete"], row["short"], row["stopped"], row["server_lost"]) == (2, 1, 1, 1)
    assert row["executions"] == 6 and row["not_run"] == 1 and row["answered"] == 2
    assert row["rows_returned"] == {"10": 1, "9": 1} and row["equal_to_exact"] == 1
    assert row["analyze"] == {"complete": 1, "skipped": 2, "stopped": 0, "server lost": 0}
    assert row["timed_executions"] == 2 and row["chunk_rows_read_max"] == 12.0
    assert row["chunk_access_count"] == 3

    # A stop of the part keeps the claims run so far.
    guard = _ScriptedGuard(mib, {0: [10, 10], 1: ["end"]})
    records = []
    with pytest.raises(mib.HardCapStop):
        mib.run_claims(guard, "SQL", [(0,), (1,)], 10, 1, records)
    assert [record["claim"] for record in records] == [0, 1]
    assert records[0]["analyze"] == "complete" and records[1]["outcomes"] == []
