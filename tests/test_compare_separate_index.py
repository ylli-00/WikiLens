"""Tests for scripts/compare_separate_index.py: the parts that need no server and no model."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pytest

from wikilense.config import REPO_ROOT
from wikilense.search import Filters

SCRIPT = REPO_ROOT / "scripts" / "compare_separate_index.py"


@pytest.fixture(scope="module")
def compare() -> Any:
    spec = importlib.util.spec_from_file_location("compare_separate_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


def _unit(rows: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _utc_naive(year: int, month: int, day: int) -> datetime:
    """A naive UTC datetime, as PyMySQL returns ``page.last_edited``."""
    return datetime(year, month, day, tzinfo=timezone.utc).replace(tzinfo=None)


def _copy(compare: Any) -> Any:
    """Six chunks on three pages; page 1 is short, page 3 has no edit date."""
    return compare.Copy(
        chunk_ids=np.asarray([10, 11, 12, 13, 14, 15]),
        page_ids=np.asarray([1, 1, 2, 2, 3, 3]),
        headings=["", "History", "Early history", "Geography", "HISTORY of X", "Economy"],
        vectors=_unit([[1, 0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0, 1], [-1, 0]]),
        page_words={1: 500, 2: 2000, 3: 3000},
        page_edited={1: _utc_naive(2020, 12, 5), 2: _utc_naive(2020, 12, 2), 3: None},
        category_pages={"States": {1, 2, 3}, "Rivers": {1}},
        linked_from={"Asia": {2, 3}, "Aare": {1}},
    )


def test_top_k_is_the_exact_cosine_ranking(compare: Any) -> None:
    vectors = _unit([[1, 0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0, 1], [-1, 0]])
    ids = np.asarray([10, 11, 12, 13, 14, 15])
    query = _unit([[1, 0]])[0]

    assert compare.top_k(vectors, query, ids, 3) == [10, 11, 12]
    assert compare.top_k(vectors, query, ids, 6) == [10, 11, 12, 13, 14, 15]
    assert compare.top_k(vectors, query, ids, 99) == [10, 11, 12, 13, 14, 15]  # fewer rows than k
    # A brute-force check on random data: the same ids as a full sort by distance.
    rng = np.random.default_rng(0)
    random_vectors = _unit(rng.normal(size=(200, 8)).tolist())
    random_ids = np.arange(1000, 1200)
    for _ in range(5):
        random_query = _unit([rng.normal(size=8).tolist()])[0]
        expected = random_ids[np.argsort(1.0 - random_vectors @ random_query, kind="stable")][:10]
        assert compare.top_k(random_vectors, random_query, random_ids, 10) == list(expected)


def test_top_k_breaks_ties_by_the_smaller_id_and_honours_a_pre_filter(compare: Any) -> None:
    vectors = _unit([[1, 0], [1, 0], [1, 0], [0, 1]])
    ids = np.asarray([7, 3, 5, 1])
    query = _unit([[1, 0]])[0]

    assert compare.top_k(vectors, query, ids, 2) == [3, 5]  # three tied rows: the two smallest ids
    assert compare.top_k(vectors, query, ids, 2, rows=np.asarray([0, 3])) == [7, 1]
    assert compare.top_k(vectors, query, ids, 2, rows=np.asarray([], dtype=int)) == []


def test_unit_rows_scales_rows_and_leaves_zero_rows(compare: Any) -> None:
    out = compare.unit_rows(np.asarray([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_heading_mask_ignores_case_like_the_sql_collation(compare: Any) -> None:
    copy = _copy(compare)
    assert compare.heading_mask(copy.headings, "History").tolist() == [
        False, True, True, False, True, False,
    ]


def test_four_filter_pages_applies_every_set_field(compare: Any) -> None:
    copy = _copy(compare)
    everything = Filters(category="States", min_words=1000, edited_since=date(2020, 12, 1),
                         linked_from="Asia")
    # Page 1 is too short and not linked from Asia; page 3 has no edit date.
    assert compare.four_filter_pages(copy, everything) == {2}
    assert compare.four_filter_pages(copy, Filters(category="States")) == {1, 2, 3}
    assert compare.four_filter_pages(copy, Filters(category="Unknown")) == set()
    assert compare.four_filter_pages(copy, Filters(edited_since=date(2020, 12, 3))) == {1}
    assert compare.page_mask(copy.page_ids, {2}).tolist() == [False, False, True, True, False, False]
    assert not compare.page_mask(copy.page_ids, set()).any()
    with pytest.raises(ValueError, match="max_words"):
        compare.four_filter_pages(copy, Filters(max_words=10))


def test_copy_mask_routes_both_cases(compare: Any) -> None:
    copy = _copy(compare)
    history, four = compare.FILTER_CASES
    assert (history.key, four.key) == ("history", "four_filters")
    assert compare.copy_mask(copy, history).tolist() == [False, True, True, False, True, False]
    # The real four-filter case names a category this toy copy does not have.
    assert not compare.copy_mask(copy, four).any()


def test_compare_lists_reports_overlap_set_and_order(compare: Any) -> None:
    assert compare.compare_lists([1, 2, 3], [1, 2, 3]) == (1.0, True, True)
    assert compare.compare_lists([2, 1, 3], [1, 2, 3]) == (1.0, True, False)
    assert compare.compare_lists([1, 9], [1, 2]) == (0.5, False, False)
    assert compare.compare_lists([], [1, 2]) == (0.0, False, False)
    assert compare.compare_lists([], []) == (1.0, True, True)


def test_summarise_counts_short_and_empty_answers(compare: Any) -> None:
    row = compare.summarise(
        "m", "here", "exact",
        samples_ms=[1.0, 2.0, 3.0, 4.0],
        results=[[1, 2], [1], []],
        references=[[1, 2], [1, 2], [1, 2]],
    )
    assert row["p50_ms"] == 2.5 and row["samples"] == 4 and row["queries"] == 3
    assert row["mean_overlap_with_exact"] == 0.5  # (1.0 + 0.5 + 0.0) / 3
    assert (row["same_set_as_exact"], row["same_order_as_exact"]) == (1, 1)
    assert (row["short"], row["no_row"]) == (2, 1)
    # Nothing passes the filter: an empty answer is right, neither short nor "no row".
    empty = compare.summarise("m", "here", "exact", [1.0], [[]], [[]])
    assert (empty["short"], empty["no_row"], empty["mean_overlap_with_exact"]) == (0, 0, 1.0)


def test_time_methods_interleaves_the_methods_and_rotates_who_goes_first(compare: Any) -> None:
    calls: list[tuple[str, int]] = []

    def method(name: str) -> Any:
        def answer(index: int) -> list[int]:
            calls.append((name, index))
            return [index, len(calls)]

        return answer

    timed = compare.time_methods([method("a"), method("b")], n_queries=2, repeats=2)

    # One untimed pass and two timed ones; every query goes to both methods before the next.
    assert calls == [
        ("a", 0), ("b", 0), ("b", 1), ("a", 1),  # untimed pass
        ("b", 0), ("a", 0), ("a", 1), ("b", 1),  # first timed pass: the other method starts
        ("a", 0), ("b", 0), ("b", 1), ("a", 1),  # second timed pass
    ]
    (samples_a, first_a), (samples_b, first_b) = timed
    assert len(samples_a) == len(samples_b) == 4  # 2 queries x 2 timed passes
    assert all(sample >= 0.0 for sample in samples_a + samples_b)
    assert first_a == [[0, 6], [1, 7]] and first_b == [[0, 5], [1, 8]]  # first timed pass only


def test_fetch_rows_binds_the_ids_and_builds_only_placeholders(compare: Any) -> None:
    executed: list[tuple[str, tuple[int, ...]]] = []

    class Cursor:
        def execute(self, sql: str, params: tuple[int, ...]) -> None:
            executed.append((sql, params))

        def fetchall(self) -> list[tuple[int]]:
            return [(1,), (2,)]

    class Connection:
        @contextmanager
        def cursor(self) -> Iterator[Cursor]:
            yield Cursor()

    assert compare.fetch_rows(Connection(), [5, 7]) == 2
    assert compare.fetch_rows(Connection(), []) == 0  # no statement for no ids
    assert len(executed) == 1
    sql, params = executed[0]
    assert sql.endswith("WHERE chunk.chunk_id IN (%s, %s)") and params == (5, 7)
    assert "5" not in sql and "7" not in sql


def _report(with_hnswlib: bool) -> dict[str, Any]:
    row = {
        "method": "a method", "where": "MariaDB", "kind": "exact", "p50_ms": 1.25, "p95_ms": 2.5,
        "mean_ms": 1.5, "samples": 10, "queries": 2, "mean_overlap_with_exact": 1.0,
        "same_set_as_exact": 2, "same_order_as_exact": 2, "short": 0, "no_row": 0,
    }
    hnsw = {"version": "0.8.0", "M": 16, "ef_construction": 200, "ef": 100,
            "build_seconds": 0.5, "index_file_bytes": 1234}
    return {
        "created_at": "2026-11-11T00:00:00+00:00",
        "parameters": {
            "k": 10, "repeats": 5, "n_queries": 2, "n_chunks": 6, "dimensions": 2,
            "mhnsw_ef_search": 100, "index_m": 16, "hnswlib": hnsw if with_hnswlib else None,
            "numpy_blas_threads": 4,
        },
        "copy": {"export_seconds": 0.25, "vector_bytes": 48, "rows_exported": {"chunk": 6}},
        "unfiltered": [row],
        "filtered": {"history": {"label": "a filter", "passing_chunks": 3, "rows": [row]}},
        "machine": {"cpu": "a cpu", "cpu_count": 8, "memory_gb": 16.0},
        "versions": {"mariadb": "11.8.9", "python": "3.12", "numpy": "2.5.3"},
        "notes": ["first note", "second note"],
    }


def test_markdown_shows_every_table_and_says_when_hnswlib_is_missing(compare: Any) -> None:
    with_library = compare.to_markdown(_report(True), "demo")
    without = compare.to_markdown(_report(False), "demo")

    assert with_library.startswith("# MariaDB against a separate in-process index: demo")
    assert "## No filter" in with_library and "## Filter: a filter" in with_library
    assert "| a method | exact | 1.250 | 2.500 | 1.000 | 2/2 | 0/2 | 0/2 |" in with_library
    assert "3 of 6 chunks pass." in with_library
    assert "hnswlib 0.8.0: M = 16, ef_construction = 200" in with_library
    assert "- first note" in with_library and "- second note" in with_library
    assert "hnswlib was not installed for this run" in without
    assert "hnswlib 0.8.0" not in without


def test_hnswlib_stays_out_of_every_requirements_and_lock_file() -> None:
    # The comparison may use hnswlib, but `make setup` must never install it.
    for name in ("requirements.txt", "requirements-dev.txt", "requirements.lock",
                 "requirements-cpu.lock", "pyproject.toml"):
        assert "hnswlib" not in (REPO_ROOT / name).read_text(encoding="utf-8").lower(), name


def test_the_export_statements_take_no_parameters(compare: Any) -> None:
    for statement in (compare.EXPORT_CHUNKS_SQL, compare.EXPORT_PAGES_SQL,
                      compare.EXPORT_CATEGORIES_SQL, compare.EXPORT_LINKS_SQL):
        assert statement.startswith("SELECT ") and "%" not in statement and "{" not in statement
