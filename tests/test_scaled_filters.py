"""scripts/scaled_filters.py: the rule that picks filter values on a scaled database (no server)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pytest

from wikilense.config import REPO_ROOT
from wikilense.search import Filters

SCRIPT = REPO_ROOT / "scripts" / "scaled_filters.py"


@pytest.fixture(scope="module")
def sf() -> Any:
    spec = importlib.util.spec_from_file_location("scaled_filters", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nearest_is_nearest_on_a_log_scale_and_skips_empty_values(sf: Any) -> None:
    candidates = [("a", 0), ("b", 40), ("c", 160), ("d", 1000)]
    # Target 80: 40 and 160 are both a factor of 2 away; the larger count wins the tie.
    assert sf.nearest(candidates, 80) == ("c", 160)
    assert sf.nearest(candidates, 50) == ("b", 40)
    assert sf.nearest(candidates, 10_000) == ("d", 1000)  # beyond every value: the largest
    assert sf.nearest(candidates, 1) == ("b", 40)  # a value that keeps nothing is never picked
    with pytest.raises(ValueError):
        sf.nearest([("a", 0)], 5)


def test_threshold_candidates_count_the_rows_at_or_above_each_value(sf: Any) -> None:
    values = np.asarray([300, 100, 200, 100])
    weights = np.asarray([5, 1, 3, 2])
    assert sf.threshold_candidates(values, weights) == [(100, 11), (200, 8), (300, 5)]


def _naive_utc(*parts: int) -> datetime:
    """A naive UTC datetime, as PyMySQL returns ``page.last_edited``."""
    return datetime(*parts, tzinfo=timezone.utc).replace(tzinfo=None)


def _tables() -> dict[str, Any]:
    """Five pages: lengths 100..500 words, 10..50 chunks, edited on five December days."""
    pages = {p: {"title": f"P{p}", "n_words": 100 * p,
                 "last_edited": _naive_utc(2020, 12, 10 + p, 12)} for p in range(1, 6)}
    pages[5]["last_edited"] = None
    return {
        "pages": pages,
        "chunks": {p: 10 * p for p in range(1, 6)},
        "categories": {"Big": {3, 4, 5}, "Small": {1}, "Mid": {2, 4}},
        "links": {1: {4, 5}, 2: {1, 2}, 3: {4}},
    }


def test_pick_applies_the_rule_and_records_what_passes(sf: Any) -> None:
    report = sf.pick(_tables())
    assert report["n_chunks"] == 150 and report["n_pages"] == 5
    cases = {case["name"]: case for case in report["cases"]}
    assert cases["none"]["passing_chunks"] == 150
    # Length: 75 chunks is the target of the 50% level; >= 400 words keeps 90, >= 500 keeps 50.
    assert cases["length_50pct"]["filters"] == {"min_words": 400}
    assert cases["length_50pct"]["passing_chunks"] == 90
    # Edit date: page 5 has none, so the dated chunks are 100; the 50% target (75) is nearest
    # to the 70 chunks edited since page 3's moment.
    assert cases["edited_50pct"]["filters"] == {"edited_since": "2020-12-13T12:00:00"}
    assert cases["edited_50pct"]["passing_chunks"] == 70
    # Category and linked from: every target maps to one of the few values; a value picked by
    # two targets is kept once, under the looser one.
    names = [case["name"] for case in report["cases"]]
    assert len(names) == len(set(names))
    category_values = [c["filters"]["category"] for c in report["cases"] if c["filter"] == "category"]
    assert len(category_values) == len(set(category_values))
    # Category, 50% (75 chunks): "Mid" keeps 60, "Big" 120; 60 is the nearer on a log scale.
    assert cases["category_50pct"]["filters"] == {"category": "Mid"}
    # four_filters: length and date at their 50% values (>= 400 words, since 12-13 12:00,
    # so only page 4) and the category/linking pair that keeps most of it.
    four = cases["four_filters"]
    assert four["filters"]["min_words"] == 400
    assert four["filters"]["edited_since"] == "2020-12-13T12:00:00"
    assert four["passing_chunks"] == 40
    # Pages P1 and P3 both reach page 4, which is in Big and Mid: the first in sort order wins.
    assert (four["filters"]["category"], four["filters"]["linked_from"]) == ("Big", "P1")


def test_load_cases_reads_dates_and_moments_back(sf: Any, tmp_path: Any) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"cases": [
        {"name": "a", "filters": {"edited_since": "2020-12-31"}, "passing_chunks": 3},
        {"name": "b", "filters": {"edited_since": "2020-12-31T23:59:17", "min_words": 5},
         "passing_chunks": 2},
    ]}), encoding="utf-8")
    assert sf.load_cases(path) == (
        ("a", Filters(edited_since=date(2020, 12, 31)), 3),
        ("b", Filters(edited_since=_naive_utc(2020, 12, 31, 23, 59, 17), min_words=5), 2),
    )
    assert sf.describe(Filters(min_words=5, edited_since=date(2020, 1, 2))) == {
        "min_words": 5, "edited_since": "2020-01-02"}
