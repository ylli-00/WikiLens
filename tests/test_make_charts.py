"""Tests for scripts/make_charts.py: each chart renders valid, labelled SVG from result data."""

from __future__ import annotations

import importlib.util
import json
import xml.dom.minidom
from pathlib import Path
from typing import Any

import pytest

from wikilense.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "make_charts.py"


@pytest.fixture(scope="module")
def charts() -> Any:
    spec = importlib.util.spec_from_file_location("make_charts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell(overlap: float) -> dict[str, float]:
    return {"mean_overlap": overlap, "min_overlap": overlap, "p50_ms": 0.2, "p95_ms": 0.3}


HEADLINE = {
    "n_claims": 75, "n_evidence_claims": 66,
    "per_k": [{"k": k, "article_recall": a, "article_hits": int(a * 75),
               "evidence_recall": e, "evidence_hits": int(e * 66)}
              for k, a, e in ((1, 0.84, 0.59), (3, 0.99, 0.79), (20, 0.99, 0.98))],
}  # fmt: skip
UNCOMMITTED = {
    "k": 10, "builds": 1, "efs": [20, 100],
    "datasets": {name: {"label": name, "builds": [{
        "rows": 1000, "uncommitted": {"20": _cell(0.6), "100": _cell(0.65)},
        "committed": {"20": _cell(0.8), "100": _cell(0.99)}}]} for name in ("chunks", "random")},
}  # fmt: skip
PLANNER = {
    "k": 10, "n_chunks": 4598,
    "cases": {
        "none": {"r_rows_p50": 10.0, "index_plans": 75, "queries": 75},
        "edited_before_2020_06_01": {"r_rows_p50": 4598.0, "index_plans": 0, "queries": 75},
    },
}


@pytest.mark.parametrize(
    ("draw", "data", "title"),
    [
        ("recall_at_k", HEADLINE, "Recall at k"),
        ("uncommitted_recall", UNCOMMITTED, "has not committed"),
        ("planner_rows", PLANNER, "Chunk rows read"),
    ],
)
def test_each_chart_is_valid_svg_with_a_title_and_a_description(
    charts: Any, draw: str, data: dict, title: str
) -> None:
    svg = getattr(charts, draw)(data)
    document = xml.dom.minidom.parseString(svg)
    root = document.documentElement
    assert root.tagName == "svg" and root.getAttribute("role") == "img"
    assert title in document.getElementsByTagName("title")[0].firstChild.data
    assert document.getElementsByTagName("desc")[0].firstChild.data
    assert "prefers-color-scheme: dark" in svg  # a dark variant, not an automatic flip


def test_planner_bars_are_coloured_by_the_plan_and_dates_are_dashed(charts: Any) -> None:
    svg = charts.planner_rows(PLANNER)
    assert svg.count('<path class="s1"') == 1 and svg.count('<path class="s2"') == 1
    assert "edited before 2020-06-01" in svg and ">4,000<" in svg


def test_main_skips_missing_results_and_writes_the_rest(
    charts: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "final_ef_100.json").write_text(json.dumps(HEADLINE), encoding="utf-8")
    out = tmp_path / "img"
    assert charts.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["recall_at_k.svg"]
    assert "skipped uncommitted_recall.svg" in capsys.readouterr().err
