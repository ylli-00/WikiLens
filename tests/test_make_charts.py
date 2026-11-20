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


HEADLINE = {
    "n_claims": 75, "n_evidence_claims": 66,
    "per_k": [{"k": k, "article_recall": a, "article_hits": int(a * 75),
               "evidence_recall": e, "evidence_hits": int(e * 66)}
              for k, a, e in ((1, 0.84, 0.59), (3, 0.99, 0.79), (20, 0.99, 0.98))],
}  # fmt: skip


def _llm_condition(acc: float, f1s: float, f1r: float) -> dict:
    def entry(value: float) -> dict:
        return {"acc": value, "wilson": [value - 0.02, value + 0.02], "pct": f"{100 * value:.1f}",
                "wilson_pct": [f"{100 * value - 2:.1f}", f"{100 * value + 2:.1f}"],
                "right": round(value * 2000), "n": 2000}

    def label(f1: float) -> dict:
        return {"f1": f1, "f1_pct": f"{100 * f1:.1f}", "f1_bootstrap_pct": ["1.0", "2.0"],
                "precision_pct": "90.0", "recall_pct": "80.0"}

    return {"accuracy": entry(acc), "per_label": {"SUPPORTS": label(f1s), "REFUTES": label(f1r)}}


LLM = {
    "conventions": {"bootstrap_resamples": 10_000},
    "three_class": {"conditions": {c: _llm_condition(a, 0.5, 0.5) for c, a in (
        ("no_retrieval", 0.45), ("web", 0.61), ("retrieved", 0.71), ("gold", 0.73))}},
    "supported_refuted": {"n": 2000, "conditions": {c: _llm_condition(a, s, r) for c, a, s, r in (
        ("no_retrieval", 0.35, 0.435, 0.539), ("web", 0.82, 0.839, 0.844),
        ("retrieved", 0.84, 0.882, 0.856), ("gold", 0.83, 0.905, 0.851))}},
}


@pytest.mark.parametrize(
    ("draw", "data", "title"),
    [
        ("llm_accuracy", LLM, "LLM fact-checker accuracy"),
        ("llm_f1", LLM, "F1 per label"),
        ("recall_at_k", HEADLINE, "Recall at k"),
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


def test_main_skips_missing_results_and_writes_the_rest(
    charts: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "final_ef_100.json").write_text(json.dumps(HEADLINE), encoding="utf-8")
    out = tmp_path / "img"
    assert charts.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["recall_at_k.svg"]
    assert "skipped scaled_recall_at_k.svg" in capsys.readouterr().err


def _scaled_run(n_chunks: int, shift: float) -> dict[str, Any]:
    return {"n_claims": 3000, "parameters": {"n_chunks": n_chunks},
            "per_k": [{"k": k, "article_recall": a - shift, "evidence_recall": e - shift}
                      for k, a, e in ((1, 0.96, 0.45), (5, 0.98, 0.78), (20, 0.99, 0.89))]}


def test_the_scaled_chart_draws_both_corpora_and_reads_two_files(
    charts: Any, tmp_path: Path
) -> None:
    data = {"s0_full_run1": _scaled_run(43134, 0.0), "s1_full_run1": _scaled_run(132359, 0.02)}
    svg = charts.scaled_recall_at_k(data)
    document = xml.dom.minidom.parseString(svg)
    assert "scaled run" in document.getElementsByTagName("title")[0].firstChild.data
    assert svg.count("<polyline") == 4 and svg.count('stroke-dasharray="6 4"') == 2  # S0 dashed
    assert "43,134 chunks" in svg and "132,359 chunks" in svg
    (tmp_path / "scaled" / "eval").mkdir(parents=True)
    for name, run in data.items():
        (tmp_path / "scaled" / "eval" / f"{name}.json").write_text(json.dumps(run), encoding="utf-8")
    out = tmp_path / "img"
    assert charts.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    assert (out / "scaled_recall_at_k.svg").read_text(encoding="utf-8") == svg
