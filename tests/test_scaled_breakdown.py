"""scripts/scaled_breakdown.py: groups, challenge tags and the FEVEROUS weights (no server)."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

from wikilense.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "scaled_breakdown.py"


@pytest.fixture(scope="module")
def sb() -> Any:
    spec = importlib.util.spec_from_file_location("scaled_breakdown", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_csv_kind_puts_mixed_claims_in_no_group(sb: Any) -> None:
    assert sb.csv_kind(1, 1) == "single"
    assert sb.csv_kind(2, 3) == "needs2"
    assert sb.csv_kind(1, 2) is None


def _row(label: str, lo: int, hi: int, modalities: str) -> dict[str, str]:
    return {"label": label, "min_pages": str(lo), "max_pages": str(hi),
            "modality_summary": modalities}


def test_feverous_weights_count_each_group_and_the_text_subset(sb: Any) -> None:
    rows = [
        _row("SUPPORTS", 1, 1, "sentence-only"),
        _row("SUPPORTS", 1, 1, "table-only"),
        _row("SUPPORTS", 2, 2, "mixed|list-only"),
        _row("REFUTES", 1, 1, "sentence-only|table-only"),
        _row("NOT ENOUGH INFO", 1, 2, "sentence-only"),  # sets of both kinds: no group
    ]
    weights = sb.feverous_weights(rows)
    assert weights["n_rows"] == 5
    assert weights["all"]["total"] == 4 and weights["text"]["total"] == 3
    assert weights["all"]["groups"]["SUPPORTS|single"] == {"count": 2, "weight": 0.5}
    assert weights["text"]["groups"]["SUPPORTS|single"]["count"] == 1
    assert weights["text"]["groups"]["SUPPORTS|needs2"]["count"] == 1
    for name in ("all", "text"):
        assert sum(g["weight"] for g in weights[name]["groups"].values()) == pytest.approx(1.0)


def test_measure_splits_by_group_and_tag_and_reweights(sb: Any) -> None:
    claims = {
        1: {"label": "SUPPORTS", "kind": "single", "challenge": "Other"},
        2: {"label": "SUPPORTS", "kind": "single", "challenge": "Multi-hop Reasoning"},
        3: {"label": "REFUTES", "kind": "needs2", "challenge": "Other"},
    }
    per_claim = [
        {"claim_id": 1, "gold_page_rank": 1, "evidence_rank": 2},
        {"claim_id": 2, "gold_page_rank": 6, "evidence_rank": None},
        {"claim_id": 3, "gold_page_rank": 3, "evidence_rank": 5},
    ]
    sentences = {1: {"in_top_chunks": True, "found_embed": True},
                 2: {"in_top_chunks": False, "found_embed": False},
                 3: {"in_top_chunks": True, "found_embed": False}}
    data = sb.measure(claims, per_claim, sentences)
    single = data["groups"]["SUPPORTS|single"]
    assert single["n"] == 2 and single["article@5"] == 0.5 and single["article@10"] == 1.0
    assert single["evidence@1"] == 0.0 and single["evidence@3"] == 0.5
    assert data["groups"]["REFUTES|needs2"]["evidence@5"] == 1.0
    assert data["groups"]["NOT ENOUGH INFO|single"]["n"] == 0
    assert data["overall"]["feverous_style"] == pytest.approx(1 / 3)
    assert data["tags"]["Other"]["n"] == 2 and data["tags"]["Other"]["article@5"] == 1.0
    weights = {"groups": {"SUPPORTS|single": {"weight": 0.25}, "REFUTES|needs2": {"weight": 0.75}}}
    assert sb.reweight(data["groups"], weights, "article@5") == pytest.approx(0.25 * 0.5 + 0.75)


def _rows(cells: list[tuple[str, str, str, int, bool]]) -> list[dict[str, Any]]:
    """Rows of claim_rows from (label, kind, tag, copies, evidence hit) cells; article@5 and the
    FEVEROUS-style score are hits for every claim."""
    return [{"group": f"{label}|{kind}", "challenge": tag, "article@5": True,
             "evidence@5": hit, "feverous_style": True}
            for label, kind, tag, copies, hit in cells for _ in range(copies)]


def test_label_weighting_weights_only_a_metric_that_differs_by_label(sb: Any) -> None:
    # evidence@5: every supported and refuted claim found, no NOT ENOUGH INFO claim. The other
    # two metrics find every claim, so no pair of labels differs and they stay unweighted.
    cells = []
    for label in sb.LABELS:
        for kind in sb.KINDS:
            for tag in (sb.OTHER_TAG, sb.MULTIHOP_TAG):
                cells.append((label, kind, tag, 5, label != "NOT ENOUGH INFO"))
    cells += [("SUPPORTS", "single", "Entity Disambiguation", 3, True),
              ("REFUTES", "single", "Entity Disambiguation", 1, True),
              ("NOT ENOUGH INFO", "single", "Entity Disambiguation", 1, False),
              ("SUPPORTS", "single", "Numerical Reasoning", 2, True)]
    out = sb.label_weighting(_rows(cells), seed=3)
    assert out["weighted"] == ["evidence@5"]
    assert out["differs"] == {"article@5": False, "evidence@5": True, "feverous_style": False}
    pair = out["check"]["evidence@5"]["needs2"]["SUPPORTS|NOT ENOUGH INFO"]
    assert pair["difference"] == 1.0 and pair["excludes_zero"] is True
    assert out["check"]["evidence@5"]["single"]["SUPPORTS|REFUTES"]["excludes_zero"] is False
    entity = out["tags"]["Entity Disambiguation"]["metrics"]["evidence@5"]
    assert entity["plain"]["rate"] == pytest.approx(0.8)              # 4 of 5 claims
    assert entity["equal_weight"]["point"] == pytest.approx(2 / 3)    # (1 + 1 + 0) / 3
    # Constant cells: a resample within label and tag cannot move the figure.
    assert entity["equal_weight"]["bootstrap"] == pytest.approx([2 / 3, 2 / 3])
    assert out["tags"]["Numerical Reasoning"]["metrics"]["evidence@5"]["equal_weight"] is None
    assert out["tags"]["Numerical Reasoning"]["small"] is True
    mvo = out["multihop_vs_other"]["evidence@5"]
    assert mvo["needs2"][sb.MULTIHOP_TAG]["plain"]["n"] == 15
    assert mvo["needs2"]["gap_equal_weight"]["point"] == 0.0
    assert mvo["all"]["gap_plain"]["excludes_zero"] is False
    text = "\n".join(sb.label_markdown({"metrics": list(sb.CHECK_METRICS),
                                        "bootstrap": {"resamples": 10, "seed": 3},
                                        "small_tag_below": 100, "databases": {"S0": out}}))
    assert "Left unweighted, because no pair's interval excludes 0 in either kind: article@5, " \
           "feverous_style." in text
    assert "not computable" in text


def test_label_weighting_adds_nothing_when_no_metric_differs(sb: Any) -> None:
    cells = [(label, kind, sb.OTHER_TAG, 4, True) for label in sb.LABELS for kind in sb.KINDS]
    out = sb.label_weighting(_rows(cells), seed=3)
    assert out["weighted"] == [] and out["tags"] == {} and out["multihop_vs_other"] is None
