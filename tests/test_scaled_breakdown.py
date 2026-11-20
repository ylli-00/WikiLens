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
