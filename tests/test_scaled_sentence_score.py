"""scripts/scaled_sentence_score.py without a database or a model: the 5 + 25 selection, the
found rule and the group summary."""

from __future__ import annotations

import importlib.util

import pytest

from wikilense.config import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "scaled_sentence_score", REPO_ROOT / "scripts" / "scaled_sentence_score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)

UNITS = [(10, "sentence"), (11, "item"), (12, "sentence"), (13, "sentence"), (14, "item")]


def test_selection_by_score_keeps_the_best_of_each_kind():
    scores = [0.1, 0.9, 0.8, 0.8, 0.2]
    assert score.select(UNITS, scores, n_sentences=2, n_items=1) == [11, 12, 13]


def test_ties_go_to_the_earlier_unit_and_hit_order_without_scores():
    assert score.select(UNITS, [0.5] * 5, n_sentences=1, n_items=1) == [10, 11]
    assert score.select(UNITS, None, n_sentences=2, n_items=5) == [10, 11, 12, 14]


def test_found_needs_a_complete_set():
    assert score.found([{1, 2}, {3}], [3, 9])
    assert not score.found([{1, 2}], [1, 9])
    assert not score.found([set()], [1])
    assert not score.found([], [1])


@pytest.mark.parametrize(("key", "kind"), [("sentence_0", "sentence"), ("item_2_7", "item")])
def test_unit_kind(key, kind):
    assert score.unit_kind(key) == kind


def test_unit_kind_refuses_a_cell():
    with pytest.raises(ValueError):
        score.unit_kind("cell_0_1_2")


def test_summary_counts_eligible_claims_only():
    rows = [{"label": "SUPPORTS", "kind": "single", "eligible": True, "in_top_chunks": True,
             "found_embed": True, "found_chunk_order": False},
            {"label": "SUPPORTS", "kind": "single", "eligible": False, "in_top_chunks": False,
             "found_embed": False, "found_chunk_order": False},
            {"label": "REFUTES", "kind": "needs2", "eligible": True, "in_top_chunks": True,
             "found_embed": False, "found_chunk_order": False}]
    s = score.summarize(rows)
    assert s["overall"]["claims"] == 3 and s["overall"]["eligible"] == 2
    assert s["overall"]["found_embed"] == {"hits": 1, "rate": 0.5}
    assert s["groups"]["SUPPORTS / single"]["found_embed"]["rate"] == 1.0
    assert s["groups"]["NOT ENOUGH INFO / needs2"]["eligible"] == 0
    assert s["groups"]["NOT ENOUGH INFO / needs2"]["found_embed"]["rate"] is None


spec_pc = importlib.util.spec_from_file_location(
    "scaled_page_coverage", REPO_ROOT / "scripts" / "scaled_page_coverage.py")
pc = importlib.util.module_from_spec(spec_pc)
spec_pc.loader.exec_module(pc)


def test_page_coverage_is_the_share_of_gold_pages_in_the_first_p():
    assert pc.distinct_pages([5, 5, 7, 5, 9]) == [5, 7, 9]
    assert pc.coverage({7, 9}, [5, 7, 9], 2) == 0.5
    assert pc.coverage({7, 9}, [5, 7, 9], 3) == 1.0
    with pytest.raises(ValueError):
        pc.coverage(set(), [1], 1)


def test_page_coverage_summary():
    rows = [("SUPPORTS", {1}, [1, 2]), ("NOT ENOUGH INFO", {3, 4}, [3]), ("REFUTES", set(), [1])]
    s = pc.summarize(rows)
    assert s["without_gold_pages"] == 1 and s["all"]["n"] == 2
    assert s["all"]["p1"] == {"coverage": 0.75, "complete": 0.5, "fewer_than_p_pages": 0}
    assert s["supports_refutes"]["p1"]["coverage"] == 1.0
    assert s["all"]["p2"]["fewer_than_p_pages"] == 1
