"""The statistics of ``scripts/llm_results.py``: rounding, intervals, tests and resampling."""

from __future__ import annotations

import importlib.util
import math
from fractions import Fraction

import numpy as np
import pytest

from wikilense.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("llm_results", REPO_ROOT / "scripts" / "llm_results.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)


def test_rounding_is_half_up_on_exact_halves() -> None:
    # 1,657 / 2,000 = 82.85% and 705 / 2,000 = 35.25% sit exactly on a half.
    assert lr.pct(1657, 2000) == "82.9"
    assert lr.pct(705, 2000) == "35.3"
    assert lr.r1(Fraction(3, 4)) == "0.8"
    assert lr.r1(0.25) == "0.3"
    assert lr.r1(-0.25) == "-0.3"


def test_mcnemar_exact_matches_the_binomial_tail() -> None:
    assert lr.mcnemar_exact(0, 5) == Fraction(2, 32)
    assert lr.mcnemar_exact(3, 3) == 1
    assert lr.mcnemar_exact(0, 0) == 1
    # one-sided tail of 10 of 30 discordant, doubled
    tail = sum(math.comb(30, i) for i in range(11))
    assert lr.mcnemar_exact(10, 20) == Fraction(2 * tail, 2 ** 30)
    assert lr.mcnemar_exact(20, 10) == lr.mcnemar_exact(10, 20)


def test_p_values_keep_two_significant_digits() -> None:
    assert lr.p_text(Fraction(39, 1000)) == "0.039"
    assert lr.p_text(Fraction(3049, 10000)) == "0.30"
    assert lr.p_text(Fraction(1, 1)) == "1.0"
    assert lr.p_text(Fraction(67, 10 ** 252)) == "6.7e-251"


def test_wilson_interval_is_symmetric_at_one_half_and_inside_zero_one() -> None:
    lo, hi = lr.wilson(5, 10)
    assert lo == pytest.approx(1 - hi)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    lo, hi = lr.wilson(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.2


def test_paired_interval_follows_the_formula() -> None:
    diff, lo, hi = lr.paired_interval(145, 111, 2000)
    half = 1.96 * math.sqrt(145 + 111 - 34 ** 2 / 2000) / 2000
    assert diff == pytest.approx(34 / 2000)
    assert (lo, hi) == pytest.approx((34 / 2000 - half, 34 / 2000 + half))


def test_rerun_sd() -> None:
    assert lr.rerun_sd(0.08, 2000) == pytest.approx(math.sqrt(0.08 / 4000))


def test_f1_from_counts_agrees_with_prf() -> None:
    rng = np.random.default_rng(7)
    gold, pred = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    conf = np.zeros((3, 3), dtype=int)
    np.add.at(conf, (gold, pred), 1)
    for label in range(3):
        assert float(lr.f1_from_counts(conf, label)) == pytest.approx(lr.prf(gold, pred, label)[2])


def test_resamples_repeat_for_a_seed_and_are_paired() -> None:
    first = next(lr.resample_blocks(50, 11))
    again = next(lr.resample_blocks(50, 11))
    assert first.shape == (lr.BLOCK, 50) and np.array_equal(first, again)
    assert not np.array_equal(first, next(lr.resample_blocks(50, 12)))


def test_window_use_counts_each_window_from_its_first_reading() -> None:
    def row(ts: str, value: float, reset: int) -> dict:
        info = {"unifiedWindows": {"five_hour": {"utilization": value, "resetsAt": reset}}}
        return {"ts": ts, "cli": {"last_rate_limit_event": {"rate_limit_info": info}}}

    rows = [row("t01", 0.10, 1), row("t03", 0.14, 1), row("t02", 0.15, 1),   # out of order
            row("t04", 0.02, 2), row("t05", 0.07, 2)]
    out = lr.window_use(rows, "five_hour")
    assert out["windows"] == 2
    assert out["used"] == pytest.approx(0.05 + 0.05)


def test_sides_counts_nei_answers_as_wrong_and_pairs_the_claims() -> None:
    labels = ["SUPPORTS", "REFUTES"] * 10
    answers = {
        "no_retrieval": ["NOT ENOUGH INFO"] * 20,
        "web": labels[:18] + ["REFUTES", "SUPPORTS"],
        "retrieved": labels[:],
        "gold": labels[:19] + ["NOT ENOUGH INFO"],
    }
    table = {c: {i: {"label": labels[i], "kind": "single", "answer": a[i],
                     "right": int(a[i] == labels[i])} for i in range(20)}
             for c, a in answers.items()}
    out = lr.sides(table, list(range(20)))
    assert out["conditions"]["no_retrieval"]["accuracy"]["right"] == 0
    assert out["conditions"]["no_retrieval"]["picked_a_side"] == 0
    assert out["conditions"]["gold"]["picked_a_side"] == 19
    pair = out["pairs"]["retrieved-web"]
    assert (pair["b_only_first_right"], pair["c_only_second_right"]) == (2, 0)
    assert pair["mcnemar_p"] == pytest.approx(0.5)
    assert out["pairs"]["retrieved-gold"]["level"] is True


def test_two_sd_rule_takes_the_larger_sd_and_flags_small_gaps() -> None:
    def var_entry(fs: tuple[float, float, float]) -> dict:
        pairs = {name: {"rerun_sd": {n: {"sd_difference": math.sqrt(2) * lr.rerun_sd(f, int(n))}
                                     for n in ("2000", "3000")}}
                 for name, f in zip(("original-r1", "original-r2", "r1-r2"), fs, strict=True)}
        return {"three_class_450": {"pairs": pairs}, "supported_refuted_300": {"pairs": pairs}}

    var = {c: var_entry((0.02, 0.04, 0.03)) for c in lr.CONDITIONS}
    var["web"] = var_entry((0.05, 0.08, 0.06))
    gaps = {f"{a}-{b}": {"difference": 0.012, "difference_pct": "1.2", "level": False}
            for a, b in lr.PAIRS}
    rule = lr.two_sd_rule({"three_class": gaps, "supported_refuted": gaps}, var)
    rag_web = rule["supported_refuted"]["retrieved-web"]
    assert rag_web["from"] == {"condition": "web", "runs": "original-r2"}
    assert rag_web["two_sd"] == pytest.approx(2 * math.sqrt(2) * math.sqrt(0.08 / 4000))
    assert rag_web["below_two_sd"] is True  # 1.2 points against two SDs of about 1.26
    rag_gold = rule["supported_refuted"]["retrieved-gold"]
    assert rag_gold["from"]["runs"] == "original-r2"
    assert rag_gold["two_sd"] == pytest.approx(2 * math.sqrt(2) * math.sqrt(0.04 / 4000))
    assert rag_gold["below_two_sd"] is False
    assert lr.two_sd_rule({}, {"missing": "x"}) == {"missing": "x"}
