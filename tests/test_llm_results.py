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


def test_newcombe_matches_the_published_example() -> None:
    # Newcombe (1998), 56/70 against 48/80: method 10 gives 0.0524 to 0.3339.
    diff, lo, hi = lr.newcombe(56, 70, 48, 80)
    assert diff == pytest.approx(0.2)
    assert (lo, hi) == pytest.approx((0.0524, 0.3339), abs=1e-4)
    assert all(math.isnan(x) for x in lr.newcombe(1, 0, 1, 2))


def test_resample_within_keeps_every_stratum_and_repeats_for_a_seed() -> None:
    strata = np.array([0, 0, 1, 1, 1, 2, 2, 0])
    first = next(lr.resample_within(strata, 5))
    assert first.shape == (lr.BLOCK, len(strata))
    assert (strata[first] == strata).all()
    assert np.array_equal(first, next(lr.resample_within(strata, 5)))
    assert not np.array_equal(first, next(lr.resample_within(strata, 6)))


def test_read_claim_tags_checks_the_sha256(tmp_path) -> None:
    path = tmp_path / "claims.jsonl"
    path.write_text('{"id": 3, "challenge": "Other"}\n\n{"id": 7, "challenge": "Multi-hop Reasoning"}\n',
                    encoding="utf-8")
    digest = lr.hashlib.sha256(path.read_bytes()).hexdigest()
    assert lr.read_claim_tags(path, digest) == {3: "Other", 7: "Multi-hop Reasoning"}
    with pytest.raises(SystemExit, match="SHA-256"):
        lr.read_claim_tags(path, "0" * 64)


def _mh_table(cells: list[tuple[str, str, str, dict[str, int]]]) -> tuple[dict, dict]:
    """A per-claim table from (label, kind, tag, {condition: right}) cells, and the tags."""
    table: dict = {c: {} for c in lr.CONDITIONS}
    tags = {}
    for cid, (label, kind, tag, right) in enumerate(cells):
        tags[cid] = tag
        for c in lr.CONDITIONS:
            table[c][cid] = {"label": label, "kind": kind, "answer": label if right[c] else "x",
                             "right": right[c]}
    return table, tags


def test_multihop_kinds_pairs_coverage_and_tags_on_a_hand_made_table() -> None:
    # Two claims in each (label, kind, tag) cell. Gold is always right, no passages never;
    # RAG misses every supported claim on two or more pages, web every refuted one-page claim.
    cells, flags = [], {}
    for label in lr.SIDES:
        for kind in lr.KINDS:
            for tag in (lr.MULTIHOP_TAG, lr.OTHER_TAG):
                for j in range(2):
                    right = {"gold": 1, "no_retrieval": 0,
                             "retrieved": int(not (label == "SUPPORTS" and kind == "needs2")),
                             "web": int(not (label == "REFUTES" and kind == "single"))}
                    flags[len(cells)] = (1, int(kind == "single" or j == 0))
                    cells.append((label, kind, tag, right))
    for _ in range(2):   # NOT ENOUGH INFO claims are left out but count in the order of the tags
        flags[len(cells)] = (1, 1)
        cells.append(("NOT ENOUGH INFO", "single", lr.OTHER_TAG, dict.fromkeys(lr.CONDITIONS, 1)))
    table, tags = _mh_table(cells)
    # Claims 0 and 4 are supported (one article, two or more), 16 is NOT ENOUGH INFO.
    out = lr.multihop(table, sorted(tags), tags, flags, repeat_ids=frozenset({0, 4, 16}))
    assert out["n"] == 16 and out["tag_order"] == [lr.OTHER_TAG, lr.MULTIHOP_TAG]
    assert out["repeats_per_kind"] == {"single": 1, "needs2": 1}
    assert out["meanings"] == {"two_or_more_articles": 8, "multi_hop_reasoning_tag": 8, "both": 4,
                               "both_per_label": {"SUPPORTS": 2, "REFUTES": 2}}
    k = out["kinds"]
    rag = k["accuracy"]["retrieved"]
    assert rag["needs2"]["SUPPORTS"]["right"] == 0 and rag["needs2"]["REFUTES"]["right"] == 4
    # Every cell is constant, so a resample that stays inside the strata never moves a figure.
    assert rag["needs2"]["equal_weight"]["point"] == 0.5
    assert rag["needs2"]["equal_weight"]["bootstrap"] == [0.5, 0.5]
    gap = k["gap_one_minus_two"]["retrieved"]
    assert gap["SUPPORTS"]["difference"] == 1.0 and gap["REFUTES"]["level"] is True
    assert gap["equal_weight"]["point"] == 0.5 and gap["equal_weight"]["level"] is False
    paired = k["paired"]["needs2"]["retrieved-gold"]
    assert (paired["SUPPORTS"]["b_only_first_right"], paired["SUPPORTS"]["c_only_second_right"]) == (0, 4)
    assert paired["SUPPORTS"]["mcnemar_p"] == pytest.approx(0.125)
    assert paired["equal_weight"]["point"] == -0.5
    change = k["paired_change_two_minus_one"]
    assert change["retrieved-gold"]["equal_weight"]["point"] == -0.5
    assert change["retrieved-web"]["equal_weight"]["point"] == -1.0
    assert change["retrieved-web"]["SUPPORTS"]["point"] == -1.0
    assert change["retrieved-web"]["REFUTES"]["point"] == -1.0
    cov = out["coverage"]
    assert cov["needs2"]["gold_set_in_top5"]["SUPPORTS"]["pct"] == "50.0"
    assert cov["needs2"]["rag_set_in_minus_not"]["equal_weight"]["point"] == 0.0
    assert cov["single"]["rag_when_set_not_in_top5"]["equal_weight"] is None
    assert set(cov["single"]["rag_set_in_minus_not"]) == {"equal_weight"}
    tag = out["tags"][lr.MULTIHOP_TAG]
    assert tag["n"] == 8 and tag["small"] is True
    assert tag["n_per_label"] == {"SUPPORTS": 4, "REFUTES": 4}
    assert tag["counts"]["REFUTES"] == {"single": 2, "needs2": 2}
    mvo = out["multihop_vs_other"]
    assert mvo["accuracy"]["all"]["retrieved"][lr.MULTIHOP_TAG]["equal_weight"]["point"] == 0.75
    assert mvo["accuracy"]["needs2"]["web"][lr.OTHER_TAG]["equal_weight"]["point"] == 1.0
    assert mvo["gap"]["single"]["retrieved"]["equal_weight"]["point"] == 0.0
    assert mvo["rag_minus_gold"][lr.MULTIHOP_TAG]["equal_weight"]["point"] == -0.25
    assert mvo["rag_minus_gold_difference"]["equal_weight"]["point"] == 0.0
    text = "\n".join(lr.multihop_lines(out))
    assert "## 6. Multi-hop claims" in text and "not computable" in text


def test_multihop_weights_the_two_labels_equally_and_flags_a_missing_label() -> None:
    def right(r: int) -> dict[str, int]:
        return dict.fromkeys(lr.CONDITIONS, r)

    cells = [("SUPPORTS", "single", "Numerical Reasoning", right(1)),
             ("SUPPORTS", "single", "Numerical Reasoning", right(1)),
             ("SUPPORTS", "single", "Numerical Reasoning", right(0)),
             ("REFUTES", "single", "Numerical Reasoning", right(0)),
             ("SUPPORTS", "single", "Search terms not in claim", right(1)),
             ("SUPPORTS", "needs2", lr.OTHER_TAG, right(1)),
             ("REFUTES", "needs2", lr.OTHER_TAG, right(0))]
    table, tags = _mh_table(cells)
    out = lr.multihop(table, sorted(tags), tags, dict.fromkeys(tags, (1, 1)))
    numerical = out["tags"]["Numerical Reasoning"]["accuracy"]["gold"]
    # 2 of 3 supported and 0 of 1 refuted: equal weight 1/3, where the plain share is 2/4.
    assert numerical["equal_weight"]["point"] == pytest.approx(1 / 3)
    assert numerical["equal_weight"]["point_pct"] == "33.3"
    assert out["tags"]["Search terms not in claim"]["accuracy"]["gold"]["equal_weight"] is None
    assert out["multihop_vs_other"] is None
