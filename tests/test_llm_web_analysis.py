"""Tests of scripts/llm_web_analysis.py: the scores, the McNemar test, the run time, a full report."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from wikilense.config import REPO_ROOT


def load_module():
    spec = importlib.util.spec_from_file_location("llm_web_analysis",
                                                  REPO_ROOT / "scripts" / "llm_web_analysis.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_web_analysis"] = module
    spec.loader.exec_module(module)
    return module


wa = load_module()
S, R, N = "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"


def test_confusion_and_f1_count_a_missing_answer_as_a_miss():
    items = [(S, S), (S, R), (R, R), (N, wa.NO_ANSWER), (N, S)]
    table = wa.confusion(items)
    assert table[S] == {S: 1, R: 1, N: 0, wa.NO_ANSWER: 0}
    assert table[N] == {S: 1, R: 0, N: 0, wa.NO_ANSWER: 1}
    f1 = wa.f1_scores(items)
    # S: tp 1, predicted 2, actual 2 -> 0.5; R: tp 1, predicted 2, actual 1 -> 2/3; NEI: 0
    assert f1[S] == pytest.approx(0.5)
    assert f1[R] == pytest.approx(2 / 3)
    assert f1[N] == 0.0
    assert f1["macro"] == pytest.approx((0.5 + 2 / 3) / 3)


def test_mcnemar_exact_values():
    assert wa.mcnemar_exact(0, 0) == 1.0
    assert wa.mcnemar_exact(4, 4) == 1.0
    assert wa.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert wa.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert wa.mcnemar_exact(9, 1) == wa.mcnemar_exact(1, 9)
    assert wa.mcnemar_exact(0, 2000) == 0.0          # below the smallest float
    assert wa.fmt_p(0.0) == "< 1e-300" and wa.fmt_p(0.021484375) == "0.0215"


def test_percentile_is_nearest_rank_and_dist():
    xs = [float(x) for x in range(1, 11)]
    assert wa.percentile(xs, 0.90) == 9.0
    assert wa.percentile(xs, 0.95) == 10.0
    assert wa.percentile([7.0], 0.5) == 7.0
    d = wa.dist(xs)
    assert d["median"] == 5.5 and d["max"] == 10.0 and d["sum"] == 55.0
    assert wa.dist([]) is None


def test_active_seconds_is_the_union_of_intervals():
    assert wa.active_seconds([(0, 10), (5, 15), (20, 25)]) == 20
    assert wa.active_seconds([(0, 10), (2, 3)]) == 10
    assert wa.active_seconds([]) == 0


def row(cid, cond, gold, label, *, status="ok", final=True, ts="2026-11-16T03:00:00+01:00",
        latency=10.0, key=None, searches=1, fetches=1, window=None):
    return {"key": key or f"{cid}-{cond}", "claim_id": cid, "condition": cond, "gold_label": gold,
            "status": status, "final": final, "label": label, "latency_s": latency,
            "elapsed_s_total": latency, "attempts": 1, "ts": ts,
            "computed_usage": {"prompt_tokens": 100, "output_tokens": 10},
            "window_utilization": window, "n_web_search": searches, "n_web_fetch": fetches,
            "n_denied": 0, "num_turns": 3, "assistant_models": ["claude-haiku-5-5"]}


def test_final_rows_takes_the_newest_final_row_per_key():
    rows = [row(1, "web", S, R, key="k1"), row(1, "web", S, None, key="k1", final=False),
            row(1, "web", S, S, key="k1"), row(2, "web", R, None, status="error", final=False)]
    out = wa.final_rows(rows)
    assert set(out["web"]) == {1}
    assert out["web"][1]["label"] == S
    with pytest.raises(SystemExit):
        wa.final_rows([row(3, "web", S, S, key="a"), row(3, "web", S, S, key="b")])


def test_sessions_split_at_a_long_gap():
    rows = [row(1, "web", S, S, ts="2026-11-16T03:00:10+01:00", latency=10, window={"five_hour": 0.1}),
            row(2, "web", S, S, ts="2026-11-16T03:00:15+01:00", latency=10, window={"five_hour": 0.2}),
            row(3, "web", S, S, ts="2026-11-16T05:00:00+01:00", latency=30)]
    ss = wa.sessions(rows)
    assert len(ss) == 2
    assert ss[0]["rows"] == 2 and ss[0]["active_seconds"] == 15
    assert ss[0]["window_first"] == {"five_hour": 0.1} and ss[0]["window_last"] == {"five_hour": 0.2}
    assert ss[1]["active_seconds"] == 30 and ss[1]["window_first"] is None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_full_report(tmp_path):
    claims = [{"id": 1, "label": S, "kind": "single"}, {"id": 2, "label": R, "kind": "single"},
              {"id": 3, "label": N, "kind": "needs2"}, {"id": 4, "label": S, "kind": "needs2"}]
    claims_file = tmp_path / "claims.jsonl"
    write_jsonl(claims_file, claims)
    sha = hashlib.sha256(claims_file.read_bytes()).hexdigest()
    web = tmp_path / "web"
    write_jsonl(web / "results.jsonl", [
        row(1, "web", S, S), row(2, "web", R, S, searches=0, fetches=0), row(3, "web", N, N),
        row(4, "web", S, None, status="parse_failed")])              # claim 4: no answer
    old = tmp_path / "old"
    write_jsonl(old / "results.jsonl", [row(c["id"], "retrieved", c["label"], c["label"])
                                        for c in claims[:3]] + [row(4, "retrieved", S, R)])
    rep = tmp_path / "web50"
    write_jsonl(rep / "results.jsonl", [row(1, "web", S, R), row(3, "web", N, N)])
    assert wa.main(["--web-run", str(web), "--old-run", str(old), "--claims", str(claims_file),
                    "--claims-sha256", sha, "--repeat-run", str(rep)]) == 0
    res = json.loads((web / "web_analysis.json").read_text(encoding="utf-8"))
    w = res["conditions"]["web"]
    assert w["correct"] == 2 and w["answered"] == 3 and w["accuracy"] == 0.5
    assert w["confusion"][S][wa.NO_ANSWER] == 1 and w["confusion"][R][S] == 1
    assert w["by_kind"] == {"single": {"n": 2, "correct": 1}, "needs2": {"n": 2, "correct": 1}}
    assert res["conditions"]["retrieved"]["correct"] == 3
    assert res["paired"]["retrieved"] == {"both_right": 2, "only_web": 0, "only_other": 1,
                                          "both_wrong": 1, "mcnemar_p": 1.0}
    assert res["web"]["no_web_tool"] == {"n": 1, "correct": 0}
    assert res["repeat"] == {"run": str(rep), "shared": 2, "same": 1, "right_before": 1,
                             "right_now": 2}
    md = (web / "WEB_ANALYSIS.md").read_text(encoding="utf-8")
    assert "| web | 2 = 50.0% |" in md and "| SUPPORTS | **1** | 0 | 0 | 1 | 2 |" in md
    with pytest.raises(SystemExit):
        wa.main(["--web-run", str(web), "--claims", str(claims_file), "--claims-sha256", "0" * 64])


def test_a_row_whose_label_disagrees_with_the_claims_file_is_refused(tmp_path):
    claims_file = tmp_path / "claims.jsonl"
    write_jsonl(claims_file, [{"id": 1, "label": S, "kind": "single"}])
    write_jsonl(tmp_path / "web" / "results.jsonl", [row(1, "web", R, R)])
    with pytest.raises(SystemExit):
        wa.main(["--web-run", str(tmp_path / "web"), "--claims", str(claims_file)])


def test_sessions_accept_rows_with_the_same_interval():
    rows = [row(1, "web", S, S), row(2, "web", S, R)]          # same time stamp and seconds
    assert wa.sessions(rows)[0]["rows"] == 2
