#!/usr/bin/env python3
"""Compare the web condition with the three earlier conditions on the same claims.

Reads the run folders (``results.jsonl``; for each item key the newest final row) and the claims
file (the label and kind of every claim; its sha256 is checked). For every condition: accuracy
over all claims, per label and per kind, the confusion matrix (rows: FEVEROUS label, columns: the
answer, "no answer" for a claim without a final ``ok`` answer), F1 per label and macro F1 (the
definitions of ``scaled_data/HAIKU_RUN_2026-11-12.md``), seconds per call, and tokens per call.
For the web condition also: the time calls were running (the union of the calls' intervals), the
run's sessions with their usage-window readings, the web tool use and accuracy by tool use, the
failures, and the audit summary (``web_audit.json``, from ``scripts/llm_web_audit.py``). Paired
comparison of web with each other condition: claims right in one and wrong in the other, and the
exact two-sided McNemar test. Optionally the agreement with an earlier web run on shared claims.

Nothing is estimated: every number is counted from the rows.

Writes ``<out_dir>/WEB_ANALYSIS.md`` and ``<out_dir>/web_analysis.json``.

Usage: .venv/bin/python scripts/llm_web_analysis.py --web-run <dir> --old-run <dir> [--old-run ...]
           --claims <claims.jsonl> [--claims-sha256 <hex>] [--repeat-run <dir>] [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from wikilense.llm.runner import load_rows

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
SHORT = {"SUPPORTS": "S", "REFUTES": "R", "NOT ENOUGH INFO": "NEI"}
NO_ANSWER = "no answer"
KINDS = ("single", "needs2")
CONDITION_ORDER = ("no_retrieval", "retrieved", "gold", "web")
SESSION_GAP_S = 600.0


# ---------------------------------------------------------------- reading
def load_claims(path: Path, sha256: str | None = None) -> dict[int, dict[str, str]]:
    """Return {claim id: {"label", "kind"}} from a claims file, after checking its sha256."""
    data = path.read_bytes()
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{path}: sha256 differs from {sha256}")
    claims = {}
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            claims[int(c["id"])] = {"label": c["label"], "kind": c["kind"]}
    return claims


def final_rows(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    """Return {condition: {claim id: row}}: the newest final row of each item key (file order is
    the order of writing)."""
    newest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("final"):
            newest[row["key"]] = row
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in newest.values():
        cid = int(row["claim_id"])
        if cid in out[row["condition"]]:
            raise SystemExit(f"claim {cid} has two final keys in condition {row['condition']}")
        out[row["condition"]][cid] = row
    return dict(out)


def answer(row: dict[str, Any] | None) -> str:
    """Return the row's label, or NO_ANSWER when there is no final ``ok`` answer."""
    if row is None or row.get("status") != "ok" or row.get("label") not in LABELS:
        return NO_ANSWER
    return row["label"]


# ---------------------------------------------------------------- scores
def confusion(items: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Return {gold: {answer: count}} over (gold, answer) pairs, every cell present."""
    table = {g: dict.fromkeys((*LABELS, NO_ANSWER), 0) for g in LABELS}
    for gold, ans in items:
        table[gold][ans] += 1
    return table


def f1_scores(items: list[tuple[str, str]]) -> dict[str, float]:
    """Return F1 per label and their mean ("macro"). A missing answer is a miss for its gold label
    and a false positive for none."""
    out = {}
    for label in LABELS:
        tp = sum(1 for g, a in items if g == label and a == label)
        predicted = sum(1 for _, a in items if a == label)
        actual = sum(1 for g, _ in items if g == label)
        p = tp / predicted if predicted else 0.0
        r = tp / actual if actual else 0.0
        out[label] = 2 * p * r / (p + r) if p + r else 0.0
    out["macro"] = sum(out[label] for label in LABELS) / len(LABELS)
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Return the exact two-sided McNemar p-value for b and c discordant pairs (binomial, p=1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def fmt_p(p: float) -> str:
    """Return a p-value for the report; one below the smallest float shows as < 1e-300."""
    return "< 1e-300" if p == 0.0 else f"{p:.3g}"


def percentile(values: list[float], q: float) -> float:
    """Return the nearest-rank q-quantile (0 < q <= 1) of a non-empty list."""
    xs = sorted(values)
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def dist(values: list[float]) -> dict[str, float] | None:
    """Return n, mean, median, p90, p95, max, sum of the values (None for no values)."""
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p90": percentile(values, 0.90), "p95": percentile(values, 0.95), "max": max(values),
            "sum": sum(values)}


def active_seconds(intervals: list[tuple[float, float]]) -> float:
    """Return the length of the union of (start, end) intervals."""
    total, end = 0.0, -math.inf
    for s, e in sorted(intervals):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def row_interval(row: dict[str, Any]) -> tuple[float, float]:
    """Return a row's (start, end) in epoch seconds: end = its time stamp, start = end minus the
    item's total seconds (all attempts)."""
    end = datetime.fromisoformat(row["ts"]).timestamp()
    return end - float(row.get("elapsed_s_total") or row.get("latency_s") or 0.0), end


def sessions(rows: list[dict[str, Any]], gap_s: float = SESSION_GAP_S) -> list[dict[str, Any]]:
    """Split the rows into sessions where no call ran for more than ``gap_s`` seconds."""
    spans = sorted(((row_interval(r), r) for r in rows), key=lambda item: item[0])
    out: list[dict[str, Any]] = []
    for (s, e), r in spans:
        if out and s - out[-1]["_end"] <= gap_s:
            cur = out[-1]
            cur["_end"] = max(cur["_end"], e)
            cur["rows"].append(r)
        else:
            out.append({"_start": s, "_end": e, "rows": [r]})
    result = []
    for cur in out:
        rs = sorted(cur["rows"], key=lambda r: r["ts"])
        windows = [r.get("window_utilization") for r in rs if r.get("window_utilization")]
        result.append({
            "start": datetime.fromtimestamp(cur["_start"]).astimezone().isoformat(timespec="seconds"),
            "end": datetime.fromtimestamp(cur["_end"]).astimezone().isoformat(timespec="seconds"),
            "seconds": cur["_end"] - cur["_start"],
            "active_seconds": active_seconds([row_interval(r) for r in rs]),
            "rows": len(rs), "final_rows": sum(1 for r in rs if r.get("final")),
            "window_first": windows[0] if windows else None,
            "window_last": windows[-1] if windows else None})
    return result


def tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return sums and per-call medians of the rows' computed usage (the final attempt's)."""
    fields = ("prompt_tokens", "cache_read_tokens", "cache_creation_tokens",
              "input_tokens_uncached", "output_tokens", "thinking_tokens", "side_model_tokens")
    usage = [r.get("computed_usage") or {} for r in rows]
    return {f: {"sum": sum(u.get(f) or 0 for u in usage),
                "median": statistics.median([u.get(f) or 0 for u in usage]) if usage else None}
            for f in fields}


# ---------------------------------------------------------------- one condition
def score_condition(rows_by_claim: dict[int, dict[str, Any]],
                    claims: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Return every score of one condition over all claims of the claims file."""
    for cid, row in rows_by_claim.items():
        if cid not in claims:
            raise SystemExit(f"claim {cid} is not in the claims file")
        if row.get("gold_label") != claims[cid]["label"]:
            raise SystemExit(f"claim {cid}: row label {row.get('gold_label')} != claims file")
    items = {cid: (c["label"], answer(rows_by_claim.get(cid))) for cid, c in claims.items()}
    pairs = list(items.values())
    n_correct = sum(1 for g, a in pairs if g == a)
    by_kind = {}
    for kind in KINDS:
        sub = [items[cid] for cid, c in claims.items() if c["kind"] == kind]
        by_kind[kind] = {"n": len(sub), "correct": sum(1 for g, a in sub if g == a)}
    by_label = {}
    for label in LABELS:
        sub = [p for p in pairs if p[0] == label]
        by_label[label] = {"n": len(sub), "correct": sum(1 for g, a in sub if g == a)}
    rows = list(rows_by_claim.values())
    return {
        "claims": len(claims), "final_rows": len(rows),
        "answered": sum(1 for _, a in pairs if a != NO_ANSWER),
        "status": dict(Counter(r.get("status") for r in rows)),
        "error_kinds": dict(Counter(r.get("error_kind") for r in rows if r.get("error_kind"))),
        "correct": n_correct, "accuracy": n_correct / len(claims),
        "by_kind": by_kind, "by_label": by_label,
        "confusion": confusion(pairs), "f1": f1_scores(pairs),
        "latency_s": dist([float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]),
        "elapsed_s_total": dist([float(r["elapsed_s_total"]) for r in rows
                                 if r.get("elapsed_s_total") is not None]),
        "attempts": dict(Counter(r.get("attempts") for r in rows)),
        "tokens": tokens(rows),
        "_correct_ids": sorted(cid for cid, (g, a) in items.items() if g == a),
    }


def web_details(all_rows: list[dict[str, Any]], final: dict[int, dict[str, Any]],
                claims: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Return the web-only figures: run time, sessions, tool use, accuracy by tool use."""
    web_rows = [r for r in all_rows if r.get("condition") == "web"]
    rows = list(final.values())

    def acc(sub: list[dict[str, Any]]) -> dict[str, int]:
        return {"n": len(sub),
                "correct": sum(1 for r in sub if answer(r) == claims[int(r["claim_id"])]["label"])}

    searches = [int(r.get("n_web_search") or 0) for r in rows]
    by_search: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r, s in zip(rows, searches, strict=True):
        by_search["0" if s == 0 else "1" if s == 1 else "2" if s == 2 else "3+"].append(r)
    no_tool = [r for r in rows if not r.get("n_web_search") and not r.get("n_web_fetch")]
    tool = [r for r in rows if r.get("n_web_search") or r.get("n_web_fetch")]
    return {
        "rows_in_file": len(web_rows), "non_final_rows": sum(1 for r in web_rows if not r.get("final")),
        "active_seconds": active_seconds([row_interval(r) for r in web_rows]),
        "sessions": sessions(web_rows),
        "turns": dist([float(r.get("num_turns") or 0) for r in rows]),
        "web_search": dist([float(s) for s in searches]),
        "web_fetch": dist([float(r.get("n_web_fetch") or 0) for r in rows]),
        "denied": sum(int(r.get("n_denied") or 0) for r in rows),
        "assistant_models": dict(Counter(m for r in rows for m in r.get("assistant_models") or [])),
        "no_web_tool": acc(no_tool), "web_tool_used": acc(tool),
        "by_searches": {k: acc(v) for k, v in sorted(by_search.items())},
        "by_kind_and_label": {f"{SHORT[label]}/{kind}": acc(
            [r for r in rows if claims[int(r["claim_id"])] == {"label": label, "kind": kind}])
            for label in LABELS for kind in KINDS},
    }


def paired(a_ids: set[int], b_ids: set[int], all_ids: set[int]) -> dict[str, Any]:
    """Return both right / only a / only b / both wrong and the exact McNemar p-value."""
    only_a, only_b = len(a_ids - b_ids), len(b_ids - a_ids)
    return {"both_right": len(a_ids & b_ids), "only_web": only_a, "only_other": only_b,
            "both_wrong": len(all_ids - a_ids - b_ids), "mcnemar_p": mcnemar_exact(only_a, only_b)}


# ---------------------------------------------------------------- report
def pct(n: int, d: int) -> str:
    return f"{n:,} = {100 * n / d:.1f}%" if d else "-"


def fmt_dist(d: dict[str, float] | None, digits: int = 1) -> str:
    if not d:
        return "-"
    return " / ".join(f"{d[k]:,.{digits}f}" for k in ("median", "p90", "p95", "max"))


def markdown(res: dict[str, Any]) -> str:
    conds = [c for c in CONDITION_ORDER if c in res["conditions"]]
    sc = res["conditions"]
    n = res["claims"]["n"]
    L = [f"# The web condition compared with the earlier conditions ({res['written_at'][:10]})", "",
         (f"Claims: `{res['claims']['path']}` ({n:,} claims, sha256 `{res['claims']['sha256'][:16]}…`). "
         "Haiku 5.5 in every run. Computed by `scripts/llm_web_analysis.py` from the rows of:"), ""]
    L += [f"- `{c}`: `{res['runs'][c]}`" for c in conds]
    L += ["", "## 1. Accuracy", "",
          ("Accuracy = the answer equals the FEVEROUS label; a claim without a final answer counts as "
          "wrong. F1 per label and macro F1 as in `scaled_data/HAIKU_RUN_2026-11-12.md`."), "",
          ("| condition | all | single | needs2 | SUPPORTS | REFUTES | NOT ENOUGH INFO | macro F1 | "
          "F1 S / R / NEI | answered |"), "|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        s = sc[c]
        L.append(f"| {c} | {pct(s['correct'], n)} | "
                 + " | ".join(pct(s["by_kind"][k]["correct"], s["by_kind"][k]["n"]) for k in KINDS)
                 + " | " + " | ".join(pct(s["by_label"][lab]["correct"], s["by_label"][lab]["n"])
                                      for lab in LABELS)
                 + f" | {s['f1']['macro']:.3f} | "
                 + " / ".join(f"{s['f1'][lab]:.3f}" for lab in LABELS)
                 + f" | {s['answered']:,} |")
    L += ["", "## 2. Confusion matrices", "",
          "Rows: the FEVEROUS label; columns: the answer. The diagonal (bold) is right.", ""]
    for c in conds:
        t = sc[c]["confusion"]
        L += [f"**{c}**", "", "| label \\ answer | SUPPORTS | REFUTES | NOT ENOUGH INFO | no answer | total |",
              "|---|---|---|---|---|---|"]
        for g in LABELS:
            cells = [f"**{t[g][a]:,}**" if a == g else f"{t[g][a]:,}" for a in (*LABELS, NO_ANSWER)]
            L.append(f"| {g} | " + " | ".join(cells) + f" | {sum(t[g].values()):,} |")
        col = [sum(t[g][a] for g in LABELS) for a in (*LABELS, NO_ANSWER)]
        L += ["| total | " + " | ".join(f"{x:,}" for x in col) + f" | {sum(col):,} |", ""]
    if res.get("paired"):
        L += ["## 3. Web against each other condition, claim by claim", "",
              "Exact two-sided McNemar test on the claims where exactly one of the two is right.", "",
              "| web vs | both right | only web right | only the other right | both wrong | McNemar p |",
              "|---|---|---|---|---|---|"]
        for c, p in res["paired"].items():
            L.append(f"| {c} | {p['both_right']:,} | {p['only_web']:,} | {p['only_other']:,} | "
                     f"{p['both_wrong']:,} | {fmt_p(p['mcnemar_p'])} |")
        L.append("")
    L += ["## 4. Seconds per call", "",
          ("Per claim, the final attempt (`latency_s`) and all attempts (`elapsed_s_total`); "
          "median / p90 / p95 / max (nearest rank)."), "",
          "| condition | seconds, final attempt | seconds, all attempts | attempts |",
          "|---|---|---|---|"]
    for c in conds:
        s = sc[c]
        L.append(f"| {c} | {fmt_dist(s['latency_s'])} | {fmt_dist(s['elapsed_s_total'])} | "
                 + ", ".join(f"{k}: {v:,}" for k, v in sorted(s["attempts"].items(),
                                                             key=lambda kv: str(kv[0]))) + " |")
    w = res.get("web")
    if w:
        L += ["", (f"Web run: calls were running for {w['active_seconds'] / 3600:.2f} h in total "
              f"(the union of the calls' intervals, 5 workers); {w['rows_in_file']:,} rows in the "
              f"file, {w['non_final_rows']:,} of them not final (interrupted or retried calls)."), "",
              ("| session start | end | hours running | rows | final | window first (5 h / week) | "
              "window last |"), "|---|---|---|---|---|---|---|"]
        for s in w["sessions"]:
            def win(x: dict[str, float] | None) -> str:
                return f"{x.get('five_hour')} / {x.get('seven_day')}" if x else "-"
            L.append(f"| {s['start']} | {s['end'][11:]} | {s['active_seconds'] / 3600:.2f} | "
                     f"{s['rows']:,} | {s['final_rows']:,} | {win(s['window_first'])} | "
                     f"{win(s['window_last'])} |")
    L += ["", "## 5. Tokens per call", "",
          "Sum over the claims (the final attempt's `computed_usage`); median per call in brackets.",
          "", ("| condition | prompt | cache read | cache write | uncached input | output | of which "
          "thinking |"), "|---|---|---|---|---|---|---|"]
    for c in conds:
        t = sc[c]["tokens"]
        L.append(f"| {c} | " + " | ".join(
            f"{t[f]['sum']:,} ({t[f]['median']:,.0f})" for f in (
                "prompt_tokens", "cache_read_tokens", "cache_creation_tokens",
                "input_tokens_uncached", "output_tokens", "thinking_tokens")) + " |")
    if w:
        L += ["", "## 6. Web tool use", "",
              (f"- Turns per call: {fmt_dist(w['turns'], 0)} (median / p90 / p95 / max); WebSearch "
              f"{fmt_dist(w['web_search'], 0)}, sum {w['web_search']['sum']:,.0f}; WebFetch "
              f"{fmt_dist(w['web_fetch'], 0)}, sum {w['web_fetch']['sum']:,.0f}."),
              (f"- Denied fetches: {w['denied']:,}. Models in the assistant messages: "
              f"{w['assistant_models']}."),
              (f"- No web tool used: {w['no_web_tool']['n']:,} claims, "
              f"{pct(w['no_web_tool']['correct'], w['no_web_tool']['n'])} right; a web tool used: "
              f"{w['web_tool_used']['n']:,}, {pct(w['web_tool_used']['correct'], w['web_tool_used']['n'])}."),
              f"- Failures: status {sc['web']['status']}, error kinds {sc['web']['error_kinds'] or 'none'}.",
              "", "| WebSearch calls | claims | right |", "|---|---|---|"]
        for k, v in w["by_searches"].items():
            L.append(f"| {k} | {v['n']:,} | {pct(v['correct'], v['n'])} |")
        L += ["", "| label / kind | claims | right |", "|---|---|---|"]
        for k, v in w["by_kind_and_label"].items():
            L.append(f"| {k} | {v['n']:,} | {pct(v['correct'], v['n'])} |")
    if res.get("repeat"):
        r = res["repeat"]
        L += ["", "## 7. The same claims asked twice", "",
              (f"`{r['run']}` and this run share {r['shared']:,} claims: the same answer for "
              f"{r['same']:,}; right in the earlier run {r['right_before']:,}, in this run "
              f"{r['right_now']:,}.")]
    if res.get("audit"):
        a = res["audit"]
        top = ", ".join(f"{h} {c:,}" for h, c in list(a["fetched_hosts"].items())[:8])
        L += ["", "## 8. Audit (`web_audit.json`)", "",
              (f"{a['web_rows']:,} rows audited at {a['audited_at']}; {len(a['flagged']):,} flagged; "
              f"{a['denied_calls']:,} denied; {a['rows_without_raw_stream']:,} without a raw stream. "
              f"Fetched hosts (top 8): {top}.")]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--web-run", required=True, type=Path)
    p.add_argument("--old-run", action="append", default=[], type=Path)
    p.add_argument("--claims", required=True, type=Path)
    p.add_argument("--claims-sha256", default=None)
    p.add_argument("--repeat-run", type=Path, default=None, help="an earlier web run to compare")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    claims = load_claims(args.claims, args.claims_sha256)
    conditions: dict[str, dict[int, dict[str, Any]]] = {}
    runs: dict[str, str] = {}
    web_all_rows = load_rows(args.web_run / "results.jsonl")
    for run_dir, rows in [(args.web_run, web_all_rows),
                          *[(d, load_rows(d / "results.jsonl")) for d in args.old_run]]:
        for cond, by_claim in final_rows(rows).items():
            if cond in conditions:
                raise SystemExit(f"condition {cond} appears in two run folders")
            conditions[cond], runs[cond] = by_claim, str(run_dir)
    if "web" not in conditions:
        raise SystemExit(f"{args.web_run} has no final web rows")
    scores = {c: score_condition(conditions[c], claims) for c in conditions}
    ids = set(claims)
    web_ok = set(scores["web"]["_correct_ids"])
    res: dict[str, Any] = {
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "claims": {"path": str(args.claims), "n": len(claims),
                   "sha256": hashlib.sha256(args.claims.read_bytes()).hexdigest()},
        "runs": runs, "conditions": scores,
        "paired": {c: paired(web_ok, set(scores[c]["_correct_ids"]), ids)
                   for c in CONDITION_ORDER if c in scores and c != "web"},
        "web": web_details(web_all_rows, conditions["web"], claims)}
    if args.repeat_run:
        before = final_rows(load_rows(args.repeat_run / "results.jsonl")).get("web", {})
        shared = sorted(set(before) & set(conditions["web"]))
        res["repeat"] = {
            "run": str(args.repeat_run), "shared": len(shared),
            "same": sum(1 for c in shared if answer(before[c]) == answer(conditions["web"][c])),
            "right_before": sum(1 for c in shared if answer(before[c]) == claims[c]["label"]),
            "right_now": sum(1 for c in shared
                             if answer(conditions["web"][c]) == claims[c]["label"])}
    audit = args.web_run / "web_audit.json"
    if audit.exists():
        res["audit"] = json.loads(audit.read_text(encoding="utf-8"))
    out = args.out_dir or args.web_run
    out.mkdir(parents=True, exist_ok=True)
    (out / "WEB_ANALYSIS.md").write_text(markdown(res), encoding="utf-8")
    for s in scores.values():
        s.pop("_correct_ids")
    (out / "web_analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n",
                                           encoding="utf-8")
    print(f"wrote {out / 'WEB_ANALYSIS.md'} and {out / 'web_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
