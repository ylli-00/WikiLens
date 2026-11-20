"""Per-claim table and statistics of the LLM fact-checker runs, written to ``results/llm/``.

Two steps:

``table`` reads the run folders, which stay outside the repository (rows, manifests and the
retrieval cache), and writes the committed inputs:

- ``per_claim.csv``: one row per claim, condition and run, with the columns ``claim_id``,
  ``label``, ``kind``, ``condition``, ``run`` (``original``, ``r1`` or ``r2``), ``answer`` and
  ``right`` (1 when the answer equals the FEVEROUS label);
- ``retrieval_top5.csv``: per claim, whether the five passages of the RAG condition came from a
  page in ``gold_pages`` and from every page of the evidence set shown in the gold condition;
- ``runs.json``: per run folder, the CLI version, the calls, seconds and tokens per condition,
  the usage-window readings, and the check of every repeat row's user text against the row of
  the original run.

``stats`` reads only those three files, three database result files under ``results/scaled/``
and the claims file ``data/scaled/claims_3000.jsonl`` (for the FEVEROUS challenge tag; its
SHA-256 is checked first), and writes ``llm_results.json`` (every number) and ``LLM_RESULTS.md``
(the summary report). Every number in the report can therefore be recomputed from committed
files.

Conventions: a claim is right when the answer equals the FEVEROUS label, so a NOT ENOUGH INFO
answer to a supported or refuted claim is wrong. Percentages are rounded to one decimal, half
up, with :mod:`decimal`. Wilson intervals and the paired-difference interval use z = 1.96. The
bootstraps resample claims with replacement, the same resamples for every condition (paired),
:data:`BOOTSTRAP_RESAMPLES` times, from ``numpy.random.default_rng`` with the seeds in
:data:`SEEDS`; an interval is the 2.5th to 97.5th percentile of the resampled values.

    .venv/bin/python scripts/llm_results.py table --runs-root <llm_runs> --claims <claims_3000.jsonl>
    .venv/bin/python scripts/llm_results.py stats
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "llm"
SCALED = REPO_ROOT / "results" / "scaled"
WEIGHTS_FILE = SCALED / "breakdown" / "breakdown.json"
SQL_TIMING_FILE = SCALED / "run1" / "s1" / "separate_index.json"
EMBED_TIMING_FILE = SCALED / "eval" / "s1_full_run1.json"
CLAIMS_FILE = REPO_ROOT / "data" / "scaled" / "claims_3000.jsonl"
CLAIMS_SHA256 = "905cf1281d01e310702e820121864ac96ab0c07d5e857c24ad2d1314f532ab8a"

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
NEI = "NOT ENOUGH INFO"
SIDES = ("SUPPORTS", "REFUTES")
KINDS = ("single", "needs2")
#: Condition ids as the runner writes them, in the order of every table.
CONDITIONS = ("no_retrieval", "web", "retrieved", "gold")
NAMES = {"no_retrieval": "no passages", "web": "web", "retrieved": "RAG", "gold": "gold"}
#: The six paired comparisons: (first, second); a difference is first minus second.
PAIRS = (("retrieved", "web"), ("retrieved", "gold"), ("gold", "web"),
         ("retrieved", "no_retrieval"), ("gold", "no_retrieval"), ("web", "no_retrieval"))
RUNS = ("original", "r1", "r2")
#: The paired comparisons of the multi-hop part, within each kind: RAG minus the other condition.
RAG_PAIRS = (("retrieved", "no_retrieval"), ("retrieved", "gold"), ("retrieved", "web"))
#: The two FEVEROUS challenge tags compared in the multi-hop part, and the size below which a
#: tag's figures are given without a conclusion.
MULTIHOP_TAG, OTHER_TAG = "Multi-hop Reasoning", "Other"
SMALL_TAG = 100
#: Run folder of each condition; a repeat folder is the same name plus ``_rep450_<run>``.
RUN_FOLDERS = {"no_retrieval": "haiku_high_fact_check_v1_noret_gold",
               "gold": "haiku_high_fact_check_v1_noret_gold",
               "retrieved": "haiku_high_fact_check_v1_retrieved",
               "web": "haiku_high_fact_check_web_v1_web"}
RETRIEVAL_CACHE = Path("retrieval_cache/wikilense_s1/k5_inline_ef100_BAAI_bge-small-en-v1.5")
N_CLAIMS, N_REPEAT = 3000, 450
Z = 1.96
BOOTSTRAP_RESAMPLES = 10_000
#: Seeds of the resamples of the 3,000 claims and of the 2,000 supported/refuted claims.
SEEDS = {"all": 20261120, "sides": 20261121}
BLOCK = 500
TOKEN_FIELDS = ("prompt_tokens", "cache_read_tokens", "cache_creation_tokens",
                "input_tokens_uncached", "output_tokens", "thinking_tokens")
CSV_COLUMNS = ("claim_id", "label", "kind", "condition", "run", "answer", "right")


# ----------------------------------------------------------------------------------------------
# Rounding and the closed-form statistics
# ----------------------------------------------------------------------------------------------
def r1(value: float | Fraction | Decimal) -> str:
    """Return ``value`` rounded to one decimal, half up (``Fraction`` and ``Decimal`` exactly)."""
    if isinstance(value, Fraction):
        value = Decimal(value.numerator) / Decimal(value.denominator)
    elif not isinstance(value, Decimal):
        value = Decimal(repr(float(value)))
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def pct(num: int, den: int) -> str:
    """Return ``num / den`` in percent, one decimal, half up, computed exactly."""
    return r1(Fraction(100 * num, den))


def p_text(p: Fraction) -> str:
    """Return a p-value with two significant digits, half up (scientific below 0.001)."""
    d = Decimal(p.numerator) / Decimal(p.denominator)
    if d == 0:
        return "0"
    exponent = d.adjusted()
    q = d.scaleb(-exponent).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if q >= 10:
        q, exponent = q / 10, exponent + 1
    if exponent >= -3:
        return str((q.scaleb(exponent)).quantize(Decimal(1).scaleb(exponent - 1)))
    return f"{q}e{exponent}"


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Return the Wilson score interval of ``k`` successes in ``n`` trials, as shares."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half, centre + half)


def mcnemar_exact(b: int, c: int) -> Fraction:
    """Return the exact two-sided McNemar p: twice the binomial tail at ``min(b, c)``, capped at 1.

    Under the null each of the ``b + c`` discordant claims goes either way with probability
    one half; the sum is done in integers, so very small p-values stay exact.
    """
    n = b + c
    if n == 0:
        return Fraction(1)
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(Fraction(1), Fraction(2 * tail, 2 ** n))


def paired_interval(b: int, c: int, n: int, z: float = Z) -> tuple[float, float, float]:
    """Return (difference, low, high) of two paired accuracies as shares:
    ``(b - c) / n ± z * sqrt(b + c - (b - c)**2 / n) / n``."""
    diff = (b - c) / n
    half = z * math.sqrt(b + c - (b - c) ** 2 / n) / n
    return diff, diff - half, diff + half


def newcombe(k1: int, n1: int, k2: int, n2: int, z: float = Z) -> tuple[float, float, float]:
    """Return (difference, low, high) of ``k1 / n1 - k2 / n2`` for two independent sets of
    claims, as shares: Newcombe's hybrid score interval, built from the two Wilson intervals
    (Newcombe 1998, method 10)."""
    if n1 == 0 or n2 == 0:
        return (math.nan, math.nan, math.nan)
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1, z), wilson(k2, n2, z)
    diff = p1 - p2
    return (diff, diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def prf(gold: np.ndarray, pred: np.ndarray, label: int) -> tuple[float, float, float]:
    """Return precision, recall and F1 of one label (codes are indexes into :data:`LABELS`)."""
    tp = int(np.sum((gold == label) & (pred == label)))
    fp = int(np.sum((gold != label) & (pred == label)))
    fn = int(np.sum((gold == label) & (pred != label)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def f1_from_counts(cells: np.ndarray, label: int) -> np.ndarray:
    """F1 of one label from confusion counts ``cells[..., gold, pred]`` (any leading axes)."""
    tp = cells[..., label, label]
    fp = cells[..., :, label].sum(axis=-1) - tp
    fn = cells[..., label, :].sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def rerun_sd(f: float, n: int) -> float:
    """Rerun standard deviation of an accuracy over ``n`` claims when a share ``f`` of the claims
    switches between right and wrong across two runs: ``sqrt(f / (2 n))``."""
    return math.sqrt(f / (2 * n))


# ----------------------------------------------------------------------------------------------
# table: from the run folders to the committed inputs
# ----------------------------------------------------------------------------------------------
def load_claims(path: Path) -> dict[int, dict[str, Any]]:
    """Return the claims file as ``{id: record}``."""
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                c = json.loads(line)
                out[c["id"]] = c
    return out


def final_rows(run_dir: Path, condition: str) -> dict[int, dict[str, Any]]:
    """Return the final rows of one condition of a run folder, ``{claim_id: row}``."""
    out: dict[int, dict[str, Any]] = {}
    with (run_dir / "results.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("final") and row["condition"] == condition:
                if row["claim_id"] in out:
                    raise SystemExit(f"{run_dir.name}: two final rows for claim "
                                     f"{row['claim_id']} {condition}")
                out[row["claim_id"]] = row
    return out


def all_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Return every row of a run folder in file order."""
    with (run_dir / "results.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def quantile(values: list[float], q: float) -> float | None:
    """Return the nearest-rank quantile ``q`` (0..1) of ``values``, None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def window_use(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Return how much of one usage window a run's rows used, from the readings in each row.

    A row's rate-limit event names the window's reset time, which identifies the window. Per
    window, the use is the highest reading minus the first one; ``used`` sums the windows.
    Parallel calls report slightly out of order, which taking the highest reading absorbs. The
    readings cover the whole account, so other use of the subscription at the same time is
    included, and the use before the first reading of each window is not.
    """
    windows: dict[Any, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        event = ((r.get("cli") or {}).get("last_rate_limit_event") or {}).get("rate_limit_info")
        w = ((event or {}).get("unifiedWindows") or {}).get(name) or {}
        value = w.get("utilization")
        if value is None:
            value = (r.get("window_utilization") or {}).get(name)
        if value is not None:
            windows[w.get("resetsAt")].append((r["ts"], float(value)))
    segments = []
    for reset in sorted(windows, key=lambda k: min(windows[k])[0]):
        readings = sorted(windows[reset])
        segments.append({"resets_at": reset, "first": readings[0][1],
                         "highest": max(v for _, v in readings), "readings": len(readings)})
    used = sum(seg["highest"] - seg["first"] for seg in segments)
    return {"used": round(used, 4), "windows": len(segments), "segments": segments}


def condition_costs(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Seconds and tokens of one condition's final rows (the final attempt's numbers)."""
    lat = [r["latency_s"] for r in rows.values() if r.get("latency_s") is not None]
    tot = [r["elapsed_s_total"] for r in rows.values() if r.get("elapsed_s_total") is not None]
    out: dict[str, Any] = {
        "calls": len(rows),
        "attempts": dict(sorted(Counter(r["attempts"] for r in rows.values()).items())),
        "seconds_final_attempt": {"median": statistics.median(lat), "p95": quantile(lat, 0.95),
                                  "max": max(lat)},
        "seconds_all_attempts": {"median": statistics.median(tot), "p95": quantile(tot, 0.95),
                                 "max": max(tot)},
        "tokens_sum": {}, "tokens_median": {},
    }
    for name in TOKEN_FIELDS:
        vals = [int((r.get("computed_usage") or {}).get(name) or 0) for r in rows.values()]
        out["tokens_sum"][name] = sum(vals)
        out["tokens_median"][name] = statistics.median(vals)
    if any(r["condition"] == "web" for r in rows.values()):
        out["web"] = {
            "searches_sum": sum(int(r.get("n_web_search") or 0) for r in rows.values()),
            "fetches_sum": sum(int(r.get("n_web_fetch") or 0) for r in rows.values()),
            "denied_sum": sum(int(r.get("n_denied") or 0) for r in rows.values()),
            "turns_median": statistics.median(int(r.get("num_turns") or 0)
                                              for r in rows.values()),
            "claims_without_a_tool_call": sum(1 for r in rows.values()
                                              if not (r.get("n_web_search") or r.get("n_web_fetch"))),
        }
    return out


def run_record(run_dir: Path, conditions: tuple[str, ...]) -> tuple[dict[str, Any], dict]:
    """Return (the runs.json entry of one run folder, its final rows per condition)."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = all_rows(run_dir)
    finals = {c: final_rows(run_dir, c) for c in conditions}
    record = {
        "run_folder": run_dir.name,
        "conditions": list(conditions),
        "model": sorted(set(manifest["models"].values())),
        "effort": manifest["effort"],
        "prompt_id": manifest["prompt"]["prompt_id"],
        "prompt_hash": manifest["prompt"]["prompt_hash"],
        "backend": manifest["backend"],
        "cli_version": (manifest.get("backend_info") or {}).get("version"),
        "cli_binary_sha256": (manifest.get("backend_info") or {}).get("binary_sha256"),
        "invocations": [{"started_at": i["started_at"], "ended_at": i.get("ended_at"),
                         "backend_version": i.get("backend_version"),
                         "exit_code": i.get("exit_code"), "stop_reason": i.get("stop_reason"),
                         "workers": i.get("workers"), "thresholds": i.get("thresholds"),
                         "counts": i.get("counts")}
                        for i in manifest.get("invocations", [])],
        "rows_in_file": len(rows),
        "rows_not_final": sum(1 for r in rows if not r.get("final")),
        "retrieval": {k: (manifest.get("retrieval") or {}).get(k)
                      for k in ("database", "k", "strategy", "ef_search", "embedding_model",
                                "embedding_revision", "device", "n_chunks_at_start")},
        "web": ({k: manifest["web"].get(k) for k in ("tools", "deny_hosts", "deny_hosts_sha256",
                                                      "max_turns", "timeout_s")}
                if manifest.get("web") else None),
        "window_five_hour": window_use(rows, "five_hour"),
        "window_seven_day": window_use(rows, "seven_day"),
        "per_condition": {c: condition_costs(finals[c]) for c in conditions},
    }
    return record, finals


def gold_set_pages(claim: dict[str, Any]) -> set[str]:
    """Return the pages of the evidence set shown in the gold condition (NFC)."""
    from wikilense.corpus import parse_element_id

    elements = claim["evidence_sets"][claim["gold_set"]]["elements"]
    return {unicodedata.normalize("NFC", parse_element_id(e)[0]) for e in elements}


def nei_audit_summary(path: Path) -> dict[str, Any]:
    """Counts of the reviewed NOT ENOUGH INFO audit (two reviewer agents per case, a third on
    disagreement): its agreement statistics and, per condition, the final field values."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    by_cond: dict[str, dict[str, Any]] = {}
    for case in audit["finals"]:
        cond = case["case_id"].split(":")[0]
        entry = by_cond.setdefault(cond, {"cases": 0, "error": Counter(), "settle": Counter()})
        entry["cases"] += 1
        entry["error"][case["error"]] += 1
        entry["settle"][case["settle"]] += 1
    return {"stats": audit["stats"],
            "by_condition": {c: {"cases": e["cases"], "error": dict(e["error"].most_common()),
                                 "settle": dict(e["settle"].most_common())}
                             for c, e in sorted(by_cond.items())}}


def web_audit_summary(path: Path) -> dict[str, Any]:
    """Counts of the web audit: rows, tool calls, denied calls, flagged rows and fetched hosts."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    hosts = audit.get("fetched_hosts") or {}
    return {"audited_at": audit.get("audited_at"), "web_rows": audit.get("web_rows"),
            "web_search_calls": audit.get("web_search_calls"),
            "web_fetch_calls": audit.get("web_fetch_calls"),
            "denied_calls": audit.get("denied_calls"),
            "flagged_rows": len(audit.get("flagged") or []),
            "fetches_en_wikipedia": hosts.get("en.wikipedia.org"),
            "top_fetched_hosts": dict(list(hosts.items())[:8])}


def cmd_table(args: argparse.Namespace) -> int:
    """Write per_claim.csv, retrieval_top5.csv and runs.json from the run folders."""
    claims = load_claims(args.claims)
    if len(claims) != N_CLAIMS:
        raise SystemExit(f"{args.claims}: {len(claims)} claims, not {N_CLAIMS}")
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    folders: dict[str, dict[str, Path]] = {run: {} for run in RUNS}
    for cond, name in RUN_FOLDERS.items():
        folders["original"][cond] = args.runs_root / name
        for run in [r for r in args.runs if r != "original"]:
            path = args.runs_root / f"{name}_rep450_{run}"
            if (path / "results.jsonl").is_file():
                folders[run][cond] = path
    table: list[dict[str, Any]] = []
    runs_json: dict[str, Any] = {"runs": [], "user_text_check": {}}
    originals: dict[str, dict[int, dict[str, Any]]] = {}
    for run in [r for r in RUNS if r in args.runs]:
        by_folder: dict[Path, list[str]] = defaultdict(list)
        for cond, path in folders[run].items():
            by_folder[path].append(cond)
        for path, conds in by_folder.items():
            record, finals = run_record(path, tuple(c for c in CONDITIONS if c in conds))
            record["run"] = run
            runs_json["runs"].append(record)
            for cond, rows in finals.items():
                want = N_CLAIMS if run == "original" else N_REPEAT
                if len(rows) != want:
                    raise SystemExit(f"{path.name} {cond}: {len(rows)} final rows, not {want}")
                if run == "original":
                    originals[cond] = rows
                else:
                    bad = sorted(cid for cid, r in rows.items()
                                 if r["user_text_sha256"] != originals[cond][cid]["user_text_sha256"])
                    runs_json["user_text_check"][f"{run}/{cond}"] = {
                        "rows": len(rows), "mismatches": len(bad), "mismatched_claims": bad}
                for cid, r in sorted(rows.items()):
                    c = claims[cid]
                    if r["gold_label"] != c["label"] or r["label"] not in LABELS:
                        raise SystemExit(f"{path.name} claim {cid}: label {r['label']!r}, "
                                         f"gold {r['gold_label']!r} / {c['label']!r}")
                    table.append({"claim_id": cid, "label": c["label"], "kind": c["kind"],
                                  "condition": cond, "run": run, "answer": r["label"],
                                  "right": int(r["label"] == c["label"])})
    order = {c: i for i, c in enumerate(CONDITIONS)}
    table.sort(key=lambda t: (RUNS.index(t["run"]), order[t["condition"]], t["claim_id"]))
    with (out_dir / "per_claim.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    cache = args.runs_root / RETRIEVAL_CACHE
    with (out_dir / "retrieval_top5.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(("claim_id", "gold_page_in_top5", "gold_set_in_top5"))
        for cid in sorted(claims):
            c = claims[cid]
            hits = json.loads((cache / f"{cid}.json").read_text(encoding="utf-8"))["hits"]
            row_titles = [h["title"] for h in originals["retrieved"][cid]["retrieved"]]
            if [h["title"] for h in hits] != row_titles:
                raise SystemExit(f"claim {cid}: the cache and the RAG row name other passages")
            titles = {unicodedata.normalize("NFC", t) for t in row_titles}
            pages = {unicodedata.normalize("NFC", p) for p in c["gold_pages"]}
            writer.writerow((cid, int(bool(titles & pages)), int(gold_set_pages(c) <= titles)))
    nei = [r["answer"]["result"]["justification"] for r in originals["no_retrieval"].values()
           if r["label"] == NEI]
    runs_json["no_passages_nei_justifications"] = {
        "nei_answers": len(nei),
        "contain_no_evidence": sum("no evidence" in j.lower() for j in nei),
        "contain_cannot_confirm": sum("cannot confirm" in j.lower() for j in nei),
        "rule": "case-insensitive substring of the justification",
    }
    runs_json["earlier_analyses"] = {}
    if args.nei_audit:
        runs_json["earlier_analyses"]["nei_audit"] = nei_audit_summary(args.nei_audit)
    if args.web_audit:
        runs_json["earlier_analyses"]["web_audit"] = web_audit_summary(args.web_audit)
    (out_dir / "runs.json").write_text(json.dumps(runs_json, indent=1) + "\n", encoding="utf-8")
    print(f"{len(table)} rows -> {out_dir / 'per_claim.csv'}; runs.json; retrieval_top5.csv")
    for key, check in runs_json["user_text_check"].items():
        print(f"user text {key}: {check['rows']} rows, {check['mismatches']} mismatches")
    return 0


# ----------------------------------------------------------------------------------------------
# stats: from the committed inputs to every number
# ----------------------------------------------------------------------------------------------
def read_table(path: Path) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """Return ``{run: {condition: {claim_id: row}}}`` from per_claim.csv."""
    out: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = int(row["claim_id"])
            out[row["run"]][row["condition"]][cid] = {
                "label": row["label"], "kind": row["kind"], "answer": row["answer"],
                "right": int(row["right"])}
    return out


def acc_entry(k: int, n: int) -> dict[str, Any]:
    """Accuracy with its Wilson interval, as counts, shares and rounded percentages."""
    lo, hi = wilson(k, n)
    return {"right": k, "n": n, "acc": k / n if n else None, "wilson": [lo, hi],
            "pct": pct(k, n) if n else None, "wilson_pct": [r1(100 * lo), r1(100 * hi)]}


def interval_pct(values: np.ndarray) -> list[str]:
    """Return the 2.5th and 97.5th percentiles of bootstrap values (shares) in rounded percent."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [r1(100 * float(lo)), r1(100 * float(hi))]


def resample_blocks(n: int, seed: int) -> Any:
    """Yield blocks of bootstrap index rows (``BLOCK`` x ``n``), always the same for a seed."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < BOOTSTRAP_RESAMPLES:
        size = min(BLOCK, BOOTSTRAP_RESAMPLES - done)
        yield rng.integers(0, n, size=(size, n), dtype=np.int32)
        done += size


def resample_within(strata: np.ndarray, seed: int) -> Any:
    """Yield blocks of bootstrap index rows (``BLOCK`` x ``n``) that resample the claims within
    each stratum, always the same for a seed and the same strata.

    Position ``j`` of a row holds a claim drawn from the stratum of claim ``j``, so every
    stratum, and every group made of whole strata, keeps its size in every resample.
    """
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    done = 0
    while done < BOOTSTRAP_RESAMPLES:
        size = min(BLOCK, BOOTSTRAP_RESAMPLES - done)
        idx = np.empty((size, len(strata)), dtype=np.int32)
        for pos in members:
            idx[:, pos] = pos[rng.integers(0, len(pos), size=(size, len(pos)))]
        yield idx
        done += size


def boot_means(values: dict[str, np.ndarray], masks: dict[Any, np.ndarray], strata: np.ndarray,
               seed: int) -> dict[tuple[str, Any], np.ndarray]:
    """Bootstrap means of each value vector over each mask, ``{(value, mask): resampled means}``.

    The resamples are those of :func:`resample_within` and the same for every value vector, so
    differences between conditions are paired. A mask must be a union of strata.
    """
    out: dict[tuple[str, Any], list[np.ndarray]] = {(v, m): [] for v in values for m in masks}
    for idx in resample_within(strata, seed):
        for v, vec in values.items():
            res = vec[idx]
            for m, mask in masks.items():
                out[(v, m)].append(res[:, mask].mean(axis=1))
    return {key: np.concatenate(parts) for key, parts in out.items()}


def confusion_counts(cells: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Confusion counts per resample: ``cells`` holds ``3 * gold + pred`` per claim."""
    flat = cells[idx] + 9 * np.arange(idx.shape[0])[:, None]
    return np.bincount(flat.ravel(), minlength=9 * idx.shape[0]).reshape(idx.shape[0], 3, 3)


def three_class(table: dict, claim_ids: list[int], weights: dict[str, dict[str, float]]) -> dict:
    """Item 1, 2 and 3: per condition on all claims, the six groups, and the reweighted
    accuracies with paired bootstrap intervals (also for the differences of each pair)."""
    code = {label: i for i, label in enumerate(LABELS)}
    groups = [(label, kind) for label in LABELS for kind in KINDS]
    gidx = {g: i for i, g in enumerate(groups)}
    out: dict[str, Any] = {"conditions": {}, "groups": {}, "reweighted": {}}
    cells, right, grp = {}, {}, None
    for cond in CONDITIONS:
        rows = table[cond]
        gold = np.array([code[rows[c]["label"]] for c in claim_ids])
        pred = np.array([code[rows[c]["answer"]] for c in claim_ids])
        right[cond] = (gold == pred).astype(np.int64)
        cells[cond] = 3 * gold + pred
        grp = np.array([gidx[(rows[c]["label"], rows[c]["kind"])] for c in claim_ids])
        conf = np.zeros((3, 3), dtype=int)
        np.add.at(conf, (gold, pred), 1)
        per_label = {}
        for i, label in enumerate(LABELS):
            p, r, f = prf(gold, pred, i)
            per_label[label] = {"precision": p, "recall": r, "f1": f,
                                "recall_pct": pct(int(conf[i, i]), int(conf[i].sum())),
                                "f1_pct": r1(100 * f)}
        macro = float(np.mean([per_label[label]["f1"] for label in LABELS]))
        nei_answers = int(conf[:, 2].sum())
        nei_claims_sided = int(conf[2, 0] + conf[2, 1])
        out["conditions"][cond] = {
            "name": NAMES[cond],
            "accuracy": acc_entry(int(right[cond].sum()), len(claim_ids)),
            "macro_f1": macro, "macro_f1_pct": r1(100 * macro),
            "per_label": per_label,
            "confusion": {LABELS[i]: {LABELS[j]: int(conf[i, j]) for j in range(3)}
                          for i in range(3)},
            "nei_answers": nei_answers,
            "nei_answers_right": int(conf[2, 2]),
            "nei_answer_precision_pct": pct(int(conf[2, 2]), nei_answers) if nei_answers else None,
            "nei_claims_picked_a_side": nei_claims_sided,
            "nei_claims_picked_a_side_pct": pct(nei_claims_sided, int(conf[2].sum())),
        }
        out["groups"][cond] = {
            f"{label}|{kind}": acc_entry(int(right[cond][grp == gidx[(label, kind)]].sum()),
                                         int((grp == gidx[(label, kind)]).sum()))
            for label, kind in groups}
    in_kind = {kind: np.isin(grp, [gidx[(label, kind)] for label in LABELS]) for kind in KINDS}
    out["kinds"] = {c: {kind: acc_entry(int(right[c][in_kind[kind]].sum()),
                                        int(in_kind[kind].sum())) for kind in KINDS}
                    for c in CONDITIONS}
    out["pairs"] = {f"{a}-{b}": paired_entry(right[a], right[b]) for a, b in PAIRS}
    # Reweighted: sum over the six groups of weight x the group's accuracy.
    for wname, wset in weights.items():
        w = np.array([wset[f"{label}|{kind}"] for label, kind in groups])
        point = {c: float(sum(w[gidx[g]] * right[c][grp == gidx[g]].mean() for g in groups))
                 for c in CONDITIONS}
        out["reweighted"][wname] = {"point": point,
                                    "point_pct": {c: r1(100 * v) for c, v in point.items()}}
    # Paired bootstrap over the claims: macro F1 and the reweighted accuracies.
    boot_macro = {c: [] for c in CONDITIONS}
    boot_rw = {wname: {c: [] for c in CONDITIONS} for wname in weights}
    for idx in resample_blocks(len(claim_ids), SEEDS["all"]):
        slots = (grp[idx] + 6 * np.arange(idx.shape[0])[:, None]).ravel()
        g_counts = np.bincount(slots, minlength=6 * idx.shape[0]).reshape(-1, 6)
        for c in CONDITIONS:
            conf = confusion_counts(cells[c], idx)
            boot_macro[c].append(np.mean([f1_from_counts(conf, i) for i in range(3)], axis=0))
            hits = np.bincount(slots, weights=right[c][idx].ravel(), minlength=6 * idx.shape[0])
            g_acc = hits.reshape(-1, 6) / np.maximum(g_counts, 1)
            for wname, wset in weights.items():
                w = np.array([wset[f"{label}|{kind}"] for label, kind in groups])
                boot_rw[wname][c].append(g_acc @ w)
    for c in CONDITIONS:
        vals = np.concatenate(boot_macro[c])
        out["conditions"][c]["macro_f1_bootstrap_pct"] = interval_pct(vals)
    for wname in weights:
        vals = {c: np.concatenate(boot_rw[wname][c]) for c in CONDITIONS}
        out["reweighted"][wname]["bootstrap_pct"] = {c: interval_pct(v) for c, v in vals.items()}
        out["reweighted"][wname]["differences"] = {
            f"{a}-{b}": {"point_pct": r1(100 * (out["reweighted"][wname]["point"][a]
                                                 - out["reweighted"][wname]["point"][b])),
                         "bootstrap_pct": interval_pct(vals[a] - vals[b])}
            for a, b in PAIRS}
    return out


def sides(table: dict, claim_ids: list[int]) -> dict:
    """Items 4, 5 and 6 on the supported/refuted claims (a NOT ENOUGH INFO answer is wrong)."""
    code = {label: i for i, label in enumerate(LABELS)}
    out: dict[str, Any] = {"n": len(claim_ids), "conditions": {}, "pairs": {}, "f1_differences": {}}
    right, cells = {}, {}
    for cond in CONDITIONS:
        rows = table[cond]
        gold = np.array([code[rows[c]["label"]] for c in claim_ids])
        pred = np.array([code[rows[c]["answer"]] for c in claim_ids])
        right[cond] = (gold == pred).astype(np.int64)
        cells[cond] = 3 * gold + pred
        sided = int(np.sum(pred != code[NEI]))
        sided_right = int(np.sum((pred != code[NEI]) & (gold == pred)))
        per_label = {}
        for label in SIDES:
            p, r, f = prf(gold, pred, code[label])
            per_label[label] = {"precision": p, "recall": r, "f1": f,
                                "precision_pct": r1(100 * p), "recall_pct": r1(100 * r),
                                "f1_pct": r1(100 * f)}
        out["conditions"][cond] = {
            "name": NAMES[cond],
            "accuracy": acc_entry(int(right[cond].sum()), len(claim_ids)),
            "picked_a_side": sided, "picked_a_side_pct": pct(sided, len(claim_ids)),
            "right_when_picked": acc_entry(sided_right, sided),
            "per_label": per_label,
        }
    n = len(claim_ids)
    out["kinds"] = {c: {kind: acc_entry(
        sum(table[c][cid]["right"] for cid in claim_ids if table[c][cid]["kind"] == kind),
        sum(1 for cid in claim_ids if table[c][cid]["kind"] == kind)) for kind in KINDS}
        for c in CONDITIONS}
    for a, b in PAIRS:
        out["pairs"][f"{a}-{b}"] = paired_entry(right[a], right[b])
    boot_f1 = {c: {label: [] for label in SIDES} for c in CONDITIONS}
    for idx in resample_blocks(n, SEEDS["sides"]):
        for c in CONDITIONS:
            conf = confusion_counts(cells[c], idx)
            for label in SIDES:
                boot_f1[c][label].append(f1_from_counts(conf, code[label]))
    vals = {c: {label: np.concatenate(boot_f1[c][label]) for label in SIDES} for c in CONDITIONS}
    for c in CONDITIONS:
        for label in SIDES:
            out["conditions"][c]["per_label"][label]["f1_bootstrap_pct"] = interval_pct(
                vals[c][label])
    for a, b in PAIRS:
        out["f1_differences"][f"{a}-{b}"] = {
            label: {"point_pct": r1(100 * (out["conditions"][a]["per_label"][label]["f1"]
                                           - out["conditions"][b]["per_label"][label]["f1"])),
                    "bootstrap_pct": interval_pct(vals[a][label] - vals[b][label])}
            for label in SIDES}
    return out


def paired_entry(right_a: np.ndarray, right_b: np.ndarray) -> dict[str, Any]:
    """McNemar's exact p and the paired interval of two conditions' right/wrong vectors."""
    n = len(right_a)
    only_a = int(np.sum((right_a == 1) & (right_b == 0)))
    only_b = int(np.sum((right_a == 0) & (right_b == 1)))
    diff, lo, hi = paired_interval(only_a, only_b, n)
    p = mcnemar_exact(only_a, only_b)
    return {
        "n": n, "b_only_first_right": only_a, "c_only_second_right": only_b,
        "both_right": int(np.sum((right_a == 1) & (right_b == 1))),
        "both_wrong": int(np.sum((right_a == 0) & (right_b == 0))),
        "difference": diff, "interval": [lo, hi],
        "difference_pct": r1(Fraction(100 * (only_a - only_b), n)),
        "interval_pct": [r1(100 * lo), r1(100 * hi)],
        "interval_pct_2dp": [str(Decimal(repr(100 * x)).quantize(Decimal("0.01"), ROUND_HALF_UP))
                             for x in (lo, hi)],
        "level": lo <= 0 <= hi,
        "mcnemar_p": float(p), "mcnemar_p_text": p_text(p),
    }


def read_flags(path: Path) -> dict[int, tuple[int, int]]:
    """Return ``{claim id: (gold_page_in_top5, gold_set_in_top5)}`` from retrieval_top5.csv."""
    flags = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            flags[int(row["claim_id"])] = (int(row["gold_page_in_top5"]),
                                           int(row["gold_set_in_top5"]))
    return flags


def retrieval_link(table: dict, path: Path) -> dict:
    """Item 7: RAG accuracy split by whether the passages held a gold page / the whole gold set."""
    flags = read_flags(path)
    rows = table["retrieved"]
    out: dict[str, Any] = {}
    for scope, keep in (("all_claims", lambda r: True),
                        ("supported_refuted", lambda r: r["label"] in SIDES)):
        ids = [c for c in sorted(rows) if keep(rows[c])]
        entry = {}
        for name, pos in (("gold_page_in_top5", 0), ("gold_set_in_top5", 1)):
            yes = [c for c in ids if flags[c][pos]]
            no = [c for c in ids if not flags[c][pos]]
            entry[name] = {"share": acc_entry(len(yes), len(ids)),
                           "yes": acc_entry(sum(rows[c]["right"] for c in yes), len(yes)),
                           "no": acc_entry(sum(rows[c]["right"] for c in no), len(no))}
        out[scope] = entry
    return out


# ----------------------------------------------------------------------------------------------
# The multi-hop part: supported and refuted claims, each label weighted one half
# ----------------------------------------------------------------------------------------------
def read_claim_tags(path: Path, sha256: str = CLAIMS_SHA256) -> dict[int, str]:
    """Return ``{claim id: FEVEROUS challenge tag}`` from the claims file; a file whose SHA-256
    is not ``sha256`` stops the run."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{path.name}: SHA-256 {digest}, expected {sha256}")
    records = (json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip())
    return {c["id"]: c["challenge"] for c in records}


def newcombe_entry(k1: int, n1: int, k2: int, n2: int) -> dict[str, Any]:
    """Two accuracies on different claims and their difference (first minus second), with
    Newcombe's interval."""
    diff, lo, hi = newcombe(k1, n1, k2, n2)
    return {"first": acc_entry(k1, n1), "second": acc_entry(k2, n2), "difference": diff,
            "interval": [lo, hi],
            "difference_pct": r1(Fraction(100 * k1, n1) - Fraction(100 * k2, n2)),
            "interval_pct": [r1(100 * lo), r1(100 * hi)], "level": lo <= 0 <= hi}


def boot_entry(point: Fraction, values: np.ndarray, gap: bool = False) -> dict[str, Any]:
    """A figure computed exactly from the claims, with its bootstrap interval from ``values``;
    for a gap also ``level``, whether the interval includes 0."""
    lo, hi = (float(x) for x in np.percentile(values, [2.5, 97.5]))
    out: dict[str, Any] = {"point": float(point), "point_pct": r1(100 * point),
                           "bootstrap": [lo, hi], "bootstrap_pct": interval_pct(values)}
    if gap:
        out["level"] = lo <= 0 <= hi
    return out


def _share(entry: dict) -> Fraction:
    return Fraction(entry["right"], entry["n"])


def _sides_acc(right: np.ndarray, lab: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Accuracy of the supported and of the refuted claims under ``mask`` (``lab``: 0 or 1)."""
    return {label: acc_entry(int(right[mask & (lab == i)].sum()), int((mask & (lab == i)).sum()))
            for i, label in enumerate(SIDES)}


def _ew(entries: dict) -> Fraction | None:
    """Equal-weight mean of the supported and the refuted accuracy; None when a label has no
    claims (not computable)."""
    if any(entries[label]["n"] == 0 for label in SIDES):
        return None
    return (_share(entries[SIDES[0]]) + _share(entries[SIDES[1]])) / 2


def multihop(table: dict, claim_ids: list[int], tags: dict[int, str],
             flags: dict[int, tuple[int, int]], repeat_ids: frozenset[int] = frozenset()) -> dict:
    """Section 6: the claims that need two or more articles and the claims tagged Multi-hop
    Reasoning, in the four conditions, on the supported and refuted claims only.

    A group's equal-weight figure is the mean of its supported and its refuted figure, half each,
    so that a group's label mix does not move it. Intervals: Wilson for one label's accuracy;
    Newcombe for a gap between two sets of claims of one label; the paired formula and McNemar
    for a gap between two conditions on the same claims of one label; the paired bootstrap of
    :func:`boot_means` for every equal-weight figure, every gap between them and every change of
    a gap. The bootstrap resamples the claims within each label and group, the group being that
    of the figure (kind; kind and whether the passages held the gold set; tag; tag and kind).
    ``claim_ids`` may include NOT ENOUGH INFO claims: they are left out, but they count in the
    order of the tags (most claims first). ``repeat_ids`` (the claims of the repeat runs) only
    gives the count of repeated claims per kind.
    """
    base = table["retrieved"]
    ids = [c for c in claim_ids if base[c]["label"] in SIDES]
    n, seed = len(ids), SEEDS["sides"]
    order = Counter(tags[c] for c in claim_ids)
    tag_names = sorted({tags[c] for c in ids}, key=lambda t: (-order[t], t))
    lab = np.array([SIDES.index(base[c]["label"]) for c in ids])
    kind = np.array([KINDS.index(base[c]["kind"]) for c in ids])
    tag = np.array([tag_names.index(tags[c]) for c in ids])
    right = {cond: np.array([table[cond][c]["right"] for c in ids]) for cond in CONDITIONS}
    page = np.array([flags[c][0] for c in ids])
    gset = np.array([flags[c][1] for c in ids])
    is_kind = {k: kind == i for i, k in enumerate(KINDS)}
    is_tag = {t: tag == i for i, t in enumerate(tag_names)}
    everything = np.ones(n, dtype=bool)

    def ew_boot(boot: dict, value: str, *key: Any) -> np.ndarray:
        return (boot[(value, (SIDES[0], *key))] + boot[(value, (SIDES[1], *key))]) / 2

    def nonempty(masks: dict) -> dict:
        return {key: m for key, m in masks.items() if m.any()}

    out: dict[str, Any] = {
        "scope": "supported and refuted claims of the original run, NOT ENOUGH INFO left out; "
                 "equal weight = the mean of the supported and the refuted figure",
        "claims_file": str(CLAIMS_FILE.relative_to(REPO_ROOT)), "claims_sha256": CLAIMS_SHA256,
        "n": n, "bootstrap_seed": seed, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "small_tag_below": SMALL_TAG, "tag_order": tag_names,
    }
    both = is_kind["needs2"] & is_tag.get(MULTIHOP_TAG, np.zeros(n, dtype=bool))
    out["meanings"] = {
        "two_or_more_articles": int(is_kind["needs2"].sum()),
        "multi_hop_reasoning_tag": int(is_tag[MULTIHOP_TAG].sum()) if MULTIHOP_TAG in is_tag else 0,
        "both": int(both.sum()),
        "both_per_label": {label: int((both & (lab == i)).sum()) for i, label in enumerate(SIDES)},
    }
    out["repeats_per_kind"] = {k: sum(1 for c in ids if c in repeat_ids and base[c]["kind"] == k)
                               for k in KINDS}

    # A and B: one article against two or more; strata label x kind.
    kmasks = {(label, k): (lab == i) & is_kind[k] for i, label in enumerate(SIDES) for k in KINDS}
    kboot = boot_means({**right, "page": page, "set": gset}, nonempty(kmasks), lab * 2 + kind,
                       seed)
    kinds: dict[str, Any] = {
        "counts": {k: {label: int(kmasks[(label, k)].sum()) for label in SIDES} for k in KINDS},
        "accuracy": {}, "gap_one_minus_two": {}, "paired": {k: {} for k in KINDS},
        "paired_change_two_minus_one": {},
    }
    for c in CONDITIONS:
        kinds["accuracy"][c] = {}
        for k in KINDS:
            e = _sides_acc(right[c], lab, is_kind[k])
            e["equal_weight"] = boot_entry(_ew(e), ew_boot(kboot, c, k))
            kinds["accuracy"][c][k] = e
        one, two = kinds["accuracy"][c]["single"], kinds["accuracy"][c]["needs2"]
        gap: dict[str, Any] = {label: newcombe_entry(one[label]["right"], one[label]["n"],
                                                     two[label]["right"], two[label]["n"])
                               for label in SIDES}
        gap["equal_weight"] = boot_entry(_ew(one) - _ew(two),
                                         ew_boot(kboot, c, "single") - ew_boot(kboot, c, "needs2"),
                                         gap=True)
        kinds["gap_one_minus_two"][c] = gap
    for a, b in RAG_PAIRS:
        acc = kinds["accuracy"]
        for k in KINDS:
            e = {label: paired_entry(right[a][kmasks[(label, k)]], right[b][kmasks[(label, k)]])
                 for label in SIDES}
            e["equal_weight"] = boot_entry(_ew(acc[a][k]) - _ew(acc[b][k]),
                                           ew_boot(kboot, a, k) - ew_boot(kboot, b, k), gap=True)
            kinds["paired"][k][f"{a}-{b}"] = e
        change: dict[str, Any] = {}
        for label in SIDES:
            point = ((_share(acc[a]["needs2"][label]) - _share(acc[b]["needs2"][label]))
                     - (_share(acc[a]["single"][label]) - _share(acc[b]["single"][label])))
            values = ((kboot[(a, (label, "needs2"))] - kboot[(b, (label, "needs2"))])
                      - (kboot[(a, (label, "single"))] - kboot[(b, (label, "single"))]))
            change[label] = boot_entry(point, values, gap=True)
        change["equal_weight"] = boot_entry(
            (_ew(acc[a]["needs2"]) - _ew(acc[b]["needs2"]))
            - (_ew(acc[a]["single"]) - _ew(acc[b]["single"])),
            (ew_boot(kboot, a, "needs2") - ew_boot(kboot, b, "needs2"))
            - (ew_boot(kboot, a, "single") - ew_boot(kboot, b, "single")), gap=True)
        kinds["paired_change_two_minus_one"][f"{a}-{b}"] = change
    out["kinds"] = kinds

    # B: what the five RAG passages covered; RAG split by the whole gold set (strata label x
    # kind x flag, so that both halves keep their size).
    smasks = {(label, k, f): kmasks[(label, k)] & (gset == f)
              for label in SIDES for k in KINDS for f in (1, 0)}
    sboot = boot_means({"retrieved": right["retrieved"]}, nonempty(smasks),
                       lab * 4 + kind * 2 + gset, seed)
    coverage: dict[str, Any] = {}
    for k in KINDS:
        entry: dict[str, Any] = {}
        for name, value, vec in (("gold_page_in_top5", "page", page),
                                 ("gold_set_in_top5", "set", gset)):
            e = _sides_acc(vec, lab, is_kind[k])
            e["equal_weight"] = boot_entry(_ew(e), ew_boot(kboot, value, k))
            entry[name] = e
        for f, name in ((1, "rag_when_set_in_top5"), (0, "rag_when_set_not_in_top5")):
            e = _sides_acc(right["retrieved"], lab, is_kind[k] & (gset == f))
            point = _ew(e)
            e["equal_weight"] = (None if point is None
                                 else boot_entry(point, ew_boot(sboot, "retrieved", k, f)))
            entry[name] = e
        yes, no = entry["rag_when_set_in_top5"], entry["rag_when_set_not_in_top5"]
        split: dict[str, Any] = {label: newcombe_entry(yes[label]["right"], yes[label]["n"],
                                                       no[label]["right"], no[label]["n"])
                                 for label in SIDES if yes[label]["n"] and no[label]["n"]}
        split["equal_weight"] = (
            None if yes["equal_weight"] is None or no["equal_weight"] is None
            else boot_entry(_ew(yes) - _ew(no), ew_boot(sboot, "retrieved", k, 1)
                            - ew_boot(sboot, "retrieved", k, 0), gap=True))
        entry["rag_set_in_minus_not"] = split
        coverage[k] = entry
    out["coverage"] = coverage

    # C: the challenge tags; strata label x tag.
    tmasks = {(label, t): (lab == i) & is_tag[t] for i, label in enumerate(SIDES)
              for t in tag_names}
    tboot = boot_means(right, nonempty(tmasks), lab * len(tag_names) + tag, seed)
    tags_out: dict[str, Any] = {}
    for t in tag_names:
        e: dict[str, Any] = {
            "n": int(is_tag[t].sum()), "small": bool(is_tag[t].sum() < SMALL_TAG),
            "n_per_label": {label: int(tmasks[(label, t)].sum()) for label in SIDES},
            "counts": {label: {k: int((tmasks[(label, t)] & is_kind[k]).sum()) for k in KINDS}
                       for label in SIDES},
            "accuracy": {}}
        for c in CONDITIONS:
            a = _sides_acc(right[c], lab, is_tag[t])
            point = _ew(a)
            a["equal_weight"] = None if point is None else boot_entry(point, ew_boot(tboot, c, t))
            e["accuracy"][c] = a
        tags_out[t] = e
    out["tags"] = tags_out

    # D: Multi-hop Reasoning against Other, overall (strata label x tag) and within each kind
    # (strata label x tag x kind).
    pair = (MULTIHOP_TAG, OTHER_TAG)
    if not all(t in is_tag for t in pair):
        out["multihop_vs_other"] = None
        return out
    dmasks = {(label, t, k): (lab == i) & is_tag[t] & is_kind[k]
              for i, label in enumerate(SIDES) for t in pair for k in KINDS}
    dboot = boot_means(right, nonempty(dmasks), (lab * len(tag_names) + tag) * 2 + kind, seed)

    def ew_scope(c: str, t: str, scope: str) -> np.ndarray:
        return ew_boot(tboot, c, t) if scope == "all" else ew_boot(dboot, c, t, scope)

    scope_mask = {"all": everything, **is_kind}
    mvo: dict[str, Any] = {"accuracy": {}, "gap": {}}
    for scope in ("all", *KINDS):
        mvo["accuracy"][scope], mvo["gap"][scope] = {}, {}
        for c in CONDITIONS:
            per = {}
            for t in pair:
                a = _sides_acc(right[c], lab, is_tag[t] & scope_mask[scope])
                point = _ew(a)
                a["equal_weight"] = (None if point is None
                                     else boot_entry(point, ew_scope(c, t, scope)))
                per[t] = a
            mvo["accuracy"][scope][c] = per
            g: dict[str, Any] = {
                label: newcombe_entry(per[pair[0]][label]["right"], per[pair[0]][label]["n"],
                                      per[pair[1]][label]["right"], per[pair[1]][label]["n"])
                for label in SIDES if per[pair[0]][label]["n"] and per[pair[1]][label]["n"]}
            g["equal_weight"] = (
                None if per[pair[0]]["equal_weight"] is None or per[pair[1]]["equal_weight"] is None
                else boot_entry(_ew(per[pair[0]]) - _ew(per[pair[1]]),
                                ew_scope(c, pair[0], scope) - ew_scope(c, pair[1], scope),
                                gap=True))
            mvo["gap"][scope][c] = g
    acc_all = mvo["accuracy"]["all"]
    rag_gold: dict[str, Any] = {}
    for t in pair:
        e = {label: paired_entry(right["retrieved"][tmasks[(label, t)]],
                                 right["gold"][tmasks[(label, t)]]) for label in SIDES}
        e["equal_weight"] = boot_entry(
            _ew(acc_all["retrieved"][t]) - _ew(acc_all["gold"][t]),
            ew_boot(tboot, "retrieved", t) - ew_boot(tboot, "gold", t), gap=True)
        rag_gold[t] = e
    difference: dict[str, Any] = {}
    for label in SIDES:
        point = sum(sign * (_share(acc_all["retrieved"][t][label]) - _share(acc_all["gold"][t][label]))
                    for sign, t in ((1, pair[0]), (-1, pair[1])))
        values = sum(sign * (tboot[("retrieved", (label, t))] - tboot[("gold", (label, t))])
                     for sign, t in ((1, pair[0]), (-1, pair[1])))
        difference[label] = boot_entry(point, values, gap=True)
    difference["equal_weight"] = boot_entry(
        sum(sign * (_ew(acc_all["retrieved"][t]) - _ew(acc_all["gold"][t]))
            for sign, t in ((1, pair[0]), (-1, pair[1]))),
        sum(sign * (ew_boot(tboot, "retrieved", t) - ew_boot(tboot, "gold", t))
            for sign, t in ((1, pair[0]), (-1, pair[1]))), gap=True)
    mvo["rag_minus_gold"] = rag_gold
    mvo["rag_minus_gold_difference"] = difference
    out["multihop_vs_other"] = mvo
    return out


def variance(table: dict) -> dict:
    """Item 9: agreement and switches between the runs on the 450 repeated claims."""
    out: dict[str, Any] = {}
    present = [run for run in RUNS if table.get(run)]
    if present == ["original"]:
        return {"missing": "no repeat run in per_claim.csv"}
    repeat_ids = sorted(set.intersection(*(set(table[r]["retrieved"]) for r in present)))
    for cond in CONDITIONS:
        entry: dict[str, Any] = {}
        for scope, keep in (("three_class_450", lambda r: True),
                            ("supported_refuted_300", lambda r: r["label"] in SIDES)):
            ids = [c for c in repeat_ids if keep(table["original"][cond][c])]
            sc: dict[str, Any] = {"n": len(ids), "accuracy": {}, "pairs": {}}
            for run in present:
                sc["accuracy"][run] = acc_entry(sum(table[run][cond][c]["right"] for c in ids),
                                                len(ids))
            for a, b in combinations(present, 2):
                same = sum(table[a][cond][c]["answer"] == table[b][cond][c]["answer"] for c in ids)
                switched = sum(table[a][cond][c]["right"] != table[b][cond][c]["right"]
                               for c in ids)
                f = switched / len(ids)
                sd = {str(n): {"sd": rerun_sd(f, n), "sd_pct": r1(100 * rerun_sd(f, n)),
                               "sd_difference": math.sqrt(2) * rerun_sd(f, n),
                               "sd_difference_pct": r1(100 * math.sqrt(2) * rerun_sd(f, n))}
                      for n in (2000, 3000)}
                between = paired_entry(np.array([table[a][cond][c]["right"] for c in ids]),
                                       np.array([table[b][cond][c]["right"] for c in ids]))
                sc["pairs"][f"{a}-{b}"] = {"identical": same, "identical_pct": pct(same, len(ids)),
                                           "switched": switched, "f": f,
                                           "switched_pct": pct(switched, len(ids)),
                                           "rerun_sd": sd, "accuracy_difference": between}
            entry[scope] = sc
        out[cond] = entry
    return out


def two_sd_rule(pairs_by_scope: dict[str, dict], var: dict) -> dict:
    """The wording rule's last part: is a paired gap between two conditions below two rerun SDs?

    For each pair the rerun SD of a difference is taken from the larger switch rate f of the two
    conditions over the three run pairs (the larger SD), on the matching scope of the repeats:
    the 450 claims for the 3,000-claim comparisons, their 300 supported/refuted claims for the
    2,000-claim ones.
    """
    if "missing" in var:
        return {"missing": var["missing"]}
    out: dict[str, Any] = {}
    for scope, (repeat_scope, n) in (("three_class", ("three_class_450", "3000")),
                                     ("supported_refuted", ("supported_refuted_300", "2000"))):
        rows: dict[str, Any] = {}
        for a, b in PAIRS:
            cands = [(e["rerun_sd"][n]["sd_difference"], cond, run_pair)
                     for cond in (a, b)
                     for run_pair, e in var[cond][repeat_scope]["pairs"].items()]
            sd_diff, cond, run_pair = max(cands)
            gap = pairs_by_scope[scope][f"{a}-{b}"]
            two_sd = 2 * sd_diff
            rows[f"{a}-{b}"] = {
                "difference_pct": gap["difference_pct"], "level": gap["level"],
                "sd_difference": sd_diff, "sd_difference_pct": r1(100 * sd_diff),
                "two_sd": two_sd, "two_sd_pct": r1(100 * two_sd),
                "from": {"condition": cond, "runs": run_pair},
                "below_two_sd": abs(gap["difference"]) < two_sd,
            }
        out[scope] = rows
    return out


def costs(runs: dict, sql_file: Path, embed_file: Path) -> dict:
    """Item 8: seconds, tokens and usage windows per condition, and retrieval's share of a call."""
    out: dict[str, Any] = {"conditions": {}, "runs": []}
    for rec in runs["runs"]:
        out["runs"].append({
            "run": rec["run"], "run_folder": rec["run_folder"],
            "conditions": rec["conditions"], "cli_version": rec["cli_version"],
            "five_hour_used": rec["window_five_hour"]["used"],
            "five_hour_windows": rec["window_five_hour"]["windows"],
            "seven_day_used": rec["window_seven_day"]["used"],
            "calls": sum(v["calls"] for v in rec["per_condition"].values()),
        })
        if rec["run"] != "original":
            continue
        for cond, c in rec["per_condition"].items():
            out["conditions"][cond] = {
                "name": NAMES[cond], "calls": c["calls"],
                "seconds_median": c["seconds_final_attempt"]["median"],
                "seconds_median_text": r1(c["seconds_final_attempt"]["median"]),
                "seconds_p95": c["seconds_final_attempt"]["p95"],
                "tokens_sum": c["tokens_sum"], "tokens_median": c["tokens_median"],
                "prompt_tokens_millions_text": r1(c["tokens_sum"]["prompt_tokens"] / 1e6),
                "thinking_median": c["tokens_median"]["thinking_tokens"],
                "web": c.get("web"),
            }
    rag = out["conditions"]["retrieved"]["seconds_median"]
    out["ratios_to_rag_median"] = {cond: v["seconds_median"] / rag
                                   for cond, v in out["conditions"].items()}
    out["ratios_to_rag_median_text"] = {k: r1(v) for k, v in out["ratios_to_rag_median"].items()}
    sql = json.loads(sql_file.read_text(encoding="utf-8"))
    sql_ms = _inline_no_filter_p50(sql)
    embed = json.loads(embed_file.read_text(encoding="utf-8"))
    embed_ms = embed.get("embedding_latency", {}).get("p50_ms")
    device = (embed.get("parameters") or {}).get("embedding_device")
    share = (sql_ms + (embed_ms or 0.0)) / (rag * 1000)
    out["retrieval_share"] = {
        "sql_ms_p50": sql_ms, "sql_file": str(sql_file.relative_to(REPO_ROOT)),
        "embed_ms_p50": embed_ms, "embed_device": device,
        "embed_file": str(embed_file.relative_to(REPO_ROOT)),
        "rag_call_seconds_median": rag, "share": share, "share_pct": r1(100 * share),
        "note": "The database times were measured with the CPU capped at 50% and the GPU at "
                "2,100 MHz. The RAG run embedded its claims on the CPU; the stated embedding "
                f"time was measured on {device}.",
    }
    return out


def _inline_no_filter_p50(sql: dict) -> float:
    """Return the p50 milliseconds of the joined ``inline`` statement without a filter."""
    if sql.get("parameters", {}).get("k") != 5:
        raise SystemExit("the SQL timing file is not at k = 5")
    found = [m for m in sql.get("unfiltered", [])
             if m.get("where") == "MariaDB" and "strategy inline" in m.get("method", "")]
    if len(found) != 1:
        raise SystemExit("no single inline row without a filter in the SQL timing file")
    return float(found[0]["p50_ms"])


def cmd_stats(args: argparse.Namespace) -> int:
    """Compute every number from the committed inputs; write llm_results.json and the report."""
    out_dir: Path = args.out
    tags = read_claim_tags(args.claims)
    full = read_table(out_dir / "per_claim.csv")
    original = full["original"]
    claim_ids = sorted(original["retrieved"])
    if len(claim_ids) != N_CLAIMS or any(len(original[c]) != N_CLAIMS for c in CONDITIONS):
        raise SystemExit("per_claim.csv: the original run needs 3,000 claims per condition")
    sides_ids = [c for c in claim_ids if original["retrieved"][c]["label"] in SIDES]
    breakdown = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))["weights"]
    weights = {name: {g: v["weight"] for g, v in breakdown[name]["groups"].items()}
               for name in ("all", "text")}
    runs = json.loads((out_dir / "runs.json").read_text(encoding="utf-8"))
    tc = three_class(original, claim_ids, weights)
    gain = (tc["conditions"]["retrieved"]["accuracy"]["right"]
            - tc["conditions"]["no_retrieval"]["accuracy"]["right"])
    possible = (tc["conditions"]["gold"]["accuracy"]["right"]
                - tc["conditions"]["no_retrieval"]["accuracy"]["right"])
    sr = sides(original, sides_ids)
    var = variance(full)
    result = {
        "conventions": {
            "right": "answer equals the FEVEROUS label; NOT ENOUGH INFO on a supported or "
                     "refuted claim is wrong",
            "rounding": "one decimal, half up (decimal module)",
            "z": Z, "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seeds": SEEDS,
            "bootstrap": "paired: the same claim resamples for every condition; percentile "
                         "interval (2.5th to 97.5th)",
            "multiple_comparisons": "no correction",
            "weights_file": str(WEIGHTS_FILE.relative_to(REPO_ROOT)),
            "weights": weights,
        },
        "three_class": tc,
        "rag_share_of_gold_gain": {"gain": gain, "possible": possible, "pct": pct(gain, possible)},
        "supported_refuted": sr,
        "retrieval_link": retrieval_link(original, out_dir / "retrieval_top5.csv"),
        "costs": costs(runs, SQL_TIMING_FILE, EMBED_TIMING_FILE),
        "variance": var,
        "two_sd_rule": two_sd_rule({"three_class": tc["pairs"], "supported_refuted": sr["pairs"]},
                                   var),
        "user_text_check": runs.get("user_text_check", {}),
        "no_passages_nei_justifications": runs.get("no_passages_nei_justifications"),
        "earlier_analyses": runs.get("earlier_analyses", {}),
        "multihop": multihop(original, claim_ids, tags, read_flags(out_dir / "retrieval_top5.csv"),
                             frozenset(full["r1"]["retrieved"]) if full.get("r1") else frozenset()),
    }
    (out_dir / "llm_results.json").write_text(json.dumps(result, indent=1) + "\n",
                                              encoding="utf-8")
    (out_dir / "LLM_RESULTS.md").write_text(report(result), encoding="utf-8")
    print(f"-> {out_dir / 'llm_results.json'}, {out_dir / 'LLM_RESULTS.md'}")
    return 0


# ----------------------------------------------------------------------------------------------
# The Markdown report
# ----------------------------------------------------------------------------------------------
def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _acc(e: dict) -> str:
    return f"{_fmt_int(e['right'])} / {_fmt_int(e['n'])} = {e['pct']}% ({e['wilson_pct'][0]}–{e['wilson_pct'][1]})"


def report(res: dict) -> str:
    """Return LLM_RESULTS.md from the computed numbers."""
    conv = res["conventions"]
    tc, sr = res["three_class"], res["supported_refuted"]
    L: list[str] = [
        "# LLM fact-checker: results",
        "",
        ("Written by `scripts/llm_results.py stats` from `per_claim.csv`, `retrieval_top5.csv` and "
        f"`runs.json` in this folder, the weights in `{conv['weights_file']}` and the database "
        "timings named in the cost section. Claude Haiku 5.5, effort high, the same 3,000 "
        "FEVEROUS claims in four conditions. A claim is right when the answer equals the "
        "FEVEROUS label. Percentages are rounded to one decimal, half up. Intervals are 95%: "
        "Wilson for an accuracy; for a paired difference of accuracies (b − c)/n ± 1.96 · "
        "√(b + c − (b − c)²/n) / n with McNemar's exact two-sided p; for F1 and the reweighted "
        f"accuracy a paired bootstrap with {conv['bootstrap_resamples']:,} resamples (seeds "
        f"{conv['bootstrap_seeds']['all']} for the 3,000 claims, "
        f"{conv['bootstrap_seeds']['sides']} for the 2,000 supported/refuted claims). No "
        "correction for multiple comparisons."),
        "",
        "## 1. Three classes (3,000 claims)",
        "",
        "| condition | accuracy (95% interval) | macro F1 (interval) | recall S / R / NEI | NEI answers | NEI answers right |",
        "|---|---|---|---|---|---|",
    ]
    for c in CONDITIONS:
        e = tc["conditions"][c]
        rec = " / ".join(e["per_label"][label]["recall_pct"] for label in LABELS)
        nei = e["nei_answers"]
        L.append(f"| {NAMES[c]} | {_acc(e['accuracy'])} | {e['macro_f1_pct']} "
                 f"({e['macro_f1_bootstrap_pct'][0]}–{e['macro_f1_bootstrap_pct'][1]}) | {rec} | "
                 f"{_fmt_int(nei)} | {_fmt_int(e['nei_answers_right'])} = "
                 f"{e['nei_answer_precision_pct']}% |")
    g = res["rag_share_of_gold_gain"]
    L += ["", (f"RAG gains {g['gain']:,} right claims over no passages, of the {g['possible']:,} "
          f"that gold gains: {g['pct']}%."), "",
          "On NOT ENOUGH INFO claims, the share where the model picked a side: " + "; ".join(
              f"{NAMES[c]} {tc['conditions'][c]['nei_claims_picked_a_side_pct']}%"
              for c in CONDITIONS) + ".", "",
          "### Confusion matrices", "",
          "Rows: the FEVEROUS label; columns: the answer (SUPPORTS / REFUTES / NOT ENOUGH INFO).",
          "", "| label | " + " | ".join(NAMES[c] for c in CONDITIONS) + " |",
          "|---|" + "---|" * len(CONDITIONS)]
    for label in LABELS:
        L.append(f"| {label} | " + " | ".join(
            " / ".join(_fmt_int(tc["conditions"][c]["confusion"][label][a]) for a in LABELS)
            for c in CONDITIONS) + " |")
    L += ["", "### Paired comparisons on three classes", "",
          "| pair | only the first right (b) | only the second right (c) | difference, points | 95% interval | McNemar p |",
          "|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        p = tc["pairs"][f"{a}-{b}"]
        L.append(f"| {NAMES[a]} vs {NAMES[b]} | {p['b_only_first_right']:,} | "
                 f"{p['c_only_second_right']:,} | {p['difference_pct']} | {p['interval_pct'][0]} "
                 f"to {p['interval_pct'][1]} | {p['mcnemar_p_text']} |")
    L += ["", "### One article or two or more", "",
          "| claims | " + " | ".join(NAMES[c] for c in CONDITIONS) + " |",
          "|---|" + "---|" * len(CONDITIONS)]
    for scope, title, block in (("three", "three classes", tc), ("sides", "supported/refuted", sr)):
        for kind in KINDS:
            n = block["kinds"]["retrieved"][kind]["n"]
            L.append(f"| {title}, {'one article' if kind == 'single' else 'two or more'} "
                     f"({n:,}) | " + " | ".join(f"{block['kinds'][c][kind]['pct']}%"
                                                 for c in CONDITIONS) + " |")
    L += ["", "### The six groups (label × one article or two or more)", "",
          "| group | n | " + " | ".join(NAMES[c] for c in CONDITIONS) + " |",
          "|---|---|" + "---|" * len(CONDITIONS)]
    for label in LABELS:
        for kind in KINDS:
            key = f"{label}|{kind}"
            n = tc["groups"]["retrieved"][key]["n"]
            L.append(f"| {label}, {'one article' if kind == 'single' else 'two or more'} | {n:,} | "
                     + " | ".join(f"{tc['groups'][c][key]['pct']}%" for c in CONDITIONS) + " |")
    L += ["", "### Reweighted to FEVEROUS's mix of the six groups", "",
          (f"Weights from `{conv['weights_file']}` (`all`: every FEVEROUS train + dev claim in a "
          "group; `text`: those with a text-only evidence set). Accuracy = sum over the groups of "
          "weight × the group's accuracy."), "",
          "| weights | " + " | ".join(NAMES[c] for c in CONDITIONS) + " |",
          "|---|" + "---|" * len(CONDITIONS)]
    for wname in ("all", "text"):
        rw = tc["reweighted"][wname]
        L.append(f"| {wname} | " + " | ".join(
            f"{rw['point_pct'][c]}% ({rw['bootstrap_pct'][c][0]}–{rw['bootstrap_pct'][c][1]})"
            for c in CONDITIONS) + " |")
    L += ["", "| difference, weights `all` | point | interval |", "|---|---|---|"]
    for a, b in PAIRS:
        d = tc["reweighted"]["all"]["differences"][f"{a}-{b}"]
        L.append(f"| {NAMES[a]} − {NAMES[b]} | {d['point_pct']} | {d['bootstrap_pct'][0]} to "
                 f"{d['bootstrap_pct'][1]} |")
    L += ["", f"## 2. Supported and refuted claims only ({sr['n']:,} claims)", "",
          "A NOT ENOUGH INFO answer counts as wrong; no claim is dropped because of an answer.", "",
          "| condition | accuracy (95% interval) | picked a side | right when it picked a side |",
          "|---|---|---|---|"]
    for c in CONDITIONS:
        e = sr["conditions"][c]
        L.append(f"| {NAMES[c]} | {_acc(e['accuracy'])} | {_fmt_int(e['picked_a_side'])} = "
                 f"{e['picked_a_side_pct']}% | {_acc(e['right_when_picked'])} |")
    L += ["", "### Paired comparisons", "",
          "| pair | only the first right (b) | only the second right (c) | difference, points | 95% interval | McNemar p |",
          "|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        p = sr["pairs"][f"{a}-{b}"]
        L.append(f"| {NAMES[a]} vs {NAMES[b]} | {p['b_only_first_right']:,} | "
                 f"{p['c_only_second_right']:,} | {p['difference_pct']} | {p['interval_pct'][0]} "
                 f"to {p['interval_pct'][1]} | {p['mcnemar_p_text']} |")
    L += ["", "### Precision, recall and F1 per label", "",
          "| condition | SUPPORTS P / R / F1 (F1 interval) | REFUTES P / R / F1 (F1 interval) |",
          "|---|---|---|"]
    for c in CONDITIONS:
        cells = []
        for label in SIDES:
            e = sr["conditions"][c]["per_label"][label]
            cells.append(f"{e['precision_pct']} / {e['recall_pct']} / {e['f1_pct']} "
                         f"({e['f1_bootstrap_pct'][0]}–{e['f1_bootstrap_pct'][1]})")
        L.append(f"| {NAMES[c]} | " + " | ".join(cells) + " |")
    L += ["", "| F1 difference | SUPPORTS (interval) | REFUTES (interval) |", "|---|---|---|"]
    for a, b in PAIRS:
        d = sr["f1_differences"][f"{a}-{b}"]
        L.append(f"| {NAMES[a]} − {NAMES[b]} | " + " | ".join(
            f"{d[label]['point_pct']} ({d[label]['bootstrap_pct'][0]} to "
            f"{d[label]['bootstrap_pct'][1]})" for label in SIDES) + " |")
    link = res["retrieval_link"]
    L += ["", "## 3. RAG accuracy and what retrieval found", "",
          ("Pages of the five RAG passages against the claim's `gold_pages` (a gold page) and "
          "against every page of the evidence set shown in the gold condition (the whole gold set)."),
          "", "| claims | split | share of claims | accuracy when yes | accuracy when no |",
          "|---|---|---|---|---|"]
    for scope, title in (("all_claims", "all 3,000, three classes"),
                         ("supported_refuted", "2,000 supported/refuted")):
        for name, label in (("gold_page_in_top5", "a gold page in the top 5"),
                            ("gold_set_in_top5", "the whole gold set in the top 5")):
            e = link[scope][name]
            L.append(f"| {title} | {label} | {_fmt_int(e['share']['right'])} = "
                     f"{e['share']['pct']}% | {_acc(e['yes'])} | {_acc(e['no'])} |")
    co = res["costs"]
    L += ["", "## 4. Cost and time", "",
          ("From the run rows (the final attempt of each claim). The LLM times are observed "
          "medians of calls through the subscription, five at a time, not a controlled timing."),
          "", "| condition | calls | median s per claim | × RAG | p95 s | prompt tokens | median thinking tokens | output tokens |",
          "|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        e = co["conditions"][c]
        L.append(f"| {NAMES[c]} | {e['calls']:,} | {e['seconds_median_text']} | "
                 f"{co['ratios_to_rag_median_text'][c]} | {r1(e['seconds_p95'])} | "
                 f"{e['tokens_sum']['prompt_tokens']:,} | {e['thinking_median']:,.0f} | "
                 f"{e['tokens_sum']['output_tokens']:,} |")
    L += ["", ("Usage windows over each run folder (readings of the whole account, so other use at "
          "the same time is included; per window, the highest reading minus the first, summed over "
          "the windows a run spanned):"), "",
          "| run | folder | conditions | calls | CLI | five-hour used (windows) | seven-day used |",
          "|---|---|---|---|---|---|---|"]
    for e in co["runs"]:
        L.append(f"| {e['run']} | `{e['run_folder']}` | {', '.join(NAMES[c] for c in e['conditions'])} "
                 f"| {e['calls']:,} | {e['cli_version']} | {e['five_hour_used']:.2f} "
                 f"({e['five_hour_windows']}) | "
                 f"{e['seven_day_used']:.2f} |")
    rs = co["retrieval_share"]
    L += ["", (f"Retrieval's share of a RAG call: the joined `inline` statement on S1 at k = 5 "
          f"without a filter, median {rs['sql_ms_p50']:.3f} ms (`{rs['sql_file']}`), plus one "
          f"claim embedding, median {rs['embed_ms_p50']:.2f} ms on {rs['embed_device']} "
          f"(`{rs['embed_file']}`), against the median RAG call of "
          f"{rs['rag_call_seconds_median']:.3f} s: {rs['share_pct']}%. {rs['note']}"),
          ""]
    web = co["conditions"]["web"]["web"]
    nj = res["no_passages_nei_justifications"]
    L += ["## Tool use of the web condition and the earlier analyses", "",
          (f"Web condition: {web['searches_sum']:,} WebSearch and {web['fetches_sum']:,} WebFetch "
          f"calls over 3,000 claims, {web['denied_sum']} fetches denied by the deny list, median "
          f"{web['turns_median']:g} turns per call, {web['claims_without_a_tool_call']} claims "
          "answered without a tool call."), "",
          (f"No passages: {nj['contain_no_evidence']:,} of its {nj['nei_answers']:,} NOT ENOUGH INFO "
          "justifications contain the words \"no evidence\" and "
          f"{nj['contain_cannot_confirm']:,} contain \"cannot confirm\" ({nj['rule']})."), ""]
    ea = res.get("earlier_analyses") or {}
    if "web_audit" in ea:
        w = ea["web_audit"]
        L += [(f"Web audit ({w['audited_at']}): {w['web_rows']:,} rows audited, "
              f"{w['flagged_rows']} flagged (kept), {w['denied_calls']} denied calls; "
              f"{w['fetches_en_wikipedia']:,} of the {w['web_fetch_calls']:,} fetches opened "
              "en.wikipedia.org."), ""]
    if "nei_audit" in ea:
        a = ea["nei_audit"]
        st = a["stats"]
        L += [(f"NOT ENOUGH INFO audit: {st['cases']} reviewed cases, rated by two reviewer agents "
              f"each ({st['both_rated']} rated by both), who agreed on the main reason in "
              f"{st['agree_error_type']} and on whether the text settles the claim in "
              f"{st['agree_settle']}; {st['adjudicated']} settled by a third agent."), "",
              "| condition | cases | the FEVEROUS label looks doubtful | passages settle the claim (supports / refutes) |",
              "|---|---|---|---|"]
        for cond in ("gold", "retrieved", "no_retrieval"):
            e = a["by_condition"][cond]
            L.append(f"| {NAMES[cond]} | {e['cases']} | {e['error'].get('feverous_label_doubtful', 0)} "
                     f"| {e['settle'].get('yes_supports', 0)} / {e['settle'].get('yes_refutes', 0)} |")
        L += [""]
    var = res["variance"]
    L += ["## 5. Run-to-run variance (450 repeated claims)", ""]
    if "missing" in var:
        L += [var["missing"], ""]
    else:
        L += [("f = share of claims that switched between right and wrong across two runs; rerun "
              "standard deviation √(f / 2n), times √2 for a difference between two conditions."), "",
              "| condition | claims | runs | identical labels | switched (f) | SD n = 2,000 | SD difference | SD n = 3,000 | SD difference |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in CONDITIONS:
            for scope, title in (("three_class_450", "450"), ("supported_refuted_300", "300 S/R")):
                for pair, e in var[c][scope]["pairs"].items():
                    sd2, sd3 = e["rerun_sd"]["2000"], e["rerun_sd"]["3000"]
                    L.append(f"| {NAMES[c]} | {title} | {pair} | {e['identical']} = "
                             f"{e['identical_pct']}% | {e['switched']} = {e['switched_pct']}% | "
                             f"{sd2['sd_pct']} | {sd2['sd_difference_pct']} | {sd3['sd_pct']} | "
                             f"{sd3['sd_difference_pct']} |")
        L += ["", "| condition | claims | " + " | ".join(RUNS) + " |",
              "|---|---|" + "---|" * len(RUNS)]
        for c in CONDITIONS:
            for scope, title in (("three_class_450", "450"), ("supported_refuted_300", "300 S/R")):
                acc = var[c][scope]["accuracy"]
                L.append(f"| {NAMES[c]} | {title} | " + " | ".join(
                    f"{acc[r]['pct']}%" if r in acc else "-" for r in RUNS) + " |")
        L += ["", ("Accuracy differences between two runs of the same condition, claim by claim "
                   "(b: right only in the first run, c: right only in the second):"), "",
              "| condition | claims | runs | b | c | difference, points | 95% interval | McNemar p |",
              "|---|---|---|---|---|---|---|---|"]
        for c in CONDITIONS:
            for scope, title in (("three_class_450", "450"), ("supported_refuted_300", "300 S/R")):
                for pair, e in var[c][scope]["pairs"].items():
                    d = e["accuracy_difference"]
                    L.append(f"| {NAMES[c]} | {title} | {pair} | {d['b_only_first_right']} | "
                             f"{d['c_only_second_right']} | {d['difference_pct']} | "
                             f"{d['interval_pct'][0]} to {d['interval_pct'][1]} | "
                             f"{d['mcnemar_p_text']} |")
        L += [""]
        rule = res["two_sd_rule"]
        L += [("Two rerun SDs of a difference for each paired comparison between conditions: the "
               "larger SD of the two conditions over the three run pairs, from the 450 claims for "
               "the 3,000-claim comparisons and from their 300 supported/refuted claims for the "
               "2,000-claim ones."), "",
              "| claims | pair | difference, points | level | 2 rerun SDs | from | gap below 2 SDs |",
              "|---|---|---|---|---|---|---|"]
        for scope, title in (("three_class", "3,000"), ("supported_refuted", "2,000 S/R")):
            for pair, e in rule[scope].items():
                a, b = pair.split("-")
                L.append(f"| {title} | {NAMES[a]} vs {NAMES[b]} | {e['difference_pct']} | "
                         f"{'yes' if e['level'] else 'no'} | {e['two_sd_pct']} | "
                         f"{NAMES[e['from']['condition']]} {e['from']['runs']} | "
                         f"{'yes' if e['below_two_sd'] else 'no'} |")
        L += [""]
        check = res["user_text_check"]
        L += ["User texts of the repeats against the original rows: " + "; ".join(
            f"{k} {v['mismatches']} of {v['rows']} differ" for k, v in check.items()) + ".", ""]
    if res.get("multihop"):
        L += multihop_lines(res["multihop"])
    return "\n".join(L)


def _iv(e: dict) -> str:
    """An accuracy with its Wilson interval, in percent."""
    return f"{e['pct']}% ({e['wilson_pct'][0]}–{e['wilson_pct'][1]})"


def _gap(e: dict) -> str:
    """A difference (Newcombe or paired formula) with its interval, in points."""
    return f"{e['difference_pct']} ({e['interval_pct'][0]} to {e['interval_pct'][1]})"


def _bt(e: dict | None, gap: bool = False) -> str:
    """A bootstrap figure: a share in percent, or a gap in points."""
    if e is None:
        return "not computable"
    lo, hi = e["bootstrap_pct"]
    return f"{e['point_pct']} ({lo} to {hi})" if gap else f"{e['point_pct']}% ({lo}–{hi})"


def multihop_lines(mh: dict) -> list[str]:
    """Section 6 of LLM_RESULTS.md: the multi-hop part."""
    S, R = SIDES
    kind_name = {"single": "one article", "needs2": "two or more"}
    m, k = mh["meanings"], mh["kinds"]
    cnt = k["counts"]
    L = ["## 6. Multi-hop claims: supported and refuted claims, equal weight", "",
         (f"The {mh['n']:,} supported and refuted claims of the original run; NOT ENOUGH INFO claims "
          "are left out of this section. A group's equal-weight figure is the mean of its supported "
          "and its refuted figure, half each, so that a group's label mix does not move it. The two "
          f"kinds hold {cnt['single'][S]:,} + {cnt['single'][R]:,} and {cnt['needs2'][S]:,} + "
          f"{cnt['needs2'][R]:,} claims, so for them the equal-weight figure is the plain one. "
          "Intervals are 95%: Wilson for one label's accuracy; Newcombe's, from the two Wilson "
          "intervals, for a gap between two sets of claims of one label; the paired formula with "
          "McNemar's exact p for a gap between two conditions on the same claims of one label; a "
          f"paired bootstrap ({mh['bootstrap_resamples']:,} resamples, seed {mh['bootstrap_seed']}, "
          "the claims resampled within each label and group) for every equal-weight figure, gap "
          "and change of a gap. Gaps are in percentage points. Tags with fewer than "
          f"{mh['small_tag_below']} supported/refuted claims get their numbers but no conclusions. "
          f"The challenge tags come from `{mh['claims_file']}` (SHA-256 checked)."),
         "",
         ("Two meanings of \"multi-hop\": **two or more articles** (`kind` `needs2`: every evidence "
          f"set spans two or more pages), {m['two_or_more_articles']:,} claims; the **Multi-hop "
          "Reasoning** challenge tag (the verifying annotator's view of the main difficulty; the "
          f"pick did not balance on it), {m['multi_hop_reasoning_tag']:,} claims. {m['both']:,} "
          f"claims are both ({m['both_per_label'][S]:,} supported, {m['both_per_label'][R]:,} "
          "refuted)."),
         "",
         ("The repeat runs of section 5 hold "
          f"{mh['repeats_per_kind']['single']:,} of these supported/refuted claims on one article "
          f"and {mh['repeats_per_kind']['needs2']:,} on two or more, so this section uses the "
          "original run only, and its subgroup gaps have no rerun SD of their own."),
         "", "### One article against two or more", "",
         "| condition | claims | SUPPORTS | REFUTES | equal weight |", "|---|---|---|---|---|"]
    for c in CONDITIONS:
        for kd in KINDS:
            e = k["accuracy"][c][kd]
            L.append(f"| {NAMES[c]} | {kind_name[kd]} ({e[S]['n']:,} + {e[R]['n']:,}) | {_iv(e[S])} "
                     f"| {_iv(e[R])} | {_bt(e['equal_weight'])} |")
    L += ["", "Gap, one article minus two or more:", "",
          "| condition | SUPPORTS (Newcombe) | REFUTES (Newcombe) | equal weight (bootstrap) |",
          "|---|---|---|---|"]
    for c in CONDITIONS:
        g = k["gap_one_minus_two"][c]
        L.append(f"| {NAMES[c]} | {_gap(g[S])} | {_gap(g[R])} | {_bt(g['equal_weight'], True)} |")
    L += ["", "Paired gaps within each kind (RAG minus the other condition, on the same claims):", "",
          "| pair | claims | SUPPORTS (paired formula, McNemar p) | REFUTES (paired formula, McNemar p) | equal weight (bootstrap) |",
          "|---|---|---|---|---|"]
    for a, b in RAG_PAIRS:
        for kd in KINDS:
            e = k["paired"][kd][f"{a}-{b}"]
            L.append(f"| {NAMES[a]} − {NAMES[b]} | {kind_name[kd]} | {_gap(e[S])}, p "
                     f"{e[S]['mcnemar_p_text']} | {_gap(e[R])}, p {e[R]['mcnemar_p_text']} | "
                     f"{_bt(e['equal_weight'], True)} |")
    L += ["", ("Change of each paired gap from one article to two or more (the gap on two or more "
               "minus the gap on one article; bootstrap):"), "",
          "| pair | SUPPORTS | REFUTES | equal weight |", "|---|---|---|---|"]
    for a, b in RAG_PAIRS:
        ch = k["paired_change_two_minus_one"][f"{a}-{b}"]
        L.append(f"| {NAMES[a]} − {NAMES[b]} | {_bt(ch[S], True)} | {_bt(ch[R], True)} | "
                 f"{_bt(ch['equal_weight'], True)} |")
    cov = mh["coverage"]
    L += ["", "### What the five RAG passages covered, by kind", "",
          ("Pages, not sentences: a claim counts for \"the whole gold set\" when every page of the "
           "evidence set shown in the gold condition is among the pages of the five passages. "
           "Whether the passages held the gold sentences themselves was not measured."), "",
          "| claims | measure | SUPPORTS | REFUTES | equal weight |", "|---|---|---|---|---|"]
    rows = (("gold_page_in_top5", "a gold page in the top 5 (share of claims)"),
            ("gold_set_in_top5", "the whole gold set in the top 5 (share of claims)"),
            ("rag_when_set_in_top5", "RAG accuracy when the whole gold set is in the top 5"),
            ("rag_when_set_not_in_top5", "RAG accuracy when it is not"))
    for kd in KINDS:
        for name, title in rows:
            e = cov[kd][name]
            L.append(f"| {kind_name[kd]} | {title} | {_acc(e[S])} | {_acc(e[R])} | "
                     f"{_bt(e['equal_weight'])} |")
        g = cov[kd]["rag_set_in_minus_not"]
        L.append(f"| {kind_name[kd]} | RAG accuracy, gold set in minus not in (points; Newcombe, "
                 "bootstrap) | " + " | ".join(_gap(g[label]) if label in g else "not computable"
                                              for label in SIDES)
                 + f" | {_bt(g['equal_weight'], True)} |")
    tags = mh["tags"]
    L += ["", "### The challenge tags", "",
          "| tag | claims (S + R) | SUPPORTS: one article / two or more | REFUTES: one article / two or more | |",
          "|---|---|---|---|---|"]
    for t in mh["tag_order"]:
        e = tags[t]
        L.append(f"| {t} | {e['n']:,} ({e['n_per_label'][S]:,} + {e['n_per_label'][R]:,}) | "
                 f"{e['counts'][S]['single']:,} / {e['counts'][S]['needs2']:,} "
                 f"| {e['counts'][R]['single']:,} / {e['counts'][R]['needs2']:,} | "
                 f"{'fewer than ' + str(mh['small_tag_below']) + ': numbers only' if e['small'] else ''} |")
    L += ["", "Equal-weight accuracy per tag (bootstrap interval):", "",
          "| tag | " + " | ".join(NAMES[c] for c in CONDITIONS) + " |",
          "|---|" + "---|" * len(CONDITIONS)]
    for t in mh["tag_order"]:
        L.append(f"| {t} | " + " | ".join(_bt(tags[t]["accuracy"][c]["equal_weight"])
                                          for c in CONDITIONS) + " |")
    L += ["", "Accuracy per tag and label, SUPPORTS / REFUTES (Wilson intervals in the JSON):", "",
          "| tag | " + " | ".join(NAMES[c] for c in CONDITIONS) + " |",
          "|---|" + "---|" * len(CONDITIONS)]
    for t in mh["tag_order"]:
        L.append(f"| {t} | " + " | ".join(
            f"{tags[t]['accuracy'][c][S]['pct']} / {tags[t]['accuracy'][c][R]['pct']}"
            for c in CONDITIONS) + " |")
    mvo = mh.get("multihop_vs_other")
    if not mvo:
        return [*L, ""]
    mt, ot = MULTIHOP_TAG, OTHER_TAG
    L += ["", f"### {mt} against {ot}", "",
          (f"Per condition: SUPPORTS / REFUTES accuracy and the equal-weight accuracy of each tag, "
           f"overall and within each kind, and the gap {mt} minus {ot}."), "",
          f"| claims ({mt} S + R; {ot} S + R) | condition | {mt} | {ot} | gap SUPPORTS / REFUTES (Newcombe) | gap, equal weight (bootstrap) |",
          "|---|---|---|---|---|---|"]
    for scope in ("all", *KINDS):
        for c in CONDITIONS:
            per = mvo["accuracy"][scope][c]
            g = mvo["gap"][scope][c]
            title = "all" if scope == "all" else kind_name[scope]
            sizes = (f"{per[mt][S]['n']:,} + {per[mt][R]['n']:,}; "
                     f"{per[ot][S]['n']:,} + {per[ot][R]['n']:,}")
            cells = [f"{per[t][S]['pct']} / {per[t][R]['pct']}; {_bt(per[t]['equal_weight'])}"
                     for t in (mt, ot)]
            label_gaps = " / ".join(_gap(g[label]) if label in g else "not computable"
                                    for label in SIDES)
            L.append(f"| {title} ({sizes}) | {NAMES[c]} | {cells[0]} | {cells[1]} | {label_gaps} | "
                     f"{_bt(g['equal_weight'], True)} |")
    rg, diff = mvo["rag_minus_gold"], mvo["rag_minus_gold_difference"]
    L += ["", ("RAG minus gold within each tag, on the same claims, and the difference between the "
               f"two tags' gaps ({mt} minus {ot}; bootstrap):"), "",
          "| tag | SUPPORTS (paired formula, McNemar p) | REFUTES (paired formula, McNemar p) | equal weight (bootstrap) |",
          "|---|---|---|---|"]
    for t in (mt, ot):
        e = rg[t]
        L.append(f"| {t} | {_gap(e[S])}, p {e[S]['mcnemar_p_text']} | {_gap(e[R])}, p "
                 f"{e[R]['mcnemar_p_text']} | {_bt(e['equal_weight'], True)} |")
    L.append(f"| difference, {mt} − {ot} | {_bt(diff[S], True)} | {_bt(diff[R], True)} | "
             f"{_bt(diff['equal_weight'], True)} |")
    return [*L, ""]


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="run folders -> per_claim.csv, retrieval_top5.csv, runs.json")
    t.add_argument("--runs-root", required=True, type=Path, help="folder of the run folders")
    t.add_argument("--claims", required=True, type=Path, help="claims_3000.jsonl")
    t.add_argument("--out", type=Path, default=OUT_DIR)
    t.add_argument("--nei-audit", type=Path, default=None,
                   help="the NOT ENOUGH INFO audit's JSON (its counts go into runs.json)")
    t.add_argument("--web-audit", type=Path, default=None,
                   help="the web audit's JSON (its counts go into runs.json)")
    t.add_argument("--runs", nargs="+", default=list(RUNS), choices=RUNS,
                   help="runs to include (default: all three)")
    s = sub.add_parser("stats", help="committed inputs -> llm_results.json and LLM_RESULTS.md")
    s.add_argument("--out", type=Path, default=OUT_DIR)
    s.add_argument("--claims", type=Path, default=CLAIMS_FILE,
                   help="claims_3000.jsonl, for the challenge tags (its SHA-256 is checked)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    return cmd_table(args) if args.cmd == "table" else cmd_stats(args)


if __name__ == "__main__":
    sys.exit(main())
