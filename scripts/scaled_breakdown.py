#!/usr/bin/env python3
"""The scaled run's recall split by claim group and challenge tag, and reweighted to FEVEROUS.

Reads, without a database:

- the 3,000 claims (``claims_3000.jsonl``: FEVEROUS id, label, kind, challenge tag);
- per database, the per-claim file of a ``wikilense eval`` run (``gold_page_rank`` and
  ``evidence_rank`` of every claim, from its LIMIT 20 search; a claim counts at k when the
  rank is at most k, which reproduces the run's own counts per k);
- optionally, per database, the per-claim file of ``scripts/scaled_sentence_score.py``
  (``in_top_chunks``: a complete sentence-only gold set in the top 10 chunks;
  ``found_embed``: the FEVEROUS-style score, best 5 sentences + 25 list items by embedding);
- the FEVEROUS annotation table ``analysis/feverous/feverous_claims.csv`` (one row per train
  and dev claim), for the weights.

and writes one JSON and one Markdown file with:

1. recall in the six groups (label x one article or two or more), at k 1, 3, 5, 10, 20, and the
   two evidence scores of the sentence file;
2. recall per FEVEROUS challenge tag, at k = 5, and the two evidence scores;
3. the overall numbers as measured and reweighted to FEVEROUS's mix of the six groups;
4. the label check: per database, metric (article@5, evidence@5, the FEVEROUS-style score) and
   kind, every pair of labels compared with Newcombe's 95% interval. The kinds hold 700 or 300
   claims per label, so a difference there is a label effect, not a mix effect. A metric
   differs by label when any pair's interval excludes 0, in either kind. Only for those
   metrics come equal-weight figures: per challenge tag the mean of the tag's per-label rates,
   a third each, and Multi-hop Reasoning against Other, overall and within each kind. Their
   intervals are the paired bootstrap of ``scripts/llm_results.py`` (claims resampled within
   each label and tag, or label, tag and kind), with its seed for the 3,000 claims.

The weights
    Population: every row of ``feverous_claims.csv`` (train + dev). A claim is in group
    (label, one article) when every evidence set lies on one page (``max_pages`` = 1), and in
    (label, two or more) when every set spans two or more pages (``min_pages`` >= 2); a claim
    with sets of both kinds is in no group. Two weightings:

    - ``all``: each group's share of the claims in the six groups;
    - ``text``: the same, counting only claims with an evidence set of sentences only or of
      list items only (the table's ``modality_summary``; a set that mixes sentences and list
      items is labelled ``mixed`` there and is not counted).

    The reweighted value is ``sum over groups of weight x measured rate of the group``. The
    claim pick's own kind "one article" also requires every set to name the same page, which
    the table cannot check; its two-or-more counts equal the table's.

Usage:
  scripts/scaled_breakdown.py --claims <claims_3000.jsonl> --eval S0=<eval.json> \\
      --eval S1=<eval.json> [--sentences S0=<claims.jsonl> ...] --out <file.json>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "analysis" / "feverous" / "feverous_claims.csv"
LABELS: tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
KINDS: tuple[str, ...] = ("single", "needs2")
KIND_LABEL = {"single": "one article", "needs2": "two or more"}
KS: tuple[int, ...] = (1, 3, 5, 10, 20)
TAG_K = 5
TEXT_MODALITIES = frozenset({"sentence-only", "list-only"})
#: The metrics of the label check, and the two tags compared after it.
CHECK_METRICS: tuple[str, ...] = (f"article@{TAG_K}", f"evidence@{TAG_K}", "feverous_style")
MULTIHOP_TAG, OTHER_TAG = "Multi-hop Reasoning", "Other"
#: Tags with fewer claims than this get their numbers but no conclusions.
SMALL_TAG = 100


def _stats_module() -> Any:
    """``scripts/llm_results.py`` as a module, for its intervals and its bootstrap."""
    spec = importlib.util.spec_from_file_location("llm_results",
                                                  Path(__file__).with_name("llm_results.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATS = _stats_module()


def csv_kind(min_pages: int, max_pages: int) -> str | None:
    """The group kind of one FEVEROUS claim from its smallest and largest evidence set."""
    if max_pages == 1:
        return "single"
    if min_pages >= 2:
        return "needs2"
    return None


def feverous_weights(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Counts and weights of the six groups among FEVEROUS claims (see the module docstring)."""
    counts: dict[str, Counter[tuple[str, str]]] = {"all": Counter(), "text": Counter()}
    n_rows = 0
    for row in rows:
        n_rows += 1
        kind = csv_kind(int(row["min_pages"]), int(row["max_pages"]))
        if kind is None:
            continue
        group = (row["label"], kind)
        counts["all"][group] += 1
        if TEXT_MODALITIES & set(row["modality_summary"].split("|")):
            counts["text"][group] += 1
    out: dict[str, Any] = {"n_rows": n_rows}
    for name, counter in counts.items():
        total = sum(counter.values())
        out[name] = {
            "total": total,
            "groups": {f"{label}|{kind}": {"count": counter[(label, kind)],
                                           "weight": counter[(label, kind)] / total}
                       for label in LABELS for kind in KINDS},
        }
    return out


def rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def claim_rows(claims: dict[int, dict[str, Any]], per_claim: list[dict[str, Any]],
               sentences: dict[int, dict[str, Any]] | None) -> list[dict[str, Any]]:
    """One row per claim: its group (``label|kind``), challenge tag and every metric (bool)."""
    rows = []
    for item in per_claim:
        claim = claims[int(item["claim_id"])]
        row = {
            "group": f"{claim['label']}|{claim['kind']}",
            "challenge": claim["challenge"],
            **{f"article@{k}": item["gold_page_rank"] is not None and item["gold_page_rank"] <= k
               for k in KS},
            **{f"evidence@{k}": item["evidence_rank"] is not None and item["evidence_rank"] <= k
               for k in KS},
        }
        if sentences is not None:
            found = sentences[int(item["claim_id"])]
            row["complete_set_top10_chunks"] = bool(found["in_top_chunks"])
            row["feverous_style"] = bool(found["found_embed"])
        rows.append(row)
    return rows


def measure(claims: dict[int, dict[str, Any]], per_claim: list[dict[str, Any]],
            sentences: dict[int, dict[str, Any]] | None) -> dict[str, Any]:
    """Per group and per tag: claim counts and rates; ``claims`` is keyed by FEVEROUS id."""
    rows = claim_rows(claims, per_claim, sentences)
    metrics = [key for key in rows[0] if key not in ("group", "challenge")]

    def summarise(selected: list[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
        return {"n": len(selected), **{key: rate([r[key] for r in selected]) for key in keys}}

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
        by_tag[row["challenge"]].append(row)
    tag_keys = [f"article@{TAG_K}", f"evidence@{TAG_K}",
                *[m for m in metrics if not m.startswith(("article@", "evidence@"))]]
    return {
        "metrics": metrics,
        "overall": summarise(rows, metrics),
        "groups": {f"{label}|{kind}": summarise(by_group[f"{label}|{kind}"], metrics)
                   for label in LABELS for kind in KINDS},
        "tags": {tag: summarise(selected, tag_keys)
                 for tag, selected in sorted(by_tag.items(), key=lambda kv: -len(kv[1]))},
    }


def reweight(groups: dict[str, Any], weights: dict[str, Any], metric: str) -> float:
    return sum(weights["groups"][g]["weight"] * groups[g][metric] for g in weights["groups"])


def _rate_entry(hits: int, n: int) -> dict[str, Any]:
    lo, hi = STATS.wilson(hits, n)
    return {"hits": hits, "n": n, "rate": hits / n if n else None,
            "wilson": None if n == 0 else [lo, hi]}


def _diff_entry(first: tuple[int, int], second: tuple[int, int]) -> dict[str, Any]:
    """Two rates on different claims, first minus second, with Newcombe's interval."""
    diff, lo, hi = STATS.newcombe(*first, *second)
    known = not np.isnan(lo)
    return {"first": _rate_entry(*first), "second": _rate_entry(*second),
            "difference": diff if known else None, "interval": [lo, hi] if known else None,
            "excludes_zero": (not lo <= 0 <= hi) if known else None}


def _boot_entry(point: float, values: np.ndarray, gap: bool = False) -> dict[str, Any]:
    lo, hi = (float(x) for x in np.percentile(values, [2.5, 97.5]))
    out: dict[str, Any] = {"point": point, "bootstrap": [lo, hi]}
    if gap:
        out["excludes_zero"] = not lo <= 0 <= hi
    return out


def label_weighting(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Item 4 for one database (see the module docstring): the label check, and for the metrics
    that differ by label the equal-weight figures per tag and for Multi-hop Reasoning against
    Other. A tag without claims of some label has no equal-weight figure (None)."""
    n = len(rows)
    labels = np.array([LABELS.index(r["group"].split("|")[0]) for r in rows])
    kinds = np.array([KINDS.index(r["group"].split("|")[1]) for r in rows])
    order = Counter(r["challenge"] for r in rows)
    tags = sorted(order, key=lambda t: (-order[t], t))
    tag = np.array([tags.index(r["challenge"]) for r in rows])
    values = {m: np.array([int(r[m]) for r in rows]) for m in CHECK_METRICS}

    def counts(metric: str, mask: np.ndarray) -> tuple[int, int]:
        return int(values[metric][mask].sum()), int(mask.sum())

    check: dict[str, Any] = {}
    for m in CHECK_METRICS:
        check[m] = {}
        for ki, kind in enumerate(KINDS):
            in_kind = kinds == ki
            check[m][kind] = {
                f"{a}|{b}": _diff_entry(counts(m, in_kind & (labels == ai)),
                                        counts(m, in_kind & (labels == bi)))
                for (ai, a), (bi, b) in combinations(enumerate(LABELS), 2)}
    differs = {m: any(e["excludes_zero"] for per_kind in check[m].values()
                      for e in per_kind.values()) for m in CHECK_METRICS}
    weighted = [m for m in CHECK_METRICS if differs[m]]
    out: dict[str, Any] = {"check": check, "differs": differs, "weighted": weighted,
                           "tag_order": tags, "tags": {}, "multihop_vs_other": None}
    if not weighted:
        return out

    def nonempty(masks: dict) -> dict:
        return {key: mask for key, mask in masks.items() if mask.any()}

    def ew_values(boot: dict, metric: str, *key: Any) -> np.ndarray:
        return sum(boot[(metric, (label, *key))] for label in LABELS) / len(LABELS)

    def equal_weight(metric: str, mask: np.ndarray, boot: dict, *key: Any) -> dict[str, Any]:
        per = {label: _rate_entry(*counts(metric, mask & (labels == li)))
               for li, label in enumerate(LABELS)}
        computable = all(per[label]["n"] for label in LABELS)
        return {"plain": _rate_entry(*counts(metric, mask)), "per_label": per,
                "equal_weight": (_boot_entry(sum(per[label]["rate"] for label in LABELS) / 3,
                                             ew_values(boot, metric, *key))
                                 if computable else None)}

    tmasks = {(label, t): (labels == li) & (tag == ti)
              for li, label in enumerate(LABELS) for ti, t in enumerate(tags)}
    weighted_values = {m: values[m] for m in weighted}
    tboot = STATS.boot_means(weighted_values, nonempty(tmasks), labels * len(tags) + tag, seed)
    for ti, t in enumerate(tags):
        in_tag = tag == ti
        out["tags"][t] = {
            "n": int(in_tag.sum()), "small": bool(in_tag.sum() < SMALL_TAG),
            "per_label_n": {label: int(tmasks[(label, t)].sum()) for label in LABELS},
            "metrics": {m: equal_weight(m, in_tag, tboot, t) for m in weighted}}
    if MULTIHOP_TAG not in tags or OTHER_TAG not in tags:
        return out
    pair = (MULTIHOP_TAG, OTHER_TAG)
    dmasks = {(label, t, kind): tmasks[(label, t)] & (kinds == ki)
              for label in LABELS for t in pair for ki, kind in enumerate(KINDS)}
    dboot = STATS.boot_means(weighted_values, nonempty(dmasks),
                             (labels * len(tags) + tag) * 2 + kinds, seed)
    mvo: dict[str, Any] = {}
    for m in weighted:
        mvo[m] = {}
        for scope in ("all", *KINDS):
            in_scope = np.ones(n, dtype=bool) if scope == "all" else kinds == KINDS.index(scope)
            boot, key = (tboot, ()) if scope == "all" else (dboot, (scope,))
            per_tag = {t: equal_weight(m, in_scope & (tag == tags.index(t)), boot, t, *key)
                       for t in pair}
            first, second = (per_tag[t] for t in pair)
            gap_ew = None
            if first["equal_weight"] and second["equal_weight"]:
                gap_ew = _boot_entry(
                    first["equal_weight"]["point"] - second["equal_weight"]["point"],
                    ew_values(boot, m, pair[0], *key) - ew_values(boot, m, pair[1], *key), gap=True)
            mvo[m][scope] = {
                **per_tag,
                "gap_plain": _diff_entry((first["plain"]["hits"], first["plain"]["n"]),
                                         (second["plain"]["hits"], second["plain"]["n"])),
                "gap_equal_weight": gap_ew}
    out["multihop_vs_other"] = mvo
    return out


def pairs(values: Sequence[str]) -> dict[str, Path]:
    out = {}
    for value in values:
        name, _, path = value.partition("=")
        if not name or not path:
            raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
        out[name] = Path(path)
    return out


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def to_markdown(report: dict[str, Any]) -> str:
    w = report["weights"]
    lines = [
        "# Scaled run: recall by claim group and challenge tag, and reweighted to FEVEROUS",
        "",
        (f"Written {report['created_at']} by `scripts/scaled_breakdown.py` from the per-claim "
         "files of the runs named below (no database). A claim counts at k when its rank in "
         "the run's LIMIT 20 search is at most k; that reproduces the run's own counts per k. "
         "\"Complete set in top 10 chunks\" and \"FEVEROUS-style\" are the two evidence scores "
         "of `scripts/scaled_sentence_score.py`."),
        "",
        "Inputs: " + "; ".join(f"{name}: `{path}`" for name, path in report["inputs"].items()) + ".",
        "",
        "## Weights: FEVEROUS's mix of the six groups",
        "",
        (f"From `analysis/feverous/feverous_claims.csv` ({w['n_rows']:,} train + dev claims). A "
         "claim is in (label, one article) when every evidence set lies on one page "
         "(`max_pages` = 1), in (label, two or more) when every set spans two or more pages "
         "(`min_pages` >= 2); claims with sets of both kinds are in no group. `all` counts every "
         "such claim; `text` only those with an evidence set of sentences only or list items "
         "only. Reweighted = sum over the groups of weight x the group's measured rate."),
        "",
        "| Group | Claims here | FEVEROUS, all | weight | FEVEROUS, text | weight |",
        "|---|---|---|---|---|---|",
    ]
    first = next(iter(report["databases"].values()))
    for g in w["all"]["groups"]:
        label, kind = g.split("|")
        lines.append(
            f"| {label}, {KIND_LABEL[kind]} | {first['groups'][g]['n']} | "
            f"{w['all']['groups'][g]['count']:,} | {w['all']['groups'][g]['weight']:.4f} | "
            f"{w['text']['groups'][g]['count']:,} | {w['text']['groups'][g]['weight']:.4f} |")
    lines += [(f"| total | {first['overall']['n']} | {w['all']['total']:,} | 1 | "
               f"{w['text']['total']:,} | 1 |"), ""]
    for name, data in report["databases"].items():
        metrics = data["metrics"]
        extra = [m for m in metrics if not m.startswith(("article@", "evidence@"))]
        lines += [f"## {name}", "", "### Six groups", "",
                  "| Group | n | " + " | ".join(f"article@{k}" for k in KS) + " | "
                  + " | ".join(f"evidence@{k}" for k in KS)
                  + "".join(f" | {m}" for m in extra) + " |",
                  "|---|---|" + "---|" * (2 * len(KS) + len(extra))]
        for g, values in data["groups"].items():
            label, kind = g.split("|")
            lines.append(f"| {label}, {KIND_LABEL[kind]} | {values['n']} | "
                         + " | ".join(fmt(values[m]) for m in metrics) + " |")
        overall = data["overall"]
        lines.append(f"| all claims, as measured | {overall['n']} | "
                     + " | ".join(fmt(overall[m]) for m in metrics) + " |")
        for wname in ("all", "text"):
            lines.append(f"| reweighted to FEVEROUS ({wname}) | - | "
                         + " | ".join(fmt(data["reweighted"][wname][m]) for m in metrics) + " |")
        tag_keys = [k for k in next(iter(data["tags"].values())) if k != "n"]
        lines += ["", f"### Challenge tags (k = {TAG_K} for the recall columns)", "",
                  "| Challenge tag | n | " + " | ".join(tag_keys) + " |",
                  "|---|---|" + "---|" * len(tag_keys)]
        for tag, values in data["tags"].items():
            lines.append(f"| {tag} | {values['n']} | "
                         + " | ".join(fmt(values[k]) for k in tag_keys) + " |")
        lines.append("")
    if "label_weighting" in report:
        lines += label_markdown(report["label_weighting"])
    return "\n".join(lines)


def _with_wilson(e: dict[str, Any]) -> str:
    if not e["n"]:
        return "-"
    return f"{e['rate']:.3f} ({e['wilson'][0]:.3f}–{e['wilson'][1]:.3f})"


def _with_boot(e: dict[str, Any] | None, gap: bool = False) -> str:
    if e is None:
        return "not computable"
    lo, hi = e["bootstrap"]
    return f"{e['point']:.3f} ({lo:.3f} to {hi:.3f})" if gap else f"{e['point']:.3f} ({lo:.3f}–{hi:.3f})"


def _with_newcombe(e: dict[str, Any]) -> str:
    if e["difference"] is None:
        return "not computable"
    return f"{e['difference']:.3f} ({e['interval'][0]:.3f} to {e['interval'][1]:.3f})"


def label_markdown(lw: dict[str, Any]) -> list[str]:
    """The Markdown section of item 4."""
    short = {"SUPPORTS": "S", "REFUTES": "R", "NOT ENOUGH INFO": "NEI"}
    lines = [
        "## Label check and equal-weight figures", "",
        ("Does the search do worse on some labels than on others? All three labels stay in, as "
         "everywhere in this file. Step 1 compares, per database, metric and kind, every pair of "
         "labels with Newcombe's 95% interval (from the two Wilson intervals). Within a kind every "
         "label has 700 or 300 claims, so a difference there is a label effect, not a mix effect. "
         "A metric differs by label when any pair's interval excludes 0, in either kind. Step 2, "
         "only for those metrics: the equal-weight figure of a group is the mean of its "
         "SUPPORTS, REFUTES and NOT ENOUGH INFO rates, a third each, so that the group's label "
         "mix does not move it; a group without claims of some label has none (\"not "
         "computable\"). Its interval is a paired bootstrap "
         f"({lw['bootstrap']['resamples']:,} resamples, seed {lw['bootstrap']['seed']}, the "
         "claims resampled within each label and tag, or label, tag and kind). Tags with fewer "
         f"than {lw['small_tag_below']} claims get their numbers but no conclusions."),
        "", f"### Step 1: the three labels within each kind (k = {TAG_K})", "",
        "| Database | Metric | Kind | Pair | First | Second | Difference (95% interval) | Excludes 0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, data in lw["databases"].items():
        for m in lw["metrics"]:
            for kind in KINDS:
                for pair, e in data["check"][m][kind].items():
                    a, b = pair.split("|")
                    lines.append(
                        f"| {name} | {m} | {KIND_LABEL[kind]} | {short[a]} − {short[b]} | "
                        f"{_with_wilson(e['first'])} | {_with_wilson(e['second'])} | "
                        f"{_with_newcombe(e)} | {'yes' if e['excludes_zero'] else 'no'} |")
    lines.append("")
    for name, data in lw["databases"].items():
        kept = [m for m in lw["metrics"] if not data["differs"][m]]
        lines.append(f"{name}: differs by label: {', '.join(data['weighted']) or 'none'}. "
                     + ("Left unweighted, because no pair's interval excludes 0 in either kind: "
                        + ", ".join(kept) + "." if kept else "No metric is left unweighted."))
        lines.append("")
    for name, data in lw["databases"].items():
        weighted = data["weighted"]
        if not weighted:
            continue
        lines += [f"### Step 2, {name}: equal-weight figures per challenge tag (k = {TAG_K})", "",
                  "| Challenge tag | n (S / R / NEI) | " + " | ".join(
                      f"{m} | {m}, equal weight" for m in weighted) + " |",
                  "|---|---|" + "---|---|" * len(weighted)]
        for t in data["tag_order"]:
            e = data["tags"][t]
            counts = " / ".join(str(e["per_label_n"][label]) for label in LABELS)
            note = f" (fewer than {lw['small_tag_below']})" if e["small"] else ""
            lines.append(f"| {t}{note} | {e['n']} ({counts}) | " + " | ".join(
                f"{fmt(e['metrics'][m]['plain']['rate'])} | "
                f"{_with_boot(e['metrics'][m]['equal_weight'])}" for m in weighted) + " |")
        mvo = data["multihop_vs_other"]
        if mvo:
            lines += ["", f"### Step 2, {name}: {MULTIHOP_TAG} against {OTHER_TAG}", "",
                      (f"Plain: the share of the tag's claims, Wilson interval, gap with Newcombe's "
                       f"interval. Equal weight: bootstrap. The gaps are {MULTIHOP_TAG} minus "
                       f"{OTHER_TAG}."), "",
                      (f"| Metric | Claims | {MULTIHOP_TAG}, n (S / R / NEI) | {OTHER_TAG}, n | "
                       f"{MULTIHOP_TAG} | {OTHER_TAG} | Gap | {MULTIHOP_TAG}, equal weight | "
                       f"{OTHER_TAG}, equal weight | Gap, equal weight |"),
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for m in weighted:
                for scope in ("all", *KINDS):
                    e = mvo[m][scope]
                    first, second = e[MULTIHOP_TAG], e[OTHER_TAG]
                    sizes = [f"{x['plain']['n']} ({' / '.join(str(x['per_label'][label]['n']) for label in LABELS)})"
                             for x in (first, second)]
                    lines.append(
                        f"| {m} | {'all' if scope == 'all' else KIND_LABEL[scope]} | {sizes[0]} | "
                        f"{sizes[1]} | {_with_wilson(first['plain'])} | "
                        f"{_with_wilson(second['plain'])} | {_with_newcombe(e['gap_plain'])} | "
                        f"{_with_boot(first['equal_weight'])} | {_with_boot(second['equal_weight'])} "
                        f"| {_with_boot(e['gap_equal_weight'], gap=True)} |")
        lines.append("")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--eval", required=True, action="append", metavar="NAME=PATH")
    parser.add_argument("--sentences", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--feverous-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    evals, sentence_files = pairs(args.eval), pairs(args.sentences)
    claims = {}
    with args.claims.open(encoding="utf-8") as fh:
        for line in fh:
            claim = json.loads(line)
            claims[int(claim["id"])] = claim
    with args.feverous_csv.open(encoding="utf-8", newline="") as fh:
        weights = feverous_weights(csv.DictReader(fh))
    databases, label_check = {}, {}
    seed = STATS.SEEDS["all"]
    for name, path in evals.items():
        per_claim = json.loads(path.read_text(encoding="utf-8"))["claims"]
        sentences = None
        if name in sentence_files:
            with sentence_files[name].open(encoding="utf-8") as fh:
                sentences = {int(r["claim_id"]): r for r in map(json.loads, fh)}
        data = measure(claims, per_claim, sentences)
        data["reweighted"] = {wname: {m: reweight(data["groups"], weights[wname], m)
                                      for m in data["metrics"]} for wname in ("all", "text")}
        databases[name] = data
        if all(m in data["metrics"] for m in CHECK_METRICS):
            label_check[name] = label_weighting(claim_rows(claims, per_claim, sentences), seed)

    def shown(path: Path) -> str:
        """The input's name relative to the repository, or its last two parts."""
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return "/".join(path.parts[-2:])

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"claims": shown(args.claims), "FEVEROUS table": shown(args.feverous_csv),
                   **{f"eval {n}": shown(p) for n, p in evals.items()},
                   **{f"sentences {n}": shown(p) for n, p in sentence_files.items()}},
        "weights": weights, "databases": databases,
    }
    if label_check:
        report["label_weighting"] = {
            "metrics": list(CHECK_METRICS),
            "rule": "a metric differs by label when, within either kind, some pair of labels has a "
                    "Newcombe 95% interval that excludes 0; only those metrics get equal-weight "
                    "figures (the mean of the three per-label rates)",
            "bootstrap": {"resamples": STATS.BOOTSTRAP_RESAMPLES, "seed": seed,
                          "strata": "label x tag, or label x tag x kind"},
            "small_tag_below": SMALL_TAG, "databases": label_check}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
