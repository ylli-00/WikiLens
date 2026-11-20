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
3. the overall numbers as measured and reweighted to FEVEROUS's mix of the six groups.

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
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "analysis" / "feverous" / "feverous_claims.csv"
LABELS: tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
KINDS: tuple[str, ...] = ("single", "needs2")
KIND_LABEL = {"single": "one article", "needs2": "two or more"}
KS: tuple[int, ...] = (1, 3, 5, 10, 20)
TAG_K = 5
TEXT_MODALITIES = frozenset({"sentence-only", "list-only"})


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


def measure(claims: dict[int, dict[str, Any]], per_claim: list[dict[str, Any]],
            sentences: dict[int, dict[str, Any]] | None) -> dict[str, Any]:
    """Per group and per tag: claim counts and rates; ``claims`` is keyed by FEVEROUS id."""
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
    return "\n".join(lines)


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
    databases = {}
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
