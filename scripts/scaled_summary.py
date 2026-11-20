#!/usr/bin/env python3
"""``results/scaled/SUMMARY.md``: every table of the scaled run, from the files in ``results/scaled/``.

Reads only result files (no database, no model) and writes one Markdown file. A measurement that
was run several times gets one column per run, so a later rerun only needs its files copied in
and this script run again. A part with no file yet is listed as not measured.

Layout read (``<db>`` is ``s0`` or ``s1``, ``<n>`` the run number):

- ``eval/<db>_full_run<n>.json``: ``wikilense eval`` (recall and latency, k 1, 3, 5, 10, 20)
- ``timing/scale_k5_<db>_run<n>.json``: ``scripts/scaled_timing.py`` (fast vs slow statements)
- ``sentence_score/<db>/summary.json``, ``page_coverage/page_coverage_<db>.json``
- ``breakdown/breakdown.json``: ``scripts/scaled_breakdown.py``
- ``filters/filters_<db>.json``: ``scripts/scaled_filters.py``
- ``run<n>/<db>/``: ``join_order``, ``statement_forms``, ``cache_size``, ``build_variance``
  (``scripts/measure_index_behaviour.py``), ``separate_index``
  (``scripts/compare_separate_index.py``), ``uncommitted_recall`` (``scripts/scaled_uncommitted.py``)
- ``build_timing/<db>/build_variance.json``: the single timed build

Usage: .venv/bin/python scripts/scaled_summary.py [--dir results/scaled]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "results" / "scaled"
DATABASES: tuple[str, ...] = ("s0", "s1")
NAMES = {"s0": "S0", "s1": "S1"}
NOT_YET = "Not measured yet."


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def runs_of(base: Path, pattern: str) -> dict[int, Path]:
    """``{run number: path}`` of the files matching ``pattern`` (``{n}`` marks the number)."""
    regex = re.compile(re.escape(pattern).replace(r"\{n\}", r"(\d+)") + "$")
    found = {}
    for path in base.glob(pattern.replace("{n}", "*")):
        match = regex.search(path.relative_to(base).as_posix())
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return "-" if value is None else str(value).replace("|", "/")
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
            *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)]


def ms(value: float | None) -> str | None:
    return None if value is None else f"{value:.2f}"


def per_run(values: dict[int, Any], fmt: Callable[[Any], str | None] = str) -> str:
    """One cell holding the value of every run: ``a / b`` (run 1 / run 2)."""
    return " / ".join(fmt(v) or "-" for v in values.values()) if values else "-"


def largest_difference(values: Sequence[float]) -> float:
    """The largest relative difference between two of ``values``: (max - min) / min."""
    return (max(values) - min(values)) / min(values) if len(values) > 1 else 0.0


def runs_note(numbers: Sequence[int]) -> str:
    return ("one run (run " + str(numbers[0]) + ")" if len(numbers) == 1
            else "runs " + ", ".join(map(str, numbers)) + "; cells give run " + " / run ".join(map(str, numbers)))


# --------------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------------


def section_eval(base: Path) -> list[str]:
    lines = ["## 1. Recall and latency of the HNSW search (`wikilense eval`)", ""]
    data = {db: {n: load(p) for n, p in runs_of(base, f"eval/{db}_full_run{{n}}.json").items()}
            for db in DATABASES}
    if not any(data.values()):
        return [*lines, NOT_YET, ""]
    first = {db: next(iter(runs.values())) for db, runs in data.items() if runs}
    same = {db: len({json.dumps([(k["k"], k["article_hits"], k["evidence_hits"]) for k in r["per_k"]])
                     for r in runs.values()}) == 1 for db, runs in data.items() if runs}
    params = next(iter(first.values()))["parameters"]
    lines.append(
        f"All {params['n_claims']:,} claims, ef_search {params['ef_search']}, "
        f"{params['repeats']} timed passes per claim after one warm-up. "
        + "; ".join(f"{NAMES[db]}: {runs_note(list(data[db]))}, recall "
                    f"{'identical in every run' if same[db] else 'NOT identical across runs'}"
                    for db in first) + ".")
    lines.append("")
    rows = []
    for i, k_row in enumerate(next(iter(first.values()))["per_k"]):
        row: list[Any] = [k_row["k"]]
        for db in first:
            per_k = first[db]["per_k"][i]
            row += [f"{per_k['article_recall']:.3f} ({per_k['article_hits']:,})",
                    f"{per_k['evidence_recall']:.3f} ({per_k['evidence_hits']:,})",
                    per_run({n: r["per_k"][i]["sql_latency"] for n, r in data[db].items()},
                            lambda s: f"{s['p50_ms']:.2f}") + " ms",
                    per_run({n: r["per_k"][i]["sql_latency"] for n, r in data[db].items()},
                            lambda s: f"{s['p95_ms']:.2f}") + " ms"]
        rows.append(row)
    headers = ["k"]
    for db in first:
        headers += [f"{NAMES[db]} article recall", f"{NAMES[db]} evidence recall",
                    f"{NAMES[db]} SQL p50", f"{NAMES[db]} SQL p95"]
    lines += table(headers, rows)
    for db, runs in data.items():
        if len(runs) > 1:
            diffs = [largest_difference([r["per_k"][i]["sql_latency"][q] for r in runs.values()])
                     for i in range(len(first[db]["per_k"])) for q in ("p50_ms", "p95_ms")]
            lines += ["", (f"{NAMES[db]}: the SQL p50 and p95 differ between runs by at most "
                           f"{max(diffs) * 100:.1f}% (largest over every k).")]
    lines += ["", "Query embedding p50: " + "; ".join(
        f"{NAMES[db]} " + per_run({n: r["embedding_latency"] for n, r in data[db].items()},
                                  lambda s: f"{s['p50_ms']:.2f}") + " ms" for db in first) + ".", ""]
    return lines


TIMING_LABELS = {"hnsw": "HNSW index search", "exact": "exact search (joined statement)",
                 "history_inline": "History filter, inline",
                 "history_overfetch10": "History filter, overfetch 10"}


def section_timing(base: Path) -> list[str]:
    lines = ["## 2. Fast against slow statements (`scripts/scaled_timing.py`, k = 5)", ""]
    data = {db: {n: load(p) for n, p in runs_of(base, f"timing/scale_k5_{db}_run{{n}}.json").items()}
            for db in DATABASES}
    if not any(data.values()):
        return [*lines, NOT_YET, ""]
    first = next(r for runs in data.values() for r in runs.values())
    lines += [f"{first['repeats']} timed runs per claim and statement after one untimed run; "
              + "; ".join(f"{NAMES[db]}: {runs_note(list(runs))}" for db, runs in data.items() if runs)
              + ".", ""]
    rows = []
    for key, label in TIMING_LABELS.items():
        row: list[Any] = [label]
        for db, runs in data.items():
            if runs:
                stmts = {n: r["statements"][key] for n, r in runs.items()}
                row += [per_run(stmts, lambda s: f"{s['sql_ms_p50']:.2f}"),
                        per_run(stmts, lambda s: f"{s['sql_ms_p95']:.2f}"),
                        per_run(stmts, lambda s: f"{s['short_results']:,}")]
        rows.append(row)
    headers = ["statement"]
    for db, runs in data.items():
        if runs:
            headers += [f"{NAMES[db]} p50 ms", f"{NAMES[db]} p95 ms", f"{NAMES[db]} short of 5 rows"]
    lines += table(headers, rows)
    for db, runs in data.items():
        if len(runs) > 1:
            diffs = [largest_difference([r["statements"][key][q] for r in runs.values()])
                     for key in TIMING_LABELS for q in ("sql_ms_p50", "sql_ms_p95")]
            lines += ["", (f"{NAMES[db]}: p50 and p95 differ between runs by at most "
                           f"{max(diffs) * 100:.1f}% (largest over the four statements).")]
    lines += ["", "HNSW top 5 identical to the exact top 5: " + "; ".join(
        f"{NAMES[db]} " + per_run({n: r["statements"]["identical_hnsw_exact"] for n, r in runs.items()},
                                  lambda v: f"{v:,}") for db, runs in data.items() if runs) + ".", ""]
    return lines


def section_scores(base: Path) -> list[str]:
    lines = ["## 3. Evidence scores and page coverage", ""]
    rows = []
    for db in DATABASES:
        summary = load(base / "sentence_score" / db / "summary.json")
        if summary:
            o = summary["overall"]
            rows.append([NAMES[db], f"{o['in_top_chunks']['rate']:.3f} ({o['in_top_chunks']['hits']:,})",
                         f"{o['found_embed']['rate']:.3f} ({o['found_embed']['hits']:,})",
                         f"{o['found_chunk_order']['rate']:.3f} ({o['found_chunk_order']['hits']:,})"])
    if rows:
        lines += [("`scripts/scaled_sentence_score.py`: the text units of each claim's top 10 "
                   "chunks; the best 5 sentences and 25 list items; a claim counts when a complete "
                   "gold set is among them."), ""]
        lines += table(["database", "complete set in the top 10 chunks",
                        "5 + 25 by embedding (FEVEROUS-style)", "5 + 25 in chunk order"], rows)
        lines.append("")
    rows = []
    for db in DATABASES:
        cov = load(base / "page_coverage" / f"page_coverage_{db}.json")
        if cov:
            for subset, label in (("all", "all claims"), ("supports_refutes", "SUPPORTS + REFUTES")):
                if subset in cov:
                    rows.append([f"{NAMES[db]}, {label}",
                                 *(f"{cov[subset][f'p{p}']['coverage']:.3f}" for p in cov["pages"])])
            pages = cov["pages"]
    if rows:
        lines += [("`scripts/scaled_page_coverage.py`: page coverage as in the FEVEROUS baseline "
                   "code (share of a claim's gold pages among the first p distinct retrieved pages, "
                   "mean over claims)."), ""]
        lines += table(["claims", *(f"p = {p}" for p in pages)], rows)
        lines.append("")
    return lines if len(lines) > 2 else [*lines, NOT_YET, ""]


def section_breakdown(base: Path) -> list[str]:
    lines = ["## 4. Six groups, challenge tags, FEVEROUS's mix", ""]
    data = load(base / "breakdown" / "breakdown.json")
    if not data:
        return [*lines, NOT_YET, ""]
    lines += ["Full tables: [breakdown/breakdown.md](breakdown/breakdown.md). At k = 5:", ""]
    rows = []
    for db, d in data["databases"].items():
        for label, values in (("as measured", d["overall"]),
                              ("reweighted, all", d["reweighted"]["all"]),
                              ("reweighted, text", d["reweighted"]["text"])):
            rows.append([db, label, f"{values['article@5']:.3f}", f"{values['evidence@5']:.3f}",
                         f"{values['feverous_style']:.3f}" if "feverous_style" in values else None])
    lines += table(["database", "overall", "article@5", "evidence@5", "FEVEROUS-style"], rows)
    return [*lines, ""]


def section_filters(base: Path) -> list[str]:
    lines = ["## 5. Filter values (`scripts/scaled_filters.py`)", ""]
    data = {db: load(base / "filters" / f"filters_{db}.json") for db in DATABASES}
    if not any(data.values()):
        return [*lines, NOT_YET, ""]
    for db, report in data.items():
        if report:
            lines += [f"{NAMES[db]} ({report['n_chunks']:,} chunks): [filters_{db}.md](filters/filters_{db}.md)."]
    return [*lines, ""]


def side(entry: dict[str, Any]) -> str:
    return entry["plan"].replace("chunk first; chunk index embedding", "vector index")


def section_join_order(base: Path) -> list[str]:
    lines = ["## 6. Filtered searches: the join order fixed and free (`join_order`)", ""]
    any_data = False
    for db in DATABASES:
        runs = {n: load(p) for n, p in runs_of(base, f"run{{n}}/{db}/join_order.json").items()}
        if not runs:
            continue
        any_data = True
        first = next(iter(runs.values()))
        sample = (f"a random sample of {first['n_queries']} of {first['n_claims_total']:,} claims "
                  f"(seed {first['seed']})" if first.get("sample") else f"{first['n_queries']} claims")
        lines += [(f"### {NAMES[db]}: {first['n_chunks']:,} chunks, {sample}, k = {first['k']}, "
                   f"{first['repeats']} timed pass after one untimed pass, {runs_note(list(runs))}"), ""]
        rows = []
        for i, row in enumerate(first["results"]):
            this = {n: r["results"][i] for n, r in runs.items()}
            n_q = first["n_queries"]
            rows.append([
                row["filter"], f"{row['passing_chunks']:,}",
                side(row["as_written"]), f"{row['as_written']['chunk_rows_read_p50']:,.0f}",
                per_run({n: r["as_written"]["p50_ms"] for n, r in this.items()}, ms),
                per_run({n: r["as_written"]["same_rows_as_scan"] for n, r in this.items()},
                        lambda v, q=n_q: f"{v}/{q}"),
                side(row["free"]), f"{row['free']['chunk_rows_read_p50']:,.0f}",
                per_run({n: r["free"]["p50_ms"] for n, r in this.items()}, ms),
                per_run({n: r["free"]["same_rows_as_scan"] for n, r in this.items()},
                        lambda v, q=n_q: f"{v}/{q}"),
            ])
        lines += table(["filter", "chunks passing", "as written: plan", "rows read", "p50 ms",
                        "same rows as scan", "free: plan", "rows read", "p50 ms",
                        "same rows as scan"], rows)
        lines.append("")
    return lines if any_data else [*lines, NOT_YET, ""]


def section_forms(base: Path) -> list[str]:
    lines = ["## 7. Three regimes of one statement (`forms`)", ""]
    any_data = False
    for db in DATABASES:
        runs = {n: load(p) for n, p in runs_of(base, f"run{{n}}/{db}/statement_forms.json").items()}
        if not runs:
            continue
        any_data = True
        first = next(iter(runs.values()))
        n_q = first["n_queries"]
        lines += [(f"### {NAMES[db]}: {first['n_chunks']:,} chunks, {n_q:,} claims, "
                   f"k = {first['k']}, {runs_note(list(runs))}"), ""]
        rows = []
        for i, row in enumerate(first["results"]):
            this = {n: r["results"][i] for n, r in runs.items()}
            rows.append([row["label"], f"{row['passing_chunks']:,}", row["ef_search"],
                         per_run({n: r["p50_ms"] for n, r in this.items()}, ms),
                         per_run({n: r["equal_to_exact"] for n, r in this.items()},
                                 lambda v, q=n_q: f"{v:,}/{q:,}"),
                         per_run({n: r["short"] for n, r in this.items()}, lambda v: f"{v:,}"),
                         f"{row['chunk_rows_read_p50']:,.0f}", row["chunk_access"]])
        lines += table(["statement", "chunks passing", "ef_search", "p50 ms", "equal to exact",
                        "short", "rows read", "chunk access"], rows)
        lines.append("")
    return lines if any_data else [*lines, NOT_YET, ""]


def section_separate(base: Path) -> list[str]:
    lines = ["## 8. MariaDB against a copy of the vectors (`scripts/compare_separate_index.py`)", ""]
    any_data = False
    for db in DATABASES:
        runs = {n: load(p) for n, p in runs_of(base, f"run{{n}}/{db}/separate_index.json").items()}
        if not runs:
            continue
        any_data = True
        first = next(iter(runs.values()))
        params, copy = first["parameters"], first["copy"]
        n_q = params["n_queries"]
        grown = None
        if copy.get("process_rss_bytes_after_export") and copy.get("process_rss_bytes_before_export"):
            grown = copy["process_rss_bytes_after_export"] - copy["process_rss_bytes_before_export"]
        lines += [(f"### {NAMES[db]}: {params['n_chunks']:,} chunks, {n_q:,} claims, "
                   f"k = {params['k']}, {params['repeats']} timed pass after one untimed pass, "
                   f"{runs_note(list(runs))}"), "",
                  f"The copy: {copy['vector_bytes']:,} bytes of float32 vectors, exported in "
                  + per_run({n: r["copy"]["export_seconds"] for n, r in runs.items()},
                            lambda v: f"{v:.2f}") + " s"
                  + (f"; the process grew by {grown:,} bytes during the export" if grown else "")
                  + (f"; the hnswlib graph is {params['hnswlib']['index_file_bytes']:,} bytes saved, "
                     f"built in {params['hnswlib']['build_seconds']:.1f} s" if params.get("hnswlib") else
                     "; hnswlib was not installed") + ".", ""]
        groups = [("no filter", "unfiltered", None)] + [
            (case["label"], "filtered", key) for key, case in first["filtered"].items()]
        for label, kind, key in groups:
            rows_first = first["unfiltered"] if key is None else first["filtered"][key]["rows"]
            if key is not None:
                lines.append(f"{label}: {first['filtered'][key]['passing_chunks']:,} chunks pass.")
                lines.append("")
            rows = []
            for i, row in enumerate(rows_first):
                this = {n: (r["unfiltered"] if key is None else r["filtered"][key]["rows"])[i]
                        for n, r in runs.items()}
                rows.append([row["method"], row["kind"],
                             per_run({n: r["p50_ms"] for n, r in this.items()}, ms),
                             per_run({n: r["p95_ms"] for n, r in this.items()}, ms),
                             f"{row['mean_overlap_with_exact']:.3f}",
                             per_run({n: r["same_order_as_exact"] for n, r in this.items()},
                                     lambda v, q=n_q: f"{v:,}/{q:,}"),
                             f"{row['short']:,}/{n_q:,}"])
            lines += table(["method", "kind", "p50 ms", "p95 ms", "overlap with exact",
                            "same order as exact", "short"], rows)
            lines.append("")
    return lines if any_data else [*lines, NOT_YET, ""]


def section_cache(base: Path) -> list[str]:
    lines = ["## 9. The index cache below and above the graph (`cache`)", ""]
    any_data = False
    for db in DATABASES:
        runs = {n: load(p) for n, p in runs_of(base, f"run{{n}}/{db}/cache_size.json").items()}
        if not runs:
            continue
        any_data = True
        first = next(iter(runs.values()))
        n_q = first["n_queries"]
        lines += [(f"### {NAMES[db]}: {first['n_chunks']:,} chunks, {n_q:,} claims, "
                   f"k = {first['k']}, {first['passes']} passes per size, configured "
                   f"{first['configured_cache_bytes'] / 1048576:g} MB, restored "
                   f"{first['restored_cache_bytes'] / 1048576:g} MB, {runs_note(list(runs))}"), ""]
        rows = []
        for i, row in enumerate(first["results"]):
            this = {n: r["results"][i] for n, r in runs.items()}
            size = (f"{row['cache_bytes'] / 1048576:g} MB ({row['cache']})"
                    if row["cache"].startswith("as configured") else row["cache"])
            rows.append([size, row["statement"],
                         per_run({n: r["equal_to_exact"] for n, r in this.items()},
                                 lambda v: ", ".join(f"{x:,}" for x in v)),
                         per_run({n: r["article_recall"] for n, r in this.items()},
                                 lambda v: ", ".join(f"{x:,}" for x in v)),
                         per_run({n: r["identical_between_two_passes"] for n, r in this.items()},
                                 lambda v: ", ".join(f"{x:,}" for x in v)),
                         per_run({n: r["p50_ms"] for n, r in this.items()}, ms)])
        lines += table(["cache", "statement", f"equal to exact (of {n_q:,}), per pass",
                        "article recall, per pass", "two passes agree", "p50 ms"], rows)
        lines.append("")
    return lines if any_data else [*lines, NOT_YET, ""]


def spread(values: dict[str, float], digits: int = 0) -> str:
    low, high = values["min"], values["max"]
    return f"{low:,.{digits}f}" if low == high else f"{low:,.{digits}f} to {high:,.{digits}f}"


def section_builds(base: Path) -> list[str]:
    lines = ["## 10. The same vectors built several times (`builds`)", ""]
    found = []
    for db in DATABASES:
        timing = load(base / "build_timing" / db / "build_variance.json")
        if timing:
            found.append(("single timed build", db, {0: timing}))
        runs = {n: load(p) for n, p in runs_of(base, f"run{{n}}/{db}/build_variance.json").items()}
        if runs:
            found.append(("build series", db, runs))
    if not found:
        return [*lines, NOT_YET, ""]
    for label, db, runs in found:
        for n, report in runs.items():
            n_q = report["n_queries"]
            lines += [(f"### {NAMES[db]}, {label}{'' if n == 0 else f' (run {n})'}: "
                       f"{report['n_chunks']:,} chunks, {report['builds']} build(s) per M, "
                       f"{n_q:,} claims, k = {report['k']}; exact ranking finds a gold page for "
                       f"{report['exact_article_recall']:,}"), ""]
            rows = [[r["m"], r["ef_search"], spread(r["article_recall"]),
                     spread(r["equal_to_exact"]), spread(r["overlap_with_exact"], 3),
                     spread(r["identical_between_two_builds"]) if report["builds"] > 1 else "-",
                     spread(r["build_seconds"], 1)] for r in report["results"]]
            lines += table(["M", "ef_search", "article recall", f"equal to exact (of {n_q:,})",
                            "overlap with exact", "two builds agree", "build s"], rows)
            lines.append("")
    return lines


def section_uncommitted(base: Path) -> list[str]:
    lines = ["## 11. Search over a transaction's own uncommitted rows", ""]
    any_data = False
    for db in DATABASES:
        runs = {n: load(p) for n, p in runs_of(base, f"run{{n}}/{db}/uncommitted_recall.json").items()}
        if not runs:
            continue
        any_data = True
        first = next(iter(runs.values()))
        lines += [(f"### {NAMES[db]}: k = {first['k']}, {first['builds']} builds per dataset, "
                   f"{runs_note(list(runs))}; mean overlap with the exact top k, the smallest and "
                   "largest build mean"), ""]

        def cell(report: dict[str, Any], name: str, state: str, ef: int, key: str) -> str:
            cells = [b[state][str(ef)][key] for b in report["datasets"][name]["builds"]]
            return f"{min(cells):.3f} to {max(cells):.3f}"

        rows = []
        for name in first["datasets"]:
            for state in ("uncommitted", "committed"):
                for ef in first["efs"]:
                    rows.append([name, state, ef,
                                 per_run({n: cell(r, name, state, ef, "mean_overlap")
                                          for n, r in runs.items()}),
                                 per_run({n: cell(r, name, state, ef, "p50_ms")
                                          for n, r in runs.items()})])
        lines += table(["data", "state", "ef_search", "mean overlap", "p50 ms"], rows)
        lines += ["", "Datasets: " + "; ".join(f"`{n}`: {d['label']}" for n, d in first["datasets"].items())
                  + ".", ""]
    return lines if any_data else [*lines, NOT_YET, ""]


SECTIONS = (section_eval, section_timing, section_scores, section_breakdown, section_filters,
            section_join_order, section_forms, section_separate, section_cache, section_builds,
            section_uncommitted)


def summary(base: Path) -> str:
    lines = [
        "# Scaled run: every table",
        "",
        (f"Written {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
         "`scripts/scaled_summary.py` from the files in this folder (no database). The 3,000 claims "
         "of `data/scaled/claims_3000.jsonl` on two databases: S0 = the gold pages, S1 = S0 plus "
         "one distractor page per gold article (`docs/DATA.md`). Settings: 240-word chunks, overlap "
         "1, bge-small-en-v1.5 with the title prefix, M = 16, cosine, `mhnsw_ef_search` 100. Every "
         "run on this laptop with the CPU capped at 50% and the GPU at 2,100 MHz, so compare times "
         "only within this folder, not with `results/` at the top level."),
        "",
    ]
    for section in SECTIONS:
        lines += section(base)
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    text = summary(args.dir)
    (args.dir / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(f"wrote {args.dir / 'SUMMARY.md'} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
