#!/usr/bin/env python3
"""Draw the README charts from the JSON files in results/ as self-contained SVG files.

Seven charts, written to docs/img/ (standard library only, no plotting package):

- ``llm_accuracy.svg``: accuracy of the four LLM conditions with Wilson 95% intervals, on three
  classes and on the supported/refuted claims, as dots with interval lines in two panels on one
  scale (``results/llm/llm_results.json``).
- ``llm_f1.svg``: F1 of the supported and refuted labels for the four LLM conditions, a heatmap
  on one sequential colour scale with the F1 and its interval in each cell (same file).

- ``scaled_recall_at_k.svg``: article and evidence recall at k of the scaled run on S0 and S1
  (``results/scaled/eval/s0_full_run1.json`` and ``s1_full_run1.json``), one line each.
- ``recall_at_k.svg``: article and evidence recall at k of the headline run of the demo corpus
  (``results/final_ef_100.json``), one line each.
- ``uncommitted_recall.svg``: overlap with the exact top 10 before and after COMMIT, per
  ``mhnsw_ef_search``, for the chunk embeddings and for random vectors
  (``results/uncommitted_recall.json``), as two panels of grouped bars on one scale.
- ``planner_rows.svg``: chunk rows read (ANALYZE ``r_rows``, median over the claims) by the inline
  statement under each filter, coloured by whether the plan read chunk through the vector index
  (``results/planner_choices.json``).
- ``scale_cost.svg``: SQL time (p50) of the HNSW search, the exact inline statement and the inline
  statement under the History filter, against the number of chunks (``results/scale.json``).

A chart whose JSON file is missing is skipped with a message. Every chart carries a title,
a ``<title>``/``<desc>`` for screen readers and a legend for two series; the README puts the
same numbers in a table next to it. Colours are the first two slots of the dataviz reference
palette (blue, orange), with their dark-mode steps under ``prefers-color-scheme: dark``.

Usage (from the project root): python3 scripts/make_charts.py [--results results] [--out docs/img]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[1]

WIDTH = 720
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
STYLE = """
svg { --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --grid: #e1e0d9;
      --axis: #c3c2b7; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; }
@media (prefers-color-scheme: dark) {
  svg { --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a;
        --axis: #383835; --s1: #3987e5; --s2: #d95926; --s3: #199e70; }
}
.bg { fill: var(--surface); }
text { font-family: FONT; fill: var(--ink2); font-size: 12px; }
.title { fill: var(--ink); font-size: 15px; font-weight: 600; }
.subtitle { fill: var(--ink2); font-size: 12px; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.label { fill: var(--ink); font-size: 12px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.s1 { fill: var(--s1); } .s2 { fill: var(--s2); } .s3 { fill: var(--s3); }
.line1 { stroke: var(--s1); } .line2 { stroke: var(--s2); } .line3 { stroke: var(--s3); }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { stroke: var(--surface); stroke-width: 2; }
""".replace("FONT", FONT)


def _fmt(value, digits=2):
    """Return ``value`` with ``digits`` decimals, without a trailing ``.00`` for integers."""
    if float(value).is_integer() and digits:
        return f"{int(value)}"
    return f"{value:.{digits}f}"


class Svg:
    """A tiny SVG writer: elements are appended as strings; ``render`` wraps them."""

    def __init__(self, height, title, subtitle, description):
        self.height = height
        self.title = title
        self.subtitle = subtitle
        self.description = description
        self.parts = []

    def add(self, element):
        self.parts.append(element)

    def text(self, x, y, value, css="", anchor="start", baseline="auto"):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}" '
            f'dominant-baseline="{baseline}">{escape(str(value))}</text>'
        )

    def line(self, x1, y1, x2, y2, css):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{css}"/>')

    def column(self, x, y_top, width, y_base, css):
        """A column growing up from ``y_base``: 4px rounded top, square at the baseline."""
        height = y_base - y_top
        if height <= 0:
            return
        r = min(4.0, width / 2, height)
        self.add(
            f'<path class="{css}" d="M{x:.1f},{y_base:.1f} V{y_top + r:.1f} '
            f"Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} H{x + width - r:.1f} "
            f'Q{x + width:.1f},{y_top:.1f} {x + width:.1f},{y_top + r:.1f} V{y_base:.1f} Z"/>'
        )

    def bar(self, x_base, y, x_end, height, css):
        """A bar growing right from ``x_base``: 4px rounded end, square at the baseline."""
        length = x_end - x_base
        if length <= 0:
            return
        r = min(4.0, height / 2, length)
        self.add(
            f'<path class="{css}" d="M{x_base:.1f},{y:.1f} H{x_end - r:.1f} '
            f"Q{x_end:.1f},{y:.1f} {x_end:.1f},{y + r:.1f} V{y + height - r:.1f} "
            f'Q{x_end:.1f},{y + height:.1f} {x_end - r:.1f},{y + height:.1f} H{x_base:.1f} Z"/>'
        )

    def legend(self, x, y, entries):
        """Swatches with their names (text in ink, the colour only on the swatch)."""
        for css, name in entries:
            self.add(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="12" height="12" rx="3" class="{css}"/>')
            self.text(x + 18, y + 1, name, "label")
            x += 18 + 7.2 * len(name) + 20

    def render(self):
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {self.height}" '
            f'width="{WIDTH}" height="{self.height}" role="img" aria-labelledby="t d">'
            f"<title id=\"t\">{escape(self.title)}</title><desc id=\"d\">{escape(self.description)}</desc>"
            f"<style>{STYLE}</style>"
            f'<rect class="bg" width="{WIDTH}" height="{self.height}" rx="8"/>'
        )
        titles = (
            f'<text x="24" y="30" class="title">{escape(self.title)}</text>'
            f'<text x="24" y="50" class="subtitle">{escape(self.subtitle)}</text>'
        )
        return head + titles + "".join(self.parts) + "</svg>\n"


def recall_at_k(data):
    """Line chart: article and evidence recall of the headline run against k."""
    per_k = sorted(data["per_k"], key=lambda m: m["k"])
    ks = [m["k"] for m in per_k]
    series = [
        ("line1", "s1", "article recall", [m["article_recall"] for m in per_k],
         [f"{m['article_hits']}/{data['n_claims']}" for m in per_k]),
        ("line2", "s2", "evidence recall", [m["evidence_recall"] for m in per_k],
         [f"{m['evidence_hits']}/{data['n_evidence_claims']}" for m in per_k]),
    ]  # fmt: skip
    svg = Svg(
        360,
        "Recall at k, headline run",
        f"{data['n_claims']} claims; 240-word chunks, M=16, mhnsw_ef_search 100 "
        "(results/final_ef_100.json)",
        "Article recall and evidence recall at k = " + ", ".join(map(str, ks)) + ": "
        + "; ".join(f"{name} {', '.join(labels)}" for _, _, name, _, labels in series),
    )
    left, right, top, bottom = 64, WIDTH - 40, 104, 360 - 48
    svg.legend(24, 76, [("s1", "article recall: a gold page in the top-k chunks"),
                        ("s2", "evidence recall: every gold sentence")])
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - tick * (bottom - top)
        svg.line(left, y, right, y, "axis" if tick == 0 else "grid")
        svg.text(left - 8, y, _fmt(tick), "tick", "end", "middle")
    step = (right - left) / (len(ks) - 1)
    xs = [left + i * step for i in range(len(ks))]
    for x, k in zip(xs, ks):
        svg.text(x, bottom + 18, f"k={k}", "tick", "middle")
    for line_css, dot_css, name, values, labels in series:
        points = [(x, bottom - v * (bottom - top)) for x, v in zip(xs, values)]
        svg.add(f'<polyline class="line {line_css}" points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>')
        for x, y in points:
            svg.add(f'<circle class="dot {dot_css}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
    # the two lines meet at k=20, so end labels would collide: the legend names the series and
    # the README's table next to the chart gives every value
    return svg.render()


def uncommitted_recall(data):
    """Two panels of grouped columns: overlap before and after COMMIT per ef."""
    efs = data["efs"]
    panels = list(data["datasets"].items())
    svg = Svg(
        410,
        "Searching rows the transaction has not committed",
        f"Overlap with the exact top {data['k']}, mean of {data['builds']} index builds "
        "(results/uncommitted_recall.json)",
        "Mean overlap with the exact top 10 before and after COMMIT, per mhnsw_ef_search: "
        + "; ".join(
            f"{name}: " + ", ".join(
                f"ef {ef} {_mean(d, 'uncommitted', ef):.3f} before / {_mean(d, 'committed', ef):.3f} after"
                for ef in efs)
            for name, d in panels),
    )
    svg.legend(24, 76, [("s2", "before COMMIT (rows of the open transaction)"),
                        ("s1", "after COMMIT")])
    top, bottom = 140, 410 - 48
    panel_width = (WIDTH - 64 - 24) / len(panels)
    for p, (name, dataset) in enumerate(panels):
        left = 64 + p * panel_width
        right = left + panel_width - 24
        if p == 0:
            for tick in (0, 0.25, 0.5, 0.75, 1.0):
                y = bottom - tick * (bottom - top)
                svg.text(left - 8, y, _fmt(tick), "tick", "end", "middle")
        for tick in (0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - tick * (bottom - top)
            svg.line(left, y, right, y, "axis" if tick == 0 else "grid")
        label = "Wikipedia chunk embeddings" if name == "chunks" else "random vectors"
        svg.text(left, top - 30, f"{label} ({dataset['builds'][0]['rows']:,} rows)", "label")
        group = (right - left) / len(efs)
        for g, ef in enumerate(efs):
            centre = left + group * (g + 0.5)
            for offset, state, css in ((-13, "uncommitted", "s2"), (13, "committed", "s1")):
                value = _mean(dataset, state, ef)
                x = centre + offset - 12  # 24 px columns 26 px apart: a 2 px surface gap
                svg.column(x, bottom - value * (bottom - top), 24, bottom, css)
                svg.text(x + 12, bottom - value * (bottom - top) - 6, f"{value:.2f}", "tick",
                         "middle")
            svg.text(centre, bottom + 18, f"ef {ef}", "tick", "middle")
    return svg.render()


def _mean(dataset, state, ef):
    builds = dataset["builds"]
    return sum(b[state][str(ef)]["mean_overlap"] for b in builds) / len(builds)


def _case_label(name):
    """``edited_since_2020_12_01`` -> ``edited since 2020-12-01``: words spaced, dates dashed."""
    return re.sub(r"(\d{4}) (\d{2}) (\d{2})", r"\1-\2-\3", name.replace("_", " "))


def planner_rows(data):
    """Horizontal bars: chunk rows read per filter, coloured by the plan's access to chunk."""
    cases = list(data["cases"].items())
    row_height, gap = 24, 10
    top = 104
    height = top + len(cases) * (row_height + gap) + 48
    n_chunks = data["n_chunks"]
    svg = Svg(
        height,
        "Chunk rows read under each filter (inline statement)",
        f"ANALYZE r_rows, median over the claims, k {data['k']}; the table has {n_chunks:,} chunks "
        "(results/planner_choices.json)",
        "Rows read per filter: " + "; ".join(
            f"{name} {case['r_rows_p50']:.0f} rows, "
            f"{'vector index' if case['index_plans'] == case['queries'] else 'full scan in part'}"
            for name, case in cases),
    )
    svg.legend(24, 74, [("s1", "plan reads chunk through the vector index"),
                        ("s2", "plan scans chunk and sorts")])
    left, right = 210, WIDTH - 90
    for tick in range(0, n_chunks + 1, 1000):  # round ticks; the scale ends at the table size
        x = left + tick / n_chunks * (right - left)
        svg.line(x, top - 6, x, height - 40, "axis" if tick == 0 else "grid")
        svg.text(x, height - 24, f"{tick:,}", "tick", "middle")
    for i, (name, case) in enumerate(cases):
        y = top + i * (row_height + gap)
        index_plan = case["index_plans"] * 2 > case["queries"]
        value = case["r_rows_p50"]
        svg.text(left - 10, y + row_height / 2, _case_label(name), "label", "end", "middle")
        svg.bar(left, y, left + value / n_chunks * (right - left), row_height,
                "s1" if index_plan else "s2")
        svg.text(left + value / n_chunks * (right - left) + 8, y + row_height / 2,
                 f"{value:,.0f}", "tick", "start", "middle")
    return svg.render()


def _nice_max(value):
    """The smallest of 1, 2, 2.5 or 5 times a power of ten that is >= ``value`` (axis top)."""
    magnitude = 10 ** len(str(int(value))) / 10 if value >= 1 else 0.1
    for step in (1, 2, 2.5, 5, 10):
        if step * magnitude >= value:
            return step * magnitude
    return 10 * magnitude


def scale_cost(data):
    """Lines: SQL p50 of three statements against the chunk count, one x position per size."""
    sizes = data["sizes"]
    series = [
        ("line1", "s1", "HNSW search (strategy none)", [s["hnsw"]["sql_ms_p50"] for s in sizes]),
        ("line2", "s2", "exact ranking (inline, no filter)", [s["exact"]["sql_ms_p50"] for s in sizes]),
        ("line3", "s3", "inline under the History filter",
         [s["history_inline"]["sql_ms_p50"] for s in sizes]),
    ]
    svg = Svg(
        392,
        "Query time as the chunk table grows",
        f"SQL p50 at k={data['k']}, ef_search {data['ef_search']}, over the claims; distractor pages "
        "from the full FEVEROUS Wikipedia (results/scale.json)",
        "SQL p50 in ms by chunk count: " + "; ".join(
            f"{name}: " + ", ".join(f"{s['chunks']:,} chunks {v:.2f} ms" for s, v in zip(sizes, values))
            for _, _, name, values in series),
    )
    svg.legend(24, 76, [("s1", "HNSW search"), ("s2", "exact ranking (joined statement)"),
                        ("s3", "joined, History filter")])
    left, right, top, bottom = 64, WIDTH - 40, 104, 392 - 60
    # one linear scale per axis, with round ticks: x is the chunk count (not the size's index,
    # which would bend the lines), y the milliseconds
    y_step = _nice_max(max(v for *_, values in series for v in values) / 5)
    y_top = y_step * 5
    for i in range(6):
        tick = y_step * i
        y = bottom - tick / y_top * (bottom - top)
        svg.line(left, y, right, y, "axis" if i == 0 else "grid")
        svg.text(left - 8, y, f"{tick:g} ms" if i else "0", "tick", "end", "middle")
    x_step = _nice_max(max(s["chunks"] for s in sizes) / 4)
    x_top = x_step * 4
    for i in range(5):
        x = left + i / 4 * (right - left)
        svg.text(x, bottom + 18, f"{x_step * i:,.0f}" if i else "0", "tick", "middle")
    svg.text(right, bottom + 34, "chunks in the table", "tick", "end")
    xs = [left + s["chunks"] / x_top * (right - left) for s in sizes]
    for line_css, dot_css, _name, values in series:
        points = [(x, bottom - v / y_top * (bottom - top)) for x, v in zip(xs, values)]
        svg.add(f'<polyline class="line {line_css}" points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>')
        for x, y in points:
            svg.add(f'<circle class="dot {dot_css}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
    return svg.render()


def scaled_recall_at_k(data):
    """Line chart: article and evidence recall against k on S0 (dashed) and S1 (solid)."""
    s0, s1 = data["s0_full_run1"], data["s1_full_run1"]
    ks = [m["k"] for m in sorted(s1["per_k"], key=lambda m: m["k"])]

    def values(run, key):
        return [m[key] for m in sorted(run["per_k"], key=lambda m: m["k"])]

    series = [
        ("line1", "s1", "", "S1 article recall", values(s1, "article_recall")),
        ("line2", "s2", "", "S1 evidence recall", values(s1, "evidence_recall")),
        ("line1", "s1", ' stroke-dasharray="6 4"', "S0 article recall", values(s0, "article_recall")),
        ("line2", "s2", ' stroke-dasharray="6 4"', "S0 evidence recall", values(s0, "evidence_recall")),
    ]
    svg = Svg(
        360,
        "Recall at k, the scaled run",
        f"{s1['n_claims']:,} claims; S0 {s0['parameters']['n_chunks']:,} chunks (dashed), "
        f"S1 {s1['parameters']['n_chunks']:,} chunks (solid) (results/scaled/eval/)",
        "Article and evidence recall at k = " + ", ".join(map(str, ks)) + ": "
        + "; ".join(f"{name} " + ", ".join(f"{v:.3f}" for v in vals)
                    for _, _, _, name, vals in series),
    )
    left, right, top, bottom = 64, WIDTH - 40, 104, 360 - 48
    svg.legend(24, 76, [("s1", "article recall: a gold page in the top-k chunks"),
                        ("s2", "evidence recall: a complete gold set")])
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - tick * (bottom - top)
        svg.line(left, y, right, y, "axis" if tick == 0 else "grid")
        svg.text(left - 8, y, _fmt(tick), "tick", "end", "middle")
    step = (right - left) / (len(ks) - 1)
    xs = [left + i * step for i in range(len(ks))]
    for x, k in zip(xs, ks):
        svg.text(x, bottom + 18, f"k={k}", "tick", "middle")
    for line_css, dot_css, dash, _, vals in series:
        points = [(x, bottom - v * (bottom - top)) for x, v in zip(xs, vals)]
        svg.add(f'<polyline class="line {line_css}"{dash} points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>')
        for x, y in points:
            svg.add(f'<circle class="dot {dot_css}" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
    return svg.render()


#: The LLM conditions in the order of every LLM table and chart.
LLM_CONDITIONS = (("no_retrieval", "no passages"), ("web", "web search"), ("retrieved", "RAG"),
                  ("gold", "gold passages"))


def llm_accuracy(data):
    """Dot-and-interval chart: accuracy of the four LLM conditions with Wilson 95% intervals,
    three classes (left panel) and supported/refuted only (right panel), one x scale."""
    panels = [
        ("three classes", data["three_class"]["conditions"], data["three_class"]["conditions"]
         ["retrieved"]["accuracy"]["n"]),
        ("supported/refuted only", data["supported_refuted"]["conditions"],
         data["supported_refuted"]["n"]),
    ]
    height = 300
    svg = Svg(
        height,
        "LLM fact-checker accuracy, Claude Haiku 5.5",
        "Dot: accuracy; line: 95% Wilson interval (results/llm/llm_results.json)",
        "Accuracy with 95% intervals. " + " ".join(
            f"{title} ({n:,} claims): " + "; ".join(
                f"{name} {conds[c]['accuracy']['pct']}% ({conds[c]['accuracy']['wilson_pct'][0]}"
                f" to {conds[c]['accuracy']['wilson_pct'][1]})" for c, name in LLM_CONDITIONS) + "."
            for title, conds, n in panels),
    )
    lo, hi = 30, 90
    label_w, gap, top, row_h = 100, 40, 100, 40
    panel_w = (WIDTH - 24 - label_w - gap - 24) / 2
    for p, (title, conds, n) in enumerate(panels):
        x0 = 24 + label_w + p * (panel_w + gap)

        def sx(value, x0=x0):
            return x0 + (value - lo) / (hi - lo) * panel_w

        bottom = top + row_h * len(LLM_CONDITIONS)
        svg.text(x0, top - 20, f"{title} ({n:,} claims)", "label")
        for tick in range(lo, hi + 1, 10):
            svg.line(sx(tick), top - 6, sx(tick), bottom, "axis" if tick == lo else "grid")
            svg.text(sx(tick), bottom + 18, f"{tick}%", "tick", "middle")
        for i, (cond, name) in enumerate(LLM_CONDITIONS):
            y = top + row_h * i + row_h / 2
            acc = conds[cond]["accuracy"]
            a, (w_lo, w_hi) = 100 * acc["acc"], (100 * v for v in acc["wilson"])
            if p == 0:
                svg.text(24 + label_w - 12, y, name, "label", "end", "middle")
            svg.add(f'<path d="M{sx(w_lo):.1f},{y - 6:.1f} V{y + 6:.1f} M{sx(w_lo):.1f},{y:.1f} '
                    f'H{sx(w_hi):.1f} M{sx(w_hi):.1f},{y - 6:.1f} V{y + 6:.1f}" '
                    f'class="line line1" style="stroke-width: 2.5"/>')
            svg.add(f'<circle class="dot s1" cx="{sx(a):.1f}" cy="{y:.1f}" r="4">'
                    f"<title>{escape(name)}: {acc['pct']}% ({acc['wilson_pct'][0]} to "
                    f"{acc['wilson_pct'][1]}), {acc['right']:,} of {acc['n']:,}</title></circle>")
            svg.text(sx(w_hi) + 8, y, f"{acc['pct']}%", "label", "start", "middle")
    return svg.render()


#: Sequential blue steps (light to dark) of the F1 heatmap, and their dark-mode steps (near the
#: dark surface for low values); text on a cell switches to white where the fill is dark.
HEAT_LIGHT = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#184f95")
HEAT_DARK = ("#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4")
HEAT_WHITE_TEXT_LIGHT = frozenset({3, 4, 5, 6})
HEAT_WHITE_TEXT_DARK = frozenset({0, 1, 2, 3})


def _heat_style():
    rules = []
    for i, (light, dark) in enumerate(zip(HEAT_LIGHT, HEAT_DARK)):
        ink_l = "#ffffff" if i in HEAT_WHITE_TEXT_LIGHT else "#0b0b0b"
        ink_d = "#ffffff" if i in HEAT_WHITE_TEXT_DARK else "#0b0b0b"
        rules.append((f".h{i} {{ fill: {light}; }} .ht{i} {{ fill: {ink_l}; }}",
                      f".h{i} {{ fill: {dark}; }} .ht{i} {{ fill: {ink_d}; }}"))
    return ("<style>" + " ".join(r[0] for r in rules)
            + " @media (prefers-color-scheme: dark) { " + " ".join(r[1] for r in rules)
            + " }</style>")


def llm_f1(data):
    """Heatmap: F1 of SUPPORTS and REFUTES on the supported/refuted claims for the four LLM
    conditions, one colour scale, the F1 and its bootstrap 95% interval printed in each cell."""
    conds = data["supported_refuted"]["conditions"]
    labels = (("SUPPORTS", "supported"), ("REFUTES", "refuted"))
    values = [conds[c]["per_label"][label]["f1"] * 100 for c, _ in LLM_CONDITIONS
              for label, _ in labels]
    lo, hi = math.floor(min(values) / 5) * 5, math.ceil(max(values) / 5) * 5
    boot = data["conventions"]["bootstrap_resamples"]
    height = 340
    svg = Svg(
        height,
        "F1 per label, supported and refuted claims",
        f"{data['supported_refuted']['n']:,} claims; F1 and paired-bootstrap 95% interval "
        f"({boot:,} resamples) in each cell (results/llm/llm_results.json)",
        "F1 per label. " + "; ".join(
            f"{name}: " + ", ".join(
                f"{short} {conds[c]['per_label'][label]['f1_pct']} "
                f"({conds[c]['per_label'][label]['f1_bootstrap_pct'][0]} to "
                f"{conds[c]['per_label'][label]['f1_bootstrap_pct'][1]})"
                for label, short in labels)
            for c, name in LLM_CONDITIONS) + ".",
    )
    svg.add(_heat_style())
    label_w, top, cell_w, cell_h, gap = 132, 96, 190, 40, 2
    x0 = 24 + label_w
    for j, (_, short) in enumerate(labels):
        svg.text(x0 + j * (cell_w + gap) + cell_w / 2, top - 12, f"F1, {short} label", "label",
                 "middle")
    steps = len(HEAT_LIGHT)
    for i, (cond, name) in enumerate(LLM_CONDITIONS):
        y = top + i * (cell_h + gap)
        svg.text(x0 - 12, y + cell_h / 2, name, "label", "end", "middle")
        for j, (label, short) in enumerate(labels):
            e = conds[cond]["per_label"][label]
            step = min(steps - 1, int((e["f1"] * 100 - lo) / (hi - lo) * steps))
            x = x0 + j * (cell_w + gap)
            text = f"{e['f1_pct']} ({e['f1_bootstrap_pct'][0]}–{e['f1_bootstrap_pct'][1]})"
            svg.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w}" height="{cell_h}" rx="4" '
                    f'class="h{step}"><title>{escape(name)}, {short}: F1 {escape(text)}; '
                    f"precision {e['precision_pct']}, recall {e['recall_pct']}</title></rect>")
            svg.text(x + cell_w / 2, y + cell_h / 2, text, f"label ht{step}", "middle", "middle")
    # the colour scale
    sy = top + len(LLM_CONDITIONS) * (cell_h + gap) + 18
    sw = 2 * cell_w + gap
    for s in range(steps):
        svg.add(f'<rect x="{x0 + s * sw / steps:.1f}" y="{sy:.1f}" width="{sw / steps - 2:.1f}" '
                f'height="10" rx="2" class="h{s}"/>')
    svg.text(x0, sy + 26, f"F1 {lo}", "tick", "start")
    svg.text(x0 + sw, sy + 26, f"{hi}", "tick", "end")
    return svg.render()


CHARTS = (
    ("llm/llm_results.json", "llm_accuracy.svg", llm_accuracy),
    ("llm/llm_results.json", "llm_f1.svg", llm_f1),
    (("scaled/eval/s0_full_run1.json", "scaled/eval/s1_full_run1.json"), "scaled_recall_at_k.svg",
     scaled_recall_at_k),
    ("final_ef_100.json", "recall_at_k.svg", recall_at_k),
    ("uncommitted_recall.json", "uncommitted_recall.svg", uncommitted_recall),
    ("planner_choices.json", "planner_rows.svg", planner_rows),
    ("scale.json", "scale_cost.svg", scale_cost),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw the README charts from results/ as SVG.")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "img")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for source, target, draw in CHARTS:
        paths = [args.results / name for name in (source if isinstance(source, tuple) else (source,))]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            print(f"skipped {target}: {missing[0]} not found", file=sys.stderr)
            continue
        loaded = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in paths}
        (args.out / target).write_text(draw(loaded if len(paths) > 1 else loaded[paths[0].stem]),
                                       encoding="utf-8")
        print(f"wrote {args.out / target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
