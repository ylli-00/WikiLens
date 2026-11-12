"""A measured summary of one run folder (Markdown), computed only from its rows and manifest.

Per model and condition: rows by status, parse success, latency, and the tokens per call as the
CLI reports them (plus the runner's computed prompt sum). Over the run: the first and last
window readings, the number of rows that carried a rate-limit event, every error and stop
signal, side models, and the retrieved chunks of the first ``n_examples`` claims. Nothing is
estimated here.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wikilense.llm.runner import load_rows


def _stats(values: list[float]) -> str:
    """Return ``mean / median / min / max`` of the values, or ``-`` when there are none."""
    if not values:
        return "-"
    return (f"{statistics.mean(values):,.1f} / {statistics.median(values):,.1f} / "
            f"{min(values):,.1f} / {max(values):,.1f}")


def summarize(run_dir: Path, n_examples: int = 2, cache_root: Path | None = None) -> str:
    """Return the Markdown summary of a run folder."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = sorted(load_rows(run_dir / "results.jsonl"), key=lambda r: r.get("ts") or "")
    out: list[str] = [f"# Run {manifest.get('run_id')}", ""]
    out.append(f"- models: {manifest.get('models')}; effort {manifest.get('effort')}; prompt "
               f"{manifest['prompt']['prompt_id']} ({manifest['prompt']['prompt_hash'][:12]})")
    out.append(f"- CLI: {manifest.get('cli', {}).get('version')}; claims file sha256 "
               f"{manifest['claims_file']['sha256'][:12]}; retrieval {manifest.get('retrieval')}")
    out.append(f"- rows: {len(rows)}; statuses {dict(Counter(r['status'] for r in rows))}")
    out.append("")
    out.append("| model | condition | rows | ok | parse failed | refusal | error | latency s "
               "(mean/median/min/max) | duration_ms (mean/median/min/max) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_requested"], row["condition"])].append(row)
    for (model, cond), group in sorted(groups.items()):
        st = Counter(r["status"] for r in group)
        lat = [r["latency_s"] for r in group if r.get("latency_s") is not None]
        dur = [float(r["cli"]["duration_ms"]) for r in group
               if r.get("cli") and r["cli"].get("duration_ms") is not None]
        out.append(f"| {model} | {cond} | {len(group)} | {st['ok']} | {st['parse_failed']} | "
                   f"{st['refusal']} | {st['error'] + st['window'] + st['fatal']} | "
                   f"{_stats(lat)} | {_stats(dur)} |")
    out += ["", ("Tokens per call (mean / median / min / max), as the CLI's modelUsage reports "
                 "them for the requested model; `prompt` is the runner's computed sum "
                 "inputTokens + cacheReadInputTokens + cacheCreationInputTokens:"), ""]
    out.append("| model | condition | inputTokens | cacheRead | cacheCreation | prompt (computed) "
               "| outputTokens | thinkingTokens | num_turns |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for (model, cond), group in sorted(groups.items()):
        cols: dict[str, list[float]] = defaultdict(list)
        for r in group:
            usage = ((r.get("cli") or {}).get("model_usage") or {})
            served = r.get("model_served")
            entry = usage.get(served) if isinstance(served, str) else None
            if not entry:
                continue
            for name in ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens",
                         "outputTokens", "thinkingTokens"):
                if entry.get(name) is not None:
                    cols[name].append(float(entry[name]))
            cols["prompt"].append(float(r["computed_usage"].get("prompt_tokens", 0)))
            if r["cli"].get("num_turns") is not None:
                cols["turns"].append(float(r["cli"]["num_turns"]))
        out.append(f"| {model} | {cond} | {_stats(cols['inputTokens'])} | "
                   f"{_stats(cols['cacheReadInputTokens'])} | "
                   f"{_stats(cols['cacheCreationInputTokens'])} | {_stats(cols['prompt'])} | "
                   f"{_stats(cols['outputTokens'])} | {_stats(cols['thinkingTokens'])} | "
                   f"{_stats(cols['turns'])} |")
    readings = [(r["ts"], r["window_utilization"]) for r in rows if r.get("window_utilization")]
    with_event = sum(1 for r in rows if (r.get("cli") or {}).get("last_rate_limit_event"))
    out += ["", f"- rows with a rate_limit_event: {with_event} of {len(rows)}"]
    if readings:
        out.append(f"- first window reading {readings[0][0]}: {readings[0][1]}")
        out.append(f"- last window reading {readings[-1][0]}: {readings[-1][1]}")
        statuses = Counter(
            ((r["cli"]["last_rate_limit_event"] or {}).get("rate_limit_info") or {}).get("status")
            for r in rows if (r.get("cli") or {}).get("last_rate_limit_event"))
        out.append(f"- rate_limit_info.status values: {dict(statuses)}")
    else:
        out.append("- no window reading in any row")
    side = Counter(name for r in rows for name in (r.get("side_models") or {}))
    out.append(f"- side models: {dict(side) or 'none'}; side-model tokens (computed): "
               f"{sum(r.get('computed_usage', {}).get('side_model_tokens', 0) for r in rows)}")
    out.append(f"- served models: {dict(Counter(str(r.get('model_served')) for r in rows))}")
    out.append(f"- 5-minute cache writes (computed): "
               f"{sum(r.get('computed_usage', {}).get('cache_write_5m', 0) for r in rows)}")
    out.append(f"- attempts > 1: {sum(1 for r in rows if (r.get('attempts') or 0) > 1)}; "
               f"num_turns > 2: "
               f"{sum(1 for r in rows if ((r.get('cli') or {}).get('num_turns') or 0) > 2)}")
    problems = [r for r in rows if r["status"] != "ok" or r.get("stop_signal")]
    out += ["", f"Rows that are not ok, or carry a stop signal: {len(problems)}"]
    for r in problems:
        out.append(f"- claim {r['claim_id']} {r['condition']} {r['model_requested']}: "
                   f"{r['status']} ({r.get('error_kind')}) {str(r.get('error'))[:200]} "
                   f"stop={r.get('stop_signal')}")
    for inv in manifest.get("invocations", []):
        out.append(f"- invocation {inv.get('started_at')} -> {inv.get('ended_at')}: exit "
                   f"{inv.get('exit_code')}, stop {inv.get('stop_reason')}, counts "
                   f"{inv.get('counts')}")
    examples = [r for r in rows if r["condition"] == "retrieved" and r.get("retrieved")]
    seen: set[int] = set()
    out += ["", f"Retrieved chunks (first {n_examples} claims):"]
    for r in examples:
        if r["claim_id"] in seen or len(seen) >= n_examples:
            continue
        seen.add(r["claim_id"])
        out.append(f"\n- claim {r['claim_id']} (gold label {r['gold_label']}):")
        texts: dict[int, str] = {}
        if cache_root is not None:
            retrieval = manifest.get("retrieval", {})
            path = Path(retrieval.get("cache_dir", "")) / f"{r['claim_id']}.json"
            if path.is_file():
                entry = json.loads(path.read_text(encoding="utf-8"))
                texts = {h["chunk_id"]: h["text"] for h in entry["hits"]}
        for hit in r["retrieved"]:
            text = texts.get(hit["chunk_id"], "")
            out.append(f"  {hit['rank']}. chunk {hit['chunk_id']} (page {hit['page_id']}) "
                       f"{hit['title']} > {hit['section_path'] or '(lead)'}, distance "
                       f"{hit['distance']:.4f}" + (f": {text[:160]}..." if text else ""))
    return "\n".join(out) + "\n"
