#!/usr/bin/env python3
"""Audit the web rows of an LLM run folder: which hosts the model opened, and which rows touched
the FEVEROUS answer key.

Reads ``<run_dir>/results.jsonl``, the deny list recorded in ``<run_dir>/manifest.json`` and each
row's raw stream (the tool results the model read). Lists the hosts WebFetch opened and the hosts
WebSearch returned, with counts, and flags every row whose queries, URLs or tool results mention
fever.ai, "feverous", zenodo, huggingface or hf.co, raw.githubusercontent.com or a GitHub path
ending in .jsonl/.json/.csv, a host on the deny list, or "Verdict" next to one of the three
labels (``wikilense.llm.web.audit_row``). Writes ``<run_dir>/WEB_AUDIT.md`` and
``<run_dir>/web_audit.json``. Flagged rows stay in results.jsonl: the owner decides.

Only the newest row of each item key is audited (a re-asked item's earlier error rows are not).

Usage: .venv/bin/python scripts/llm_web_audit.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from wikilense.llm.runner import load_rows
from wikilense.llm.web import DenyList, audit_rows, tool_result_text


def newest_per_key(rows: list[dict]) -> list[dict]:
    """Return the last row of each key, in file order of those last rows."""
    last: dict[str, dict] = {}
    for row in rows:
        last.pop(row["key"], None)
        last[row["key"]] = row
    return list(last.values())


def audit_run(run_dir: Path) -> dict:
    """Audit one run folder; write WEB_AUDIT.md and web_audit.json; return the summary."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    web = manifest.get("web")
    if not web:
        raise SystemExit(f"{run_dir}: the manifest has no web section (not a web run)")
    deny = DenyList(tuple(web["deny_hosts"]), web["deny_hosts_sha256"], web["deny_hosts_file"])
    rows = newest_per_key(load_rows(run_dir / "results.jsonl"))
    raw_texts = {}
    for row in rows:
        name = row.get("raw_stream_file")
        if name and (run_dir / name).is_file():
            raw_texts[name] = tool_result_text((run_dir / name).read_text(encoding="utf-8"))
    result = audit_rows(rows, deny, raw_texts)
    web_rows = [r for r in rows if r.get("condition") == "web"]
    summary = {
        "run_dir": str(run_dir), "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "deny_hosts_sha256": deny.sha256, "web_rows": result.n_rows,
        "rows_without_raw_stream": sum(1 for r in web_rows
                                       if r.get("raw_stream_file") not in raw_texts),
        "web_search_calls": sum(r.get("n_web_search") or 0 for r in web_rows),
        "web_fetch_calls": sum(r.get("n_web_fetch") or 0 for r in web_rows),
        "denied_calls": sum(r.get("n_denied") or 0 for r in web_rows),
        "fetched_hosts": dict(result.hosts.most_common()),
        "search_result_hosts": dict(result.search_hosts.most_common()),
        "flagged": result.flagged,
    }
    (run_dir / "web_audit.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    (run_dir / "WEB_AUDIT.md").write_text(report(summary), encoding="utf-8")
    return summary


def report(s: dict) -> str:
    """Return the Markdown report of one audit."""
    lines = [f"# Web audit of `{Path(s['run_dir']).name}`", "",
             (f"Written by `scripts/llm_web_audit.py` at {s['audited_at']}. Deny list sha256 "
              f"`{s['deny_hosts_sha256'][:16]}…`."), "",
             (f"- web rows audited (newest row per item): {s['web_rows']}"
              f" ({s['rows_without_raw_stream']} without a raw stream)"),
             (f"- WebSearch calls: {s['web_search_calls']}; WebFetch calls: "
              f"{s['web_fetch_calls']}; denied: {s['denied_calls']}"),
             f"- **flagged rows: {len(s['flagged'])}** (kept in results.jsonl; the owner decides)",
             "", "## Hosts WebFetch opened", ""]
    lines += ["| host | fetches |", "|---|---|"] if s["fetched_hosts"] else ["(none)"]
    lines += [f"| {h} | {n} |" for h, n in s["fetched_hosts"].items()]
    lines += ["", "## Hosts in WebSearch results (top 30)", ""]
    top = list(s["search_result_hosts"].items())[:30]
    lines += ["| host | results |", "|---|---|"] if top else ["(none)"]
    lines += [f"| {h} | {n} |" for h, n in top]
    lines += ["", "## Flagged rows", ""]
    if not s["flagged"]:
        lines.append("(none)")
    for f in s["flagged"]:
        lines.append(f"- claim {f['claim_id']} ({f['status']}, label {f['label']}, gold "
                     f"{f['gold_label']}):")
        lines += [f"  - {reason}" for reason in f["reasons"][:12]]
        if len(f["reasons"]) > 12:
            lines.append(f"  - ... {len(f['reasons']) - 12} more")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    summary = audit_run(args.run_dir.resolve())
    print(f"{summary['web_rows']} web rows, {summary['web_fetch_calls']} fetches, "
          f"{summary['denied_calls']} denied, {len(summary['flagged'])} flagged; wrote "
          f"{args.run_dir / 'WEB_AUDIT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
