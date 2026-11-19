"""The run: one folder per run, a manifest, one row per answer, resume, workers and stops.

``<runs_root>/<run_id>/`` holds ``manifest.json``, ``results.jsonl`` (one row per (claim,
condition, model) answer, appended and fsynced) and ``raw/`` (each call's full stream-json
output). Retrieval results are cached in ``<runs_root>/retrieval_cache/`` (see
:mod:`wikilense.llm.retrieval`).

**Resume.** The key of an item is the SHA-256 of (claim id, condition, full model id, effort,
prompt hash). A key is done once any row with that key is final: a parsed label, a parse
failure after the retries, or a refusal. Keys whose last row ended in an error (timeout, CLI
error, spent usage window, fatal stop) are asked again. Rows are indexed by key, not by claim
id, so a later non-final row never hides an earlier final one. A run folder
whose manifest differs in backend, models, effort, prompt hash, k, strategy, ef_search,
embedding model, database, claims-file hash or backend version (CLI or SDK) is refused.

**Backends.** ``--backend subscription`` (Claude Code CLI on the subscription) or ``--backend api``
(Anthropic Messages API with the key from the repo's ``.env``; needs ``--max-usd``). Everything
else is identical, so a job moves between routes by changing only that option; each row and the
manifest record the backend. A ``<<<CACHE_BREAK>>>`` line in a user template marks a prefix the
API backend caches; ``--group-by <claim field>`` sends the items sharing that field (same model
and condition) to one worker, back to back.

**Conditions** come from the prompt set (its ``prompt.json``); a run may ask only for those.
``web`` (prompt ``fact_check_web_v1``) runs on the subscription backend only, needs no database
and no retriever, and records its tools, deny list and turn cap in the manifest's ``web`` section
(:mod:`wikilense.llm.web`); a folder made with another deny list or turn cap is not resumed.

**Order.** Items run claim by claim (all conditions and models of a claim, then the next
claim), so a stop leaves every condition and model on the same finished claims.

**Stops.** The whole run stops, finishing only the calls in flight, on a spent window (exit 3),
a five-hour or weekly utilization reading at or above its threshold (exit 3, both 0.80 by
default; separate thresholds), a fatal result (exit 4), five non-final results
in a row (exit 5), a sign of paid extra usage (exit 6), the API
backend's money stop: ``--max-usd`` reached, or the account's spend limit or credit (exit 7), or
SIGINT/SIGTERM/SIGHUP (exit 130). The runner never waits out a window by itself.

Exit codes: 0 every planned item is final; 1 some are not (re-run the same command); 2 bad
configuration or the item guard; 3 window; 4 fatal; 5 circuit breaker; 6 paid-usage signal;
7 money stop (API); 130 signal.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from wikilense.llm.backend import FINAL_STATUSES, Backend, CallRequest, CallResult
from wikilense.llm.claims import (
    ClaimsError,
    file_sha256,
    load_claims,
    render_gold,
    render_retrieved,
)
from wikilense.llm.guard import GuardError, check_item_budget, items_per_model
from wikilense.llm.models import resolve_models
from wikilense.llm.prompts import (
    LABELS,
    WEB_CONDITION,
    PromptSet,
    canonical_json,
    sha256_text,
)

log = logging.getLogger("wikilense.llm")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = 1
ROW_VERSION = 1
BREAKER_LIMIT = 5
#: Manifest fields that must be equal to resume into a run folder (dotted paths).
RESUME_FIELDS: tuple[str, ...] = (
    "backend", "models", "effort", "prompt.prompt_id", "prompt.prompt_hash", "retrieval.database",
    "retrieval.k", "retrieval.strategy", "retrieval.ef_search", "retrieval.embedding_model",
    "claims_file.sha256", "backend_info.version", "web.deny_hosts_sha256", "web.max_turns",
)
#: Conditions that need no database: a run made only of these gets :class:`NoRetrieval`.
NO_DATABASE_CONDITIONS = frozenset({WEB_CONDITION})
CACHE_BREAK = "<<<CACHE_BREAK>>>"
WEEKLY_PREFIX = "seven_day"

EXIT_OK, EXIT_INCOMPLETE, EXIT_CONFIG, EXIT_WINDOW = 0, 1, 2, 3
EXIT_FATAL, EXIT_BREAKER, EXIT_OVERAGE, EXIT_MONEY, EXIT_SIGNAL = 4, 5, 6, 7, 130


class RunError(RuntimeError):
    """The run cannot start as configured (exit 2)."""


class NoRetrieval:
    """The retriever of a run without the ``retrieved`` condition and without a database
    (``web`` only): it describes itself as nothing and refuses to search."""

    def describe(self) -> dict[str, Any]:
        return {"database": None, "k": None, "strategy": None, "ef_search": None,
                "embedding_model": None, "device": None, "note": "no retrieval in this run"}

    def count_chunks(self) -> None:
        raise RunError("this run has no database")

    def retrieve_many(self, claims: Any) -> dict[int, list[dict[str, Any]]]:
        raise RunError("this run has no database")

    def _read_cache(self, claim_id: int, claim_text: str) -> None:
        return None


def now_iso() -> str:
    """Return the local time with offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def item_key(claim_id: int, condition: str, model_id: str, effort: str, prompt_hash: str) -> str:
    """Return the resume key: SHA-256 of (claim, condition, full model id, effort, prompt hash)."""
    return sha256_text(canonical_json([int(claim_id), condition, model_id, effort, prompt_hash]))


def _get(obj: dict[str, Any], dotted: str) -> Any:
    """Return ``obj`` at a dotted path, None when a step is missing."""
    for part in dotted.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj


# --------------------------------------------------------------------------------------
# Rows on disk
# --------------------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict[str, Any]]:
    """Return the rows of a results file; a torn line (crash mid-write) is skipped."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s line %d is not JSON (torn write?): ignored", path, n)
                continue
            if isinstance(row, dict) and "key" in row:
                rows.append(row)
    return rows


def final_keys(rows: list[dict[str, Any]]) -> set[str]:
    """Return the keys that have at least one final row."""
    return {row["key"] for row in rows if row.get("status") in FINAL_STATUSES}


def repair_torn_tail(path: Path) -> None:
    """End the file with a newline if a crash left a torn last line."""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        if last != b"\n":
            with path.open("ab") as fh:
                fh.write(b"\n")
                fh.flush()
                os.fsync(fh.fileno())


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    """One (claim, condition, model) to answer."""

    claim: dict[str, Any]
    condition: str
    model_option: str
    model_id: str
    key: str

    @property
    def claim_id(self) -> int:
        """Return the claim's FEVEROUS id."""
        return int(self.claim["id"])


def plan_items(claims: list[dict[str, Any]], conditions: list[str], models: dict[str, str],
               effort: str, prompt_hash: str) -> list[Item]:
    """Return the items claim by claim, then condition, then model (the run order)."""
    return [
        Item(claim, condition, option, model_id,
             item_key(claim["id"], condition, model_id, effort, prompt_hash))
        for claim in claims for condition in conditions for option, model_id in models.items()
    ]


@dataclass
class RunConfig:
    """Everything a run is configured with (built by :func:`main` or a test)."""

    run_dir: Path
    claims_path: Path
    prompt: PromptSet
    models: dict[str, str]
    effort: str
    conditions: list[str]
    limit_claims: int | None = None
    workers: int = 5
    full: bool = False
    dry_run: bool = False
    stop_five_hour: float = 0.80
    stop_weekly: float = 0.80
    min_free_mb: int = 2500
    claims_sha256: str | None = None
    allow_cli_change: bool = False
    group_by: str | None = None
    argv: list[str] = field(default_factory=list)


@dataclass
class RunState:
    """State shared by the worker threads; every write goes through ``lock``."""

    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    write_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_code: int = EXIT_OK
    stop_reason: str = ""
    consecutive_errors: int = 0
    counts: Counter = field(default_factory=Counter)
    first_window: dict[str, Any] | None = None
    last_window: dict[str, Any] | None = None
    rate_limit_events: int = 0

    def request_stop(self, code: int, reason: str) -> None:
        """Stop the run (first reason wins); calls in flight finish."""
        if not self.stop.is_set():
            self.stop_code, self.stop_reason = code, reason
            self.stop.set()


def mem_available_mb() -> int | None:
    """Return MemAvailable from /proc/meminfo in MB, None where it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def git_info() -> dict[str, Any]:
    """Return the commit, the branch and the uncommitted files of the LLM runner's own paths."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                                  text=True, timeout=30, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    dirty = git("status", "--porcelain", "--", "wikilense/llm", "scripts", "tests")
    return {"commit": git("rev-parse", "HEAD") or None,
            "branch": git("branch", "--show-current") or None,
            "uncommitted_llm_paths": [line[3:] for line in dirty.splitlines()
                                      if "llm" in line]}


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a temporary file and a rename, fsynced."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def check_resume(old: dict[str, Any], new: dict[str, Any], allow_cli_change: bool) -> None:
    """Raise :class:`RunError` when a run folder's manifest differs in a resume field."""
    fields = [f for f in RESUME_FIELDS if not (allow_cli_change and f == "backend_info.version")]
    diffs = [f"{f}: {_get(old, f)!r} != {_get(new, f)!r}" for f in fields
             if _get(old, f) != _get(new, f)]
    if diffs:
        raise RunError("refusing to resume into a run folder made with other settings: "
                       + "; ".join(diffs) + ". Start a new run folder.")


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------
class Runner:
    """Runs one invocation of a configured run with a backend and a retriever."""

    def __init__(self, config: RunConfig, backend: Backend, retriever: Any) -> None:
        self.cfg = config
        self.backend = backend
        self.retriever = retriever
        self.results_path = config.run_dir / "results.jsonl"
        self.manifest_path = config.run_dir / "manifest.json"
        self.raw_dir = config.run_dir / "raw"
        self.state = RunState()
        self.hits: dict[int, list[dict[str, Any]]] = {}

    # ----------------------------------------------------------------------------------
    def _claims(self) -> tuple[list[dict[str, Any]], str, int]:
        """Return (the selected claims, the file's sha256, claims in the file); check the hash."""
        if not self.cfg.claims_path.is_file():
            raise RunError(f"no claims file {self.cfg.claims_path}")
        sha = file_sha256(self.cfg.claims_path)
        if self.cfg.claims_sha256 and sha != self.cfg.claims_sha256.lower():
            raise RunError(f"{self.cfg.claims_path} has sha256 {sha}, not the expected "
                           f"{self.cfg.claims_sha256}")
        if self.cfg.claims_path.name == "claims_3000.jsonl" and not self.cfg.claims_sha256:
            raise RunError("claims_3000.jsonl needs --claims-sha256, the hash of the final file "
                           "(the file may still be being written)")
        try:
            claims = load_claims(self.cfg.claims_path)
        except ClaimsError as exc:
            raise RunError(str(exc)) from exc
        n_in_file = len(claims)
        if self.cfg.limit_claims is not None:
            claims = claims[: self.cfg.limit_claims]
        return claims, sha, n_in_file

    def _web_config(self) -> Any:
        """Return the backend's web settings (``web`` runs only), or raise RunError."""
        web = getattr(self.backend, "web", None)
        if web is None:
            raise RunError(f"the web condition needs the subscription backend with its web "
                           f"settings; backend {self.backend.name!r} has none")
        return web

    def _base_manifest(self, claims_sha: str, n_claims: int) -> dict[str, Any]:
        """Return the manifest fields fixed for the life of the run folder."""
        manifest = self._manifest_core(claims_sha, n_claims)
        if WEB_CONDITION in self.cfg.conditions:
            manifest["web"] = self._web_config().describe()
        return manifest

    def _manifest_core(self, claims_sha: str, n_claims: int) -> dict[str, Any]:
        """Return the manifest fields every run has."""
        p = self.cfg.prompt
        return {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self.cfg.run_dir.name,
            "backend": self.backend.name,
            "models": dict(self.cfg.models),
            "effort": self.cfg.effort,
            "prompt": {"prompt_id": p.prompt_id, "prompt_hash": p.prompt_hash,
                       "files": p.file_hashes, "directory": str(p.directory),
                       "generic_probe": p.is_generic_probe},
            "retrieval": self.retriever.describe(),
            "claims_file": {"path": str(self.cfg.claims_path), "sha256": claims_sha,
                            "n_claims_in_file": n_claims},
            "backend_info": {},
            "notes": {
                "list_price_usd_not_billed": "the CLI's total_cost_usd: what the call would "
                "cost on the API at list price; the subscription does not bill it",
                "computed_usage": "sums computed by the runner from the backend's own usage "
                "fields (the CLI's modelUsage, or the API's usage)",
                "cost_usd": "API backend only: a list-price computation from each response's "
                "usage (prices and their source in backend_info), not the invoice",
            },
        }

    def prepare(self) -> tuple[list[Item], list[Item], dict[str, Any], int]:
        """Plan, guard, compare manifests. Returns (planned, pending, manifest, n_in_file)."""
        cfg = self.cfg
        unknown = [c for c in cfg.conditions if c not in cfg.prompt.conditions]
        if unknown or not cfg.conditions:
            raise RunError(f"prompt {cfg.prompt.prompt_id!r} has no condition {unknown or '(none)'}"
                           f"; it has {list(cfg.prompt.conditions)}")
        if WEB_CONDITION in cfg.conditions:
            self._web_config()
        claims, sha, n_in_file = self._claims()
        if "gold" in cfg.conditions:
            empty = [c["id"] for c in claims if c.get("gold_set") is None or not c["gold_evidence"]]
            if empty:
                raise RunError(f"claims without gold evidence cannot run 'gold': {empty[:10]}")
        planned = plan_items(claims, cfg.conditions, cfg.models, cfg.effort,
                             cfg.prompt.prompt_hash)
        rows = load_rows(self.results_path)
        done = final_keys(rows)
        done_triples = [(r["model_requested"], r["claim_id"], r["condition"])
                        for r in rows if r.get("key") in done]
        counts = items_per_model([(i.model_id, i.claim_id, i.condition) for i in planned],
                                 done_triples)
        log.info("items per model in this run (planned + already done): %s", counts)
        check_item_budget(cfg.prompt.prompt_id, counts, cfg.full)
        manifest = self._base_manifest(sha, n_in_file)
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            # the backend version is compared in run(), once the backend has been asked
            check_resume(old, manifest | {"backend_info": old.get("backend_info", {})},
                         cfg.allow_cli_change)
            manifest = old
        pending = [i for i in planned if i.key not in done]
        return planned, pending, manifest, n_in_file

    # ----------------------------------------------------------------------------------
    def render(self, item: Item) -> str:
        """Return the user turn of one item."""
        fields: dict[str, Any] = {"claim": item.claim["claim"]}
        if item.condition == "retrieved":
            fields["evidence"] = render_retrieved(self.hits[item.claim_id])
        elif item.condition == "gold":
            fields["evidence"] = render_gold(item.claim["gold_evidence"])
        return self.cfg.prompt.render_user(item.condition, fields)

    def request(self, item: Item) -> CallRequest:
        """Return the backend request of one item."""
        p = self.cfg.prompt
        text = self.render(item)
        prefix = ""
        if CACHE_BREAK in text:
            head, _, tail = text.partition(CACHE_BREAK)
            prefix, text = head, head + tail.lstrip("\n")
        return CallRequest(model_id=item.model_id, effort=self.cfg.effort, system=p.system,
                           user_text=text, schema=p.schema, schema_json=p.schema_json,
                           label_path=p.label_path, cache_prefix=prefix,
                           condition=item.condition)

    def row(self, item: Item, request: CallRequest, res: CallResult) -> dict[str, Any]:
        """Return the results row of one answer."""
        retrieved = None
        if item.condition == "retrieved":
            retrieved = [{k: h[k] for k in ("rank", "chunk_id", "page_id", "title",
                                            "section_path", "distance")}
                         for h in self.hits[item.claim_id]]
        raw_name = None
        if res.raw_stream:
            raw_name = f"raw/{item.claim_id}__{item.condition}__{item.model_option}__" \
                       f"{item.key[:12]}__{int(time.time() * 1000)}.jsonl"
        row = {
            "row_version": ROW_VERSION,
            "key": item.key,
            "claim_id": item.claim_id,
            "condition": item.condition,
            "gold_label": item.claim["label"],
            "kind": item.claim.get("kind"),
            "backend": self.backend.name,
            "model_option": item.model_option,
            "model_requested": item.model_id,
            "model_served": res.served_models[0] if len(res.served_models) == 1
            else (res.served_models or None),
            "side_models": res.side_models,
            "effort": request.effort,
            "prompt_id": self.cfg.prompt.prompt_id,
            "prompt_hash": self.cfg.prompt.prompt_hash,
            "user_text_sha256": sha256_text(request.user_text),
            "retrieved": retrieved,
            "gold_set": item.claim.get("gold_set") if item.condition == "gold" else None,
            "status": res.status,
            "final": res.final,
            "label": res.label,
            "parse_status": res.parse_status,
            "answer": res.answer,
            "error": res.error,
            "error_kind": res.error_kind,
            "stop_signal": res.stop_signal,
            "latency_s": res.latency_s,
            "elapsed_s_total": res.elapsed_s_total,
            "attempts": res.attempts,
            "attempt_log": res.attempt_log,
            "cli": res.cli or None,
            "api": res.api or None,
            "cost_usd": res.cost_usd,
            "computed_usage": res.computed_usage,
            "window_utilization": res.window_utilization or None,
            "raw_response": res.raw_response,
            "raw_stream_file": raw_name,
            "backend_version": self.cfg_cli_version,
            "cli_version": self.cfg_cli_version,
            "ts": now_iso(),
        }
        if item.condition == WEB_CONDITION:
            web = res.web or {}
            row.update({"web_trace": web.get("trace"), "n_web_search": web.get("n_web_search"),
                        "n_web_fetch": web.get("n_web_fetch"), "n_denied": web.get("n_denied"),
                        "web_search_requests": web.get("web_search_requests"),
                        "assistant_models": web.get("assistant_models"),
                        "num_turns": (res.cli or {}).get("num_turns")})
        return row

    def write_row(self, row: dict[str, Any], raw_stream: str) -> None:
        """Append one row (flush + fsync, under a lock) and its raw stream file."""
        if row["raw_stream_file"]:
            raw_path = self.cfg.run_dir / row["raw_stream_file"]
            tmp = raw_path.with_suffix(".tmp")
            tmp.write_text(raw_stream, encoding="utf-8")
            os.replace(tmp, raw_path)
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.state.write_lock, self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def account(self, item: Item, res: CallResult) -> None:
        """Count one result and apply the stop rules (under the state lock)."""
        st = self.state
        with st.lock:
            st.counts[res.status] += 1
            if res.cli.get("last_rate_limit_event"):
                st.rate_limit_events += 1
            if res.window_utilization:
                reading = {"ts": now_iso(), "claim_id": item.claim_id,
                           "condition": item.condition, "model": item.model_id,
                           **res.window_utilization}
                st.first_window = st.first_window or reading
                st.last_window = reading
            if res.status == "window":
                st.request_stop(EXIT_WINDOW, res.stop_signal or "window spent")
            elif res.status == "money":
                st.request_stop(EXIT_MONEY, res.stop_signal or f"money: {res.error}")
            elif res.status == "fatal":
                st.request_stop(EXIT_FATAL, f"fatal: {res.error}")
            if res.stop_signal and res.stop_signal.startswith("overage"):
                st.request_stop(EXIT_OVERAGE, res.stop_signal)
            if res.stop_signal and res.stop_signal.startswith("money"):
                st.request_stop(EXIT_MONEY, res.stop_signal)
            for name, value in (res.window_utilization or {}).items():
                if value is None:
                    continue
                if name == "five_hour" and value >= self.cfg.stop_five_hour:
                    st.request_stop(EXIT_WINDOW, f"five_hour utilization {value:.2f} >= "
                                                 f"{self.cfg.stop_five_hour:.2f}")
                if name.startswith(WEEKLY_PREFIX) and value >= self.cfg.stop_weekly:
                    st.request_stop(EXIT_WINDOW, f"{name} utilization {value:.2f} >= "
                                                 f"{self.cfg.stop_weekly:.2f}")
            if res.status in ("error",):
                st.consecutive_errors += 1
                if st.consecutive_errors >= BREAKER_LIMIT:
                    st.request_stop(EXIT_BREAKER, f"{BREAKER_LIMIT} items in a row ended in "
                                                  f"an error (last: {res.error})")
            elif res.status in FINAL_STATUSES:
                st.consecutive_errors = 0

    def memory_gate(self) -> None:
        """Wait while MemAvailable is below ``min_free_mb`` (re-checked every 15 s)."""
        warned = False
        while not self.state.stop.is_set():
            free = mem_available_mb()
            if free is None or free >= self.cfg.min_free_mb:
                return
            if not warned:
                log.warning("only %d MB available (< %d): waiting", free, self.cfg.min_free_mb)
                warned = True
            self.state.stop.wait(15)

    def work(self, queue: list[list[Item]], queue_lock: threading.Lock) -> None:
        """A worker thread: take the next group of items and run them in order (call, write the
        row, apply the stop rules). Without --group-by every group is one item."""
        while not self.state.stop.is_set():
            with queue_lock:
                if not queue:
                    return
                group = queue.pop(0)
            for item in group:
                if self.state.stop.is_set():
                    return
                self.run_item(item)

    def run_item(self, item: Item) -> None:
        """One item: memory gate, the call, the row, the stop rules."""
        self.memory_gate()
        if self.state.stop.is_set():
            return
        try:
            request = self.request(item)
            res = self.backend.call(request, self.state.stop)
        except Exception as exc:  # one item must not kill the worker
            log.exception("claim %s %s %s: unexpected error", item.claim_id,
                          item.condition, item.model_id)
            with self.state.lock:
                self.state.consecutive_errors += 1
                if self.state.consecutive_errors >= BREAKER_LIMIT:
                    self.state.request_stop(EXIT_BREAKER, f"unexpected errors ({exc})")
            return
        if res.attempts == 0:       # nothing was sent: no row, but a money stop still stops
            if res.stop_signal and res.stop_signal.startswith("money"):
                self.state.request_stop(EXIT_MONEY, res.stop_signal)
            return
        self.write_row(self.row(item, request, res), res.raw_stream)
        self.account(item, res)
        log.info("claim %s %-12s %-18s %s label=%s %.1fs", item.claim_id, item.condition,
                 item.model_id, res.status, res.label, res.latency_s or 0.0)

    # ----------------------------------------------------------------------------------
    def run(self) -> int:
        """Run the invocation and return its exit code."""
        cfg = self.cfg
        try:
            planned, pending, manifest, _ = self.prepare()
        except (RunError, GuardError, ClaimsError) as exc:
            log.error("%s", exc)
            return EXIT_CONFIG
        self.cfg_cli_version = manifest.get("cli", {}).get("version")
        log.info("%d items planned, %d already final, %d to call", len(planned),
                 len(planned) - len(pending), len(pending))
        if cfg.dry_run:
            self.dry_run(planned)
            return EXIT_OK
        info = self.backend.describe()
        old_info = manifest.get("backend_info") or {}
        if old_info and old_info.get("version") != info["version"] and not cfg.allow_cli_change:
            log.error("the backend version changed (%s -> %s); pass --allow-cli-change or start "
                      "a new run folder", old_info.get("version"), info["version"])
            return EXIT_CONFIG
        auth = self.backend.preflight(list(cfg.models.values()))
        manifest.setdefault("created_at", now_iso())
        manifest["backend_info"] = old_info or info
        if self.backend.name == "claude_cli_subscription":
            manifest["cli"] = manifest.get("cli") or info       # the same, under its older name
        manifest["auth"] = auth
        self.cfg_cli_version = info["version"]
        self.backend.start(load_rows(self.results_path))
        if "retrieved" in cfg.conditions:
            claims = {i.claim_id: i.claim["claim"] for i in planned if i.condition == "retrieved"}
            try:
                manifest["retrieval"]["n_chunks_at_start"] = self.retriever.count_chunks()
                self.hits = self.retriever.retrieve_many(sorted(claims.items()))
            except Exception as exc:  # noqa: BLE001 - no call is made without the evidence
                log.error("retrieval failed, no call made: %s: %s", type(exc).__name__, exc)
                return EXIT_CONFIG
        invocation = {"started_at": now_iso(), "argv": cfg.argv, "conditions": cfg.conditions,
                      "limit_claims": cfg.limit_claims, "workers": cfg.workers,
                      "n_planned": len(planned), "n_pending": len(pending),
                      "backend": self.backend.name, "backend_version": info["version"],
                      "group_by": cfg.group_by, "git": git_info(),
                      "thresholds": {"five_hour": cfg.stop_five_hour, "weekly": cfg.stop_weekly}}
        manifest.setdefault("invocations", []).append(invocation)
        manifest["started_at"] = manifest.get("started_at") or invocation["started_at"]
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        repair_torn_tail(self.results_path)
        write_json_atomic(self.manifest_path, manifest)

        code = self._execute(pending)

        invocation.update({"ended_at": now_iso(), "exit_code": code,
                           "stop_reason": self.state.stop_reason or None,
                           "counts": dict(self.state.counts),
                           "window_first": self.state.first_window,
                           "window_last": self.state.last_window,
                           "rate_limit_events": self.state.rate_limit_events})
        manifest["ended_at"] = invocation["ended_at"]
        write_json_atomic(self.manifest_path, manifest)
        log.info("done: exit %d %s | %s | window first %s last %s", code,
                 self.state.stop_reason, dict(self.state.counts), self.state.first_window,
                 self.state.last_window)
        return code

    def _execute(self, pending: list[Item]) -> int:
        """Run the pending items on the worker threads; return the exit code."""
        if not pending:
            return EXIT_OK
        queue = group_items(pending, self.cfg.group_by)
        queue_lock = threading.Lock()

        def on_signal(signum: int, _frame: Any) -> None:
            self.state.request_stop(EXIT_SIGNAL, f"signal {signum}")
            self.backend.kill_all()

        old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM,
                                                         signal.SIGHUP)}
        threads = [threading.Thread(target=self.work, args=(queue, queue_lock),
                                    name=f"w{n}", daemon=True)
                   for n in range(min(self.cfg.workers, len(queue)))]
        try:
            for t in threads:
                t.start()
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)
        if self.state.stop.is_set():
            return self.state.stop_code
        done = final_keys(load_rows(self.results_path))
        missing = [i for i in pending if i.key not in done]
        if missing:
            log.warning("%d items not final (errors): re-run the same command", len(missing))
            return EXIT_INCOMPLETE
        return EXIT_OK

    def dry_run(self, planned: list[Item]) -> None:
        """Print the plan, the argv (system prompt elided) and one user turn per condition."""
        print(json.dumps({"items": len(planned),
                          "per_model": Counter(i.model_id for i in planned),
                          "per_condition": Counter(i.condition for i in planned)}, indent=1))
        shown: set[str] = set()
        for item in planned:
            if item.condition in shown:
                continue
            shown.add(item.condition)
            if item.condition == "retrieved":
                cached = self.retriever._read_cache(item.claim_id, item.claim["claim"])
                if cached is None:
                    print(f"----- {item.condition}: retrieval not cached; not run in a dry run")
                    continue
                self.hits[item.claim_id] = cached
            request = self.request(item)
            argv_fn = getattr(self.backend, "argv", None)
            kwargs_fn = getattr(self.backend, "request_kwargs", None)
            if argv_fn is not None:
                argv = [f"<system prompt: {len(request.system)} chars>" if a == request.system
                        else a for a in argv_fn(request)]
                print(json.dumps(argv))
            if kwargs_fn is not None:
                kwargs = kwargs_fn(request)
                print(json.dumps(kwargs, ensure_ascii=False, indent=1))
                problems = request_parity(request, kwargs)
                print(f"----- parity with the subscription request ({item.condition}): "
                      + ("MATCH" if not problems else "DIFFERS: " + "; ".join(problems)))
            print(f"----- stdin of claim {item.claim_id}, {item.condition} -----")
            print(request.user_text)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def group_items(items: list[Item], group_by: str | None) -> list[list[Item]]:
    """Return the work queue: groups of items, in the order of their first item.

    Without ``group_by`` every item is its own group. With it, the items of one model and one
    condition whose claims share the value of ``group_by`` form one group, which one worker runs
    back to back so a cached shared prefix is read by the next item (``--group-by``).
    """
    if not group_by:
        return [[item] for item in items]
    groups: dict[tuple[str, str, str], list[Item]] = {}
    for item in items:
        value = json.dumps(item.claim.get(group_by), sort_keys=True, ensure_ascii=False)
        groups.setdefault((item.model_id, item.condition, value), []).append(item)
    return list(groups.values())


def request_parity(request: CallRequest, kwargs: dict[str, Any]) -> list[str]:
    """Return how an API request differs from the subscription request built from the same
    :class:`CallRequest` (empty when the model, effort, system text, user text and schema
    are the same; the CLI sends exactly ``request``'s fields)."""
    problems: list[str] = []
    if kwargs.get("model") != request.model_id:
        problems.append(f"model {kwargs.get('model')!r}")
    if (kwargs.get("output_config") or {}).get("effort") != request.effort:
        problems.append("effort")
    system = "".join(b.get("text", "") for b in kwargs.get("system") or [])
    if system != request.system:
        problems.append("system text")
    content = kwargs["messages"][0]["content"]
    user = content if isinstance(content, str) else "".join(b["text"] for b in content)
    if user != request.user_text:
        problems.append("user text")
    sent = ((kwargs.get("output_config") or {}).get("format") or {}).get("schema")
    if sent != request.schema:
        problems.append("schema (after the API's keyword sanitising)")
    for forbidden in ("temperature", "top_p", "top_k", "tools", "tool_choice", "fallbacks"):
        if forbidden in kwargs:
            problems.append(f"sends {forbidden}")
    return problems


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of ``scripts/llm_run.py``."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", choices=["subscription", "api"], default="subscription",
                   help="subscription: Claude Code CLI on the subscription; api: Anthropic API "
                        "with the key in the repo's .env (needs --max-usd)")
    p.add_argument("--runs-root", required=True, type=Path,
                   help="folder of the run folders (outside the repository)")
    p.add_argument("--run-id", default=None, help="run folder name (default: time + prompt)")
    p.add_argument("--claims", required=True, type=Path, help="claims file (the contract)")
    p.add_argument("--claims-sha256", default=None, help="expected sha256 of the claims file")
    p.add_argument("--prompt", required=True, help="prompt id (folder in wikilense/llm/prompts)")
    p.add_argument("--models", required=True, help="comma-separated: sonnet, opus, haiku")
    p.add_argument("--effort", required=True, choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--conditions", default=None,
                   help="comma-separated (default: every condition of the prompt set)")
    p.add_argument("--limit-claims", type=int, default=None, help="only the first N claims")
    p.add_argument("--workers", type=int, default=5, help="parallel calls (1-8)")
    p.add_argument("--group-by", default=None,
                   help="claim field: items sharing it (same model and condition) run on one "
                        "worker, back to back (for a cached <<<CACHE_BREAK>>> prefix)")
    p.add_argument("--database", default=None, help="database for 'retrieved' (default: settings)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--strategy", default="inline")
    p.add_argument("--ef-search", type=int, default=None)
    p.add_argument("--timeout-s", type=float, default=None,
                   help="per call (default: 300 subscription, 480 api)")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--min-free-mb", type=int, default=2500)
    p.add_argument("--allow-cli-change", action="store_true",
                   help="resume although the CLI / SDK version changed")
    g = p.add_argument_group("subscription backend")
    g.add_argument("--binary", default=None, help="claude binary (default: $CLAUDE_BIN)")
    g.add_argument("--stop-five-hour", type=float, default=0.80)
    g.add_argument("--stop-weekly", type=float, default=0.80)
    g.add_argument("--allow-5m-cache-writes", action="store_true")
    w = p.add_argument_group("web condition (subscription only)")
    w.add_argument("--web-max-turns", type=int, default=None,
                   help="the CLI's --max-turns for web calls (default: web.DEFAULT_MAX_TURNS)")
    w.add_argument("--web-timeout-s", type=float, default=None,
                   help="per web call (default: web.DEFAULT_TIMEOUT_S)")
    w.add_argument("--web-deny-hosts", type=Path, default=None,
                   help="WebFetch deny list (default: wikilense/llm/web_deny_hosts.txt)")
    a = p.add_argument_group("api backend")
    a.add_argument("--max-usd", type=float, default=None,
                   help="REQUIRED with --backend api: stop at this list-price spend, counting "
                        "the earlier rows of the same results file")
    a.add_argument("--env-file", type=Path, default=None,
                   help="dotenv file holding ANTHROPIC_API_KEY (default: the repo's .env)")
    a.add_argument("--cache-ttl", choices=["5m", "1h", "off"], default="5m")
    a.add_argument("--thinking-display", choices=["omitted", "summarized"], default="omitted")
    a.add_argument("--max-tokens", type=int, default=None, help="default: by effort (low 4,096)")
    p.add_argument("--full", action="store_true",
                   help="allow more than 50 items per model (never with a generic_probe prompt)")
    p.add_argument("--dry-run", action="store_true",
                   help="plan and print; no call, no key, no database")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def make_backend(args: argparse.Namespace, runs_root: Path) -> Backend:
    """Return the backend ``--backend`` names. Raises RunError for a bad configuration."""
    if args.backend == "api":
        from wikilense.config import DEFAULT_ENV_FILE
        from wikilense.llm.api_backend import AnthropicApiBackend

        if args.max_usd is None:
            raise RunError("--max-usd is required with --backend api")
        return AnthropicApiBackend(
            args.env_file or DEFAULT_ENV_FILE, LABELS, max_usd=args.max_usd,
            timeout_s=args.timeout_s or 480.0, max_attempts=args.max_attempts,
            cache_ttl=args.cache_ttl, thinking_display=args.thinking_display,
            max_tokens=args.max_tokens)
    from wikilense.llm import web as web_mod
    from wikilense.llm.cli_backend import ClaudeCliBackend

    binary = args.binary or os.environ.get("CLAUDE_BIN")
    if not binary:
        raise RunError("no claude binary: pass --binary or set CLAUDE_BIN")
    web = web_mod.WebConfig(
        deny=web_mod.load_deny_hosts(args.web_deny_hosts),
        max_turns=args.web_max_turns or web_mod.DEFAULT_MAX_TURNS,
        timeout_s=args.web_timeout_s or web_mod.DEFAULT_TIMEOUT_S)
    return ClaudeCliBackend(binary, runs_root / ".empty_cwd", LABELS,
                            timeout_s=args.timeout_s or 300.0, max_attempts=args.max_attempts,
                            allow_5m_cache_writes=args.allow_5m_cache_writes, web=web)


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``scripts/llm_run.py``."""
    from wikilense.config import load_settings
    from wikilense.llm.api_backend import ApiBackendError
    from wikilense.llm.cli_backend import CliBackendError
    from wikilense.llm.prompts import load_prompt_set
    from wikilense.llm.retrieval import Retriever

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    for name in ("anthropic", "httpx", "httpcore"):     # never debug-log HTTP traffic
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        models = resolve_models(args.models)
        prompt = load_prompt_set(args.prompt)
        if not 1 <= args.workers <= 8:
            raise RunError("--workers must be 1..8 (at most 5 parallel calls have been measured)")
        if args.workers > 5:
            log.warning("more than 5 parallel calls was never measured on the CLI")
        conditions = ([c.strip() for c in args.conditions.split(",") if c.strip()]
                      if args.conditions else list(prompt.conditions))
        run_id = args.run_id or datetime.now().astimezone().strftime(
            f"%Y%m%dT%H%M%S_{prompt.prompt_id}")
        runs_root = args.runs_root.resolve()
        run_dir = runs_root / run_id
        if conditions and set(conditions) <= NO_DATABASE_CONDITIONS:
            if args.database:
                raise RunError(f"conditions {conditions} use no database; drop --database")
            retriever: Any = NoRetrieval()           # no settings, no MariaDB connection
        else:
            settings = load_settings()
            database = args.database or settings.db_name
            retriever = Retriever(settings, database, runs_root / "retrieval_cache", k=args.k,
                                  strategy=args.strategy, ef_search=args.ef_search)
        backend = make_backend(args, runs_root)
    except (RunError, CliBackendError, ApiBackendError, ValueError, OSError) as exc:
        log.error("%s", exc)
        return EXIT_CONFIG
    config = RunConfig(
        run_dir=run_dir, claims_path=args.claims.resolve(), prompt=prompt, models=models,
        effort=args.effort, conditions=conditions,
        limit_claims=args.limit_claims, workers=args.workers, full=args.full,
        dry_run=args.dry_run, stop_five_hour=args.stop_five_hour, stop_weekly=args.stop_weekly,
        min_free_mb=args.min_free_mb, claims_sha256=args.claims_sha256,
        allow_cli_change=args.allow_cli_change, group_by=args.group_by,
        argv=sys.argv if argv is None else list(argv),
    )
    return run_with_lock(config, backend, retriever, runs_root)


def run_with_lock(config: RunConfig, backend: Backend, retriever: Any, runs_root: Path) -> int:
    """Run under the machine-wide runner lock: one runner at a time (at most ``workers``
    calls in flight on the machine)."""
    if config.dry_run:
        return Runner(config, backend, retriever).run()
    runs_root.mkdir(parents=True, exist_ok=True)
    lock_path = runs_root / ".llm_runner.lock"
    with lock_path.open("a+") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.error("another LLM runner holds %s: one runner at a time", lock_path)
            return EXIT_CONFIG
        config.run_dir.mkdir(parents=True, exist_ok=True)
        from wikilense.llm.api_backend import ApiBackendError
        from wikilense.llm.cli_backend import CliBackendError

        try:
            return Runner(config, backend, retriever).run()
        except (CliBackendError, ApiBackendError) as exc:   # login, key, refused variables
            log.error("%s", exc)
            return EXIT_FATAL
