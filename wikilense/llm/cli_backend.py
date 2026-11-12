"""The subscription backend: one ``claude -p`` (Claude Code CLI, headless) call per item.

Ported from the kit's ``code/standalone/cc_runner.py`` (itself distilled from the
running_benchmarks client, ``code/reference/src/client_claude_cli.py``), with the fixes the
checked review of the kit asked for (``reviews/LLM_RUNNER_KIT_REVIEW_2026-11-12.md``, ids in
brackets). ``wikilense/llm/NOTES.md`` lists every choice with its source.

- argv: :data:`BASE_FLAGS`, then ``--model <full id> --effort <level> --system-prompt <text>
  --json-schema <schema>``. The user turn goes on stdin, never on argv (kit D-42, REPORT §2.1).
- Child environment: every ``CLAUDE*``, ``ANTHROPIC*`` and ``AI_AGENT*`` variable is removed
  (this session exports ``CLAUDE_EFFORT`` and ``CLAUDECODE``; an API key could make the CLI
  bill), plus the caching and thinking variables without that prefix [A19, C22]; THEN
  ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`` (no hidden Haiku call, kit D-48) and
  ``CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1`` (no automatic re-send of a flagged claim to
  another model; read by CLI 2.1.322 [C8]) are added. Order matters: both start with CLAUDE.
- cwd: a fixed empty folder (the CLI's default prompt includes cwd and git status).
- Failures are classified in the kit's order (auth, not-your-limit, spent window, refusal,
  transient). The spent-window markers include CLI 2.1.322's wording "You've hit your 5-hour
  limit" (also Sonnet, Fable and usage credit limits) [C1], and a failed call whose
  ``rate_limit_info.status`` is ``rejected`` counts as a spent window too.
- Served model: the requested full id must appear in ``modelUsage`` (exact, or id + ``[...]``;
  for Haiku exactly the id). Anything else is fatal. Another model that wrote output is the
  CLI's fallback: the row is a refusal, never a label (kit D-95).
- Parsing: ``result.structured_output`` first, then the ``StructuredOutput`` tool blocks from
  last to first; the first one valid against the schema wins [A21, C15]. No valid one after the
  retries: ``parse_failed``. A label is never read from free text.
- Retries: transient errors and invalid answers up to ``max_attempts`` with 10/20/40 s back-off
  (+-20%), no back-off after the last attempt [C13]; at most 2 timeouts; a timeout whose output
  already holds a valid answer keeps that answer [C10].
- Paid overflow: ``isUsingOverage``, ``overageInUse``, ``rateLimitType: "overage"`` or a grace
  period paid by extra usage, and any 5-minute cache write (Claude Code drops to the 5-minute
  TTL when usage credits pay, kit D-59), set a stop signal [A1, B1, C2].
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from wikilense.llm.backend import CallRequest, CallResult
from wikilense.llm.models import check_full_id

#: Every flag was on every call of the kit project's CLI route (CLAUDE_CLI_BASE_FLAGS).
BASE_FLAGS: tuple[str, ...] = (
    "--print",                         # headless
    "--output-format", "stream-json",  # content blocks + rate_limit_event + result event
    "--verbose",                       # required by the CLI for stream-json in --print mode
    "--safe-mode",                     # no CLAUDE.md, skills, plugins, hooks, MCP
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--setting-sources", "",           # load no settings file
    "--no-session-persistence",        # no resumable session file per call
    "--tools", "none",                 # no built-in tools; never a bare --tools (it swallowed
                                       # --model once, kit D-42). The kit used "none"; 2.1.322's
                                       # help documents "" (kit D-42: only "none" was tested)
)
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

STRIPPED_ENV_PREFIXES: tuple[str, ...] = ("CLAUDE", "ANTHROPIC", "AI_AGENT",
                                          "DISABLE_PROMPT_CACHING")
STRIPPED_ENV_NAMES: frozenset[str] = frozenset({
    "FORCE_PROMPT_CACHING_5M", "ENABLE_PROMPT_CACHING_1H", "MAX_THINKING_TOKENS",
    "API_TIMEOUT_MS", "DISABLE_INTERLEAVED_THINKING", "ELECTRON_RUN_AS_NODE",
    "MCP_CONNECTION_NONBLOCKING",
})
FORCED_CHILD_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
}

#: Checked in this order; not-your-limit first, because the throttle message says "usage limit".
NOT_YOUR_LIMIT_MARKERS = ("not your usage limit", "temporarily limiting requests",
                          "temporary capacity issue", "server is temporarily")
LIMIT_EXHAUSTED_MARKERS = (
    "hit your session limit", "hit your weekly limit", "hit your opus limit",
    "hit your usage limit", "usage limit reached", "usage limit has been reached",
    "claude usage limit", "session limit reached", "weekly limit reached",
    # CLI 2.1.322 (binary strings, checked 2026-11-12): "You've hit your ${name}" with name in
    # 5-hour limit, weekly limit, Opus limit, Sonnet limit, Fable limit, usage credit limit
    "hit your 5-hour limit", "hit your sonnet limit", "hit your fable limit",
    "hit your usage credit limit",
)
_HIT_YOUR_LIMIT_RE = re.compile(r"hit your [a-z0-9 \-]{0,40}limit")
REFUSAL_MARKERS = ("violate our usage policy", "unable to respond to this request",
                   "safeguards flagged", "can't respond to this message",
                   "anthropic.com/legal/aup")
TRANSIENT_MARKERS = ("overloaded", "rate limit", "rate_limit", "timeout", "timed out",
                     "503", "529", "500", "502", "504", "econnreset", "socket hang up")
AUTH_MARKERS = ("not logged in", "please run /login", "invalid api key",
                "authentication_error", "oauth token has expired")
ALLOWED_INIT_TOOLS = frozenset({"StructuredOutput"})
MAX_TIMEOUTS = 2


class CliBackendError(RuntimeError):
    """The backend cannot be used as configured (no binary, not logged in, API-key auth)."""


# --------------------------------------------------------------------------------------
# Pure helpers (tested directly)
# --------------------------------------------------------------------------------------
def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment of every ``claude`` child: stripped, then the forced variables."""
    source = os.environ if base is None else base
    env = {
        k: v for k, v in source.items()
        if not any(k.upper().startswith(p) for p in STRIPPED_ENV_PREFIXES)
        and k.upper() not in STRIPPED_ENV_NAMES
    }
    env.update(FORCED_CHILD_ENV)
    return env


def classify(text: str, rate_info: dict[str, Any] | None = None) -> str:
    """Return ``auth`` | ``exhausted`` | ``refusal`` | ``transient`` | ``""`` for a failed call."""
    low = text.lower().replace("’", "'")
    if any(m in low for m in AUTH_MARKERS):
        return "auth"
    if any(m in low for m in NOT_YOUR_LIMIT_MARKERS):
        return "transient"
    if any(m in low for m in LIMIT_EXHAUSTED_MARKERS) or _HIT_YOUR_LIMIT_RE.search(low):
        return "exhausted"
    if (rate_info or {}).get("status") == "rejected":
        return "exhausted"
    if any(m in low for m in REFUSAL_MARKERS):
        return "refusal"
    if any(m in low for m in TRANSIENT_MARKERS):
        return "transient"
    return ""


def parse_stream(stdout: str) -> dict[str, Any]:
    """Return the parts of one call's stream-json output.

    Keys: ``result`` (the result event), ``rate_event`` (the last ``rate_limit_event``, raw),
    ``rate_info`` (its ``rate_limit_info``), ``init`` (the ``system/init`` event) and
    ``structured_blocks`` (the inputs of every ``StructuredOutput`` tool_use block, in order).
    """
    parts: dict[str, Any] = {"result": {}, "rate_event": {}, "rate_info": {}, "init": {},
                             "structured_blocks": []}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "assistant":
            content = (event.get("message") or {}).get("content")
            for block in content if isinstance(content, list) else []:
                if (isinstance(block, dict) and block.get("type") == "tool_use"
                        and block.get("name") == "StructuredOutput"):
                    parts["structured_blocks"].append(block.get("input"))
        elif etype == "result":
            parts["result"] = event
        elif etype == "rate_limit_event":
            info = event.get("rate_limit_info") or event.get("rate_limit") or event.get("rateLimit")
            parts["rate_event"] = event
            parts["rate_info"] = info if isinstance(info, dict) else {}
        elif etype == "system" and event.get("subtype") == "init":
            parts["init"] = event
    return parts


def windows_of(rate_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{window name: {utilization, resetsAt, status, ...}}`` from a rate_limit_info."""
    windows = {k: v for k, v in dict(rate_info.get("unifiedWindows") or {}).items()
               if isinstance(v, dict)}
    kind = rate_info.get("rateLimitType")
    if kind and kind not in windows and ("utilization" in rate_info or "resetsAt" in rate_info):
        windows[kind] = {k: rate_info.get(k) for k in ("utilization", "resetsAt", "status")}
    return windows


def utilization_of(rate_info: dict[str, Any]) -> dict[str, float | None]:
    """Return ``{window: utilization as a fraction}``; None where the CLI sent no number.

    Always a fraction (CLI 2.1.322: "utilization is the fraction of the window used ... values
    above 1 occur when usage legitimately runs past a window's cap"), so 1.04 means past the
    cap, never 1.04 % [C2]. A missing value is None, never 0 [A3].
    """
    result: dict[str, float | None] = {}
    for name, window in windows_of(rate_info).items():
        value = window.get("utilization")
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        result[name] = float(value) if ok else None
    return result


def overage_signal(rate_info: dict[str, Any]) -> str:
    """Return why the call looks paid by extra usage, or ``""`` (CLI 2.1.322 field names)."""
    kind = rate_info.get("rateLimitType")
    if rate_info.get("isUsingOverage") is True and kind != "seven_day_overage_included":
        return "rate_limit_info.isUsingOverage is true"
    if rate_info.get("overageInUse") is True:
        return "rate_limit_info.overageInUse is true"
    if kind == "overage":
        return "rate_limit_info.rateLimitType is 'overage'"
    if (rate_info.get("rateLimitGraceActive") is True
            and rate_info.get("overageStatus") in ("allowed", "allowed_warning")):
        return "a grace period paid by extra usage (rateLimitGraceActive, overageStatus allowed)"
    return ""


def served_names(model_usage: dict[str, Any], model_id: str) -> list[str]:
    """Return the ``modelUsage`` keys that are the requested model.

    Exact id, or id + the CLI's ``[...]`` suffix (``claude-opus-4-8[1m]``); never a prefix test
    (``claude-opus-5-5`` starts with ``claude-opus-5``). For a Haiku request only the exact id.
    """
    exact_only = model_id.startswith("claude-haiku")
    return [name for name in model_usage
            if name == model_id or (not exact_only and name.startswith(model_id + "["))]


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return the problems of ``instance`` against a small JSON-Schema subset (kit validator).

    Supports type, enum, properties, required, additionalProperties false, items.
    """
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int, "number": (int, float), "null": type(None)}
    want = schema.get("type")
    if want:
        allowed = want if isinstance(want, list) else [want]
        ok = any(
            isinstance(instance, types[t])
            and not (t in ("integer", "number") and isinstance(instance, bool))
            for t in allowed if t in types
        )
        if not ok:
            return [f"{path}: expected {want}, got {type(instance).__name__}"]
    errors: list[str] = []
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        errors += [f"{path}: missing {k!r}" for k in schema.get("required", []) if k not in instance]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected {k!r}" for k in instance if k not in props]
        for key, sub in props.items():
            if key in instance and isinstance(sub, dict):
                errors += validate(instance[key], sub, f"{path}.{key}")
    if isinstance(instance, list) and isinstance(schema.get("items"), dict):
        for i, value in enumerate(instance):
            errors += validate(value, schema["items"], f"{path}[{i}]")
    return errors


def dig(obj: Any, path: tuple[str, ...]) -> Any:
    """Follow ``path`` through nested dicts; None when a step is missing."""
    for part in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def pick_answer(parts: dict[str, Any], schema: dict[str, Any], label_path: tuple[str, ...],
                labels: tuple[str, ...]) -> tuple[dict[str, Any] | None, str | None, str, list[str]]:
    """Return ``(answer, label, source, problems)`` from one call's stream parts.

    Candidates: ``result.structured_output``, then the StructuredOutput tool blocks from last to
    first. The first candidate valid against ``schema`` whose label is one of ``labels`` wins;
    ``source`` says which (``structured_output`` or ``tool_use[i]``). Without one, ``answer``
    and ``label`` are None and ``problems`` says why.
    """
    candidates: list[tuple[str, Any]] = []
    so = parts["result"].get("structured_output")
    if so is not None:
        candidates.append(("structured_output", so))
    blocks = parts["structured_blocks"]
    candidates += [(f"tool_use[{i}]", blocks[i]) for i in range(len(blocks) - 1, -1, -1)]
    problems: list[str] = []
    for source, candidate in candidates:
        found = validate(candidate, schema)
        label = dig(candidate, label_path) if not found else None
        if not found and label in labels:
            return candidate, label, source, []
        problems.append(f"{source}: " + ("; ".join(found) or f"label {label!r} not allowed"))
    if not candidates:
        problems.append("no structured_output and no StructuredOutput tool call")
    return None, None, "", problems


def computed_usage(result: dict[str, Any], model_id: str) -> tuple[dict[str, Any], list[str],
                                                                    dict[str, dict[str, Any]]]:
    """Return ``(computed sums, served names, side models with their tokens)`` of one call.

    ``prompt_tokens`` = ``inputTokens + cacheReadInputTokens + cacheCreationInputTokens`` of the
    requested model (``inputTokens`` alone is only the uncached remainder; reading it alone
    under-reported prompts about 1,800 times in the kit project, D-42/D-45). Every other model
    in ``modelUsage`` is a side model, kept with its own unchanged entry.
    """
    model_usage = result.get("modelUsage") or {}
    served = served_names(model_usage, model_id)
    sums = {"prompt_tokens": 0, "input_tokens_uncached": 0, "cache_read_tokens": 0,
            "cache_creation_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
            "side_model_tokens": 0}
    side: dict[str, dict[str, Any]] = {}
    for name, entry in model_usage.items():
        if not isinstance(entry, dict):
            continue
        inp = int(entry.get("inputTokens") or 0)
        cr = int(entry.get("cacheReadInputTokens") or 0)
        cc = int(entry.get("cacheCreationInputTokens") or 0)
        out = int(entry.get("outputTokens") or 0)
        if name in served:
            sums["prompt_tokens"] += inp + cr + cc
            sums["input_tokens_uncached"] += inp
            sums["cache_read_tokens"] += cr
            sums["cache_creation_tokens"] += cc
            sums["output_tokens"] += out
            sums["thinking_tokens"] += int(entry.get("thinkingTokens") or 0)
        else:
            side[name] = dict(entry)
            sums["side_model_tokens"] += inp + cr + cc + out
    creation = (result.get("usage") or {}).get("cache_creation") or {}
    sums["cache_write_1h"] = int(creation.get("ephemeral_1h_input_tokens") or 0)
    sums["cache_write_5m"] = int(creation.get("ephemeral_5m_input_tokens") or 0)
    return sums, served, side


def cli_fields(parts: dict[str, Any]) -> dict[str, Any]:
    """Return the CLI's own fields of one call, unchanged (``total_cost_usd`` renamed only)."""
    result, init = parts["result"], parts["init"]
    fields = {
        "model_usage": result.get("modelUsage"),
        "usage": result.get("usage"),
        "num_turns": result.get("num_turns"),
        "duration_ms": result.get("duration_ms"),
        "duration_api_ms": result.get("duration_api_ms"),
        "stop_reason": result.get("stop_reason"),
        "subtype": result.get("subtype"),
        "is_error": result.get("is_error"),
        # the CLI's total_cost_usd: what the call would cost on the API at list price; the
        # subscription does not bill it (kit naming). Never a cost.
        "list_price_usd_not_billed": result.get("total_cost_usd"),
        "last_rate_limit_event": parts["rate_event"] or None,
        "init": {k: init.get(k) for k in ("model", "tools", "apiKeySource",
                                          "claude_code_version", "permissionMode")
                 if k in init} or None,
    }
    return fields


def kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the child's process group; SIGKILL it if it is alive after 5 s."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# --------------------------------------------------------------------------------------
# The backend
# --------------------------------------------------------------------------------------
class ClaudeCliBackend:
    """Claude Code CLI on the subscription: ``claude -p``, one subprocess per attempt."""

    name = "claude_cli_subscription"

    def __init__(
        self,
        binary: str | os.PathLike[str],
        cwd: Path,
        labels: tuple[str, ...],
        timeout_s: float = 300.0,
        max_attempts: int = 4,
        backoff_base_s: float = 10.0,
        allow_5m_cache_writes: bool = False,
    ) -> None:
        path = Path(binary)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CliBackendError(f"no executable claude binary at {path}")
        if max_attempts < 1:
            raise CliBackendError("max_attempts must be >= 1")
        self.binary = str(path.resolve())
        self.cwd = Path(cwd)
        self.cwd.mkdir(parents=True, exist_ok=True)
        if any(self.cwd.iterdir()):
            raise CliBackendError(f"the call folder {self.cwd} must be empty")
        self.labels = labels
        self.timeout_s = float(timeout_s)
        self.max_attempts = int(max_attempts)
        self.backoff_base_s = float(backoff_base_s)
        self.allow_5m_cache_writes = allow_5m_cache_writes
        self._inflight: set[subprocess.Popen] = set()
        self._inflight_lock = threading.Lock()
        self._killing = threading.Event()

    # ----------------------------------------------------------------------------------
    def version(self) -> str:
        """Return ``claude --version`` (e.g. ``2.1.322 (Claude Code)``), or ``""``."""
        try:
            done = subprocess.run([self.binary, "--version"], capture_output=True, text=True,
                                  timeout=60, env=child_env(), cwd=str(self.cwd), check=False)
            return done.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def describe(self) -> dict[str, Any]:
        """Return the backend's manifest entry: name, binary, its sha256, version, settings."""
        digest = hashlib.sha256()
        with open(self.binary, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return {
            "backend": self.name,
            "binary": self.binary,
            "binary_sha256": digest.hexdigest(),
            "version": self.version(),
            "base_flags": list(BASE_FLAGS),
            "forced_child_env": dict(FORCED_CHILD_ENV),
            "stripped_env_prefixes": list(STRIPPED_ENV_PREFIXES),
            "stripped_env_names": sorted(STRIPPED_ENV_NAMES),
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_attempts,
            "cwd": str(self.cwd),
        }

    def start(self, rows: list[dict[str, Any]]) -> None:
        """Nothing to restore from earlier rows on the subscription route."""

    def preflight(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        """``claude auth status --json`` (no model call). Fail closed [C21].

        Refuses when the output cannot be read, when not logged in, and when the login is not the
        claude.ai subscription (an API-key login bills per token). Keeps only the plan fields,
        never the email or organisation.
        """
        try:
            done = subprocess.run([self.binary, "auth", "status", "--json"], capture_output=True,
                                  text=True, timeout=60, env=child_env(), cwd=str(self.cwd),
                                  check=False)
            raw = done.stdout
            status = json.loads(raw[raw.find("{"):]) if "{" in raw else None
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise CliBackendError(f"cannot read `claude auth status --json`: {exc}") from exc
        if not isinstance(status, dict):
            raise CliBackendError("`claude auth status --json` printed no JSON object")
        if status.get("loggedIn") is not True:
            raise CliBackendError("the claude CLI is not logged in")
        if status.get("authMethod") != "claude.ai":
            raise CliBackendError(
                f"the CLI authenticates via {status.get('authMethod')!r}, not the claude.ai "
                "subscription; that route bills per token. Refusing."
            )
        if status.get("apiProvider") not in (None, "firstParty"):
            raise CliBackendError(f"apiProvider {status.get('apiProvider')!r} is not firstParty")
        return {k: status.get(k) for k in ("loggedIn", "authMethod", "apiProvider",
                                           "subscriptionType", "planTier", "loginExpiresAt")
                if k in status}

    def argv(self, request: CallRequest) -> list[str]:
        """Return the full command line of one call (the user turn is not in it)."""
        check_full_id(request.model_id)
        if request.effort not in EFFORTS:
            raise CliBackendError(f"effort {request.effort!r} not in {EFFORTS}")
        return [self.binary, *BASE_FLAGS, "--model", request.model_id,
                "--effort", request.effort, "--system-prompt", request.system,
                "--json-schema", request.schema_json]

    def kill_all(self) -> None:
        """Kill every call in flight and keep new ones from starting."""
        self._killing.set()
        with self._inflight_lock:
            procs = list(self._inflight)
        for proc in procs:
            kill_group(proc)

    # ----------------------------------------------------------------------------------
    def _run_once(self, argv: list[str], stdin_text: str) -> tuple[int | None, str, str, bool]:
        """Run ``claude`` once in its own process group: (exit code, stdout, stderr, timed out)."""
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", cwd=str(self.cwd), env=child_env(),
                                start_new_session=True)
        with self._inflight_lock:
            self._inflight.add(proc)
        if self._killing.is_set():      # a signal arrived between the check and Popen [C19]
            kill_group(proc)
        try:
            try:
                out, err = proc.communicate(stdin_text, timeout=self.timeout_s)
                return proc.returncode, out, err, False
            except subprocess.TimeoutExpired:
                kill_group(proc)
                try:
                    out, err = proc.communicate(timeout=10)      # [D13]
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, err = "", ""
                return None, out or "", err or "", True
        finally:
            with self._inflight_lock:
                self._inflight.discard(proc)

    def _backoff(self, attempt: int, stop: threading.Event) -> None:
        """Wait 10, 20, 40 ... s (capped at 300 s, +-20%), or less if the run stops."""
        delay = min(300.0, self.backoff_base_s * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
        stop.wait(delay)

    def call(self, request: CallRequest, stop: threading.Event) -> CallResult:
        """Make one item's call with its retries. Never raises for model or CLI errors."""
        argv = self.argv(request)
        log: list[dict[str, Any]] = []
        started_all = time.monotonic()
        timeouts = 0
        last: dict[str, Any] = {"status": "error", "error_kind": "not_called", "error": None}
        attempt = 0
        while attempt < self.max_attempts:
            if stop.is_set() or self._killing.is_set():
                break
            attempt += 1
            started = time.monotonic()
            try:
                code, out, err, timed_out = self._run_once(argv, request.user_text)
            except OSError as exc:
                return self._finish(CallResult("fatal", error=f"cannot run {self.binary}: {exc}",
                                               error_kind="binary"), log, started_all, attempt)
            elapsed = time.monotonic() - started
            parts = parse_stream(out)
            result = parts["result"]
            entry: dict[str, Any] = {"attempt": attempt, "exit_code": code,
                                     "elapsed_s": round(elapsed, 3), "timed_out": timed_out,
                                     "num_turns": result.get("num_turns"),
                                     "model_usage": result.get("modelUsage"),
                                     "rate_limit_info": parts["rate_info"] or None}
            log.append(entry)
            current = {"parts": parts, "out": out, "elapsed": elapsed}
            success_event = bool(result) and result.get("subtype") == "success" \
                and not result.get("is_error")
            if timed_out and not success_event:
                timeouts += 1
                entry["outcome"] = "timeout"
                last = {"status": "error", "error_kind": "timeout",
                        "error": f"timed out after {self.timeout_s:.0f} s", **current}
                if timeouts >= MAX_TIMEOUTS:
                    break
                if attempt < self.max_attempts:
                    self._backoff(attempt, stop)
                continue
            if not success_event:
                detail = str(result.get("result") or err.strip() or out.strip()[-400:]
                             or f"exit code {code}")[:600]
                kind = classify(f"{out[-4000:]}\n{err[-4000:]}", parts["rate_info"])
                entry["outcome"] = kind or "cli_error"
                entry["error"] = detail
                if kind == "auth":
                    return self._finish(CallResult("fatal", error=detail, error_kind="auth"),
                                        log, started_all, attempt, current)
                if kind == "exhausted":
                    return self._finish(CallResult("window", error=detail, error_kind="window",
                                                   stop_signal=f"window: {detail[:200]}"),
                                        log, started_all, attempt, current)
                if kind == "refusal":
                    res = CallResult("refusal", error=detail, error_kind="usage_policy")
                    return self._finish(res, log, started_all, attempt, current, request)
                last = {"status": "error", "error_kind": kind or "cli_error", "error": detail,
                        **current}
                if attempt < self.max_attempts:
                    self._backoff(attempt, stop)
                continue
            # ---- a success event: which model answered, then the answer
            if timed_out:
                entry["outcome_note"] = "timed out after a complete result; result kept"
            model_usage = result.get("modelUsage") or {}
            if not model_usage:
                entry["outcome"] = "no_model_usage"
                last = {"status": "error", "error_kind": "no_model_usage",
                        "error": "the result event has no modelUsage", **current}
                if attempt < self.max_attempts:
                    self._backoff(attempt, stop)
                continue
            sums, served, side = computed_usage(result, request.model_id)
            if not served:
                entry["outcome"] = "wrong_model"
                res = CallResult("fatal", error_kind="wrong_model",
                                 error=f"asked for {request.model_id}, modelUsage names "
                                       f"{sorted(model_usage)}: wrong model served")
                return self._finish(res, log, started_all, attempt, current, request)
            tools = [t if isinstance(t, str) else str(t)
                     for t in (parts["init"].get("tools") or [])]
            extra_tools = [t for t in tools if t not in ALLOWED_INIT_TOOLS]
            if extra_tools:                                         # [C7]
                entry["outcome"] = "unexpected_tools"
                res = CallResult("fatal", error_kind="unexpected_tools",
                                 error=f"the CLI offered the model tools {extra_tools}")
                return self._finish(res, log, started_all, attempt, current, request)
            key_source = parts["init"].get("apiKeySource")
            if key_source not in (None, "none"):                    # [C21]
                entry["outcome"] = "api_key_source"
                res = CallResult("fatal", error_kind="api_key_source",
                                 error=f"init.apiKeySource is {key_source!r}: not the subscription")
                return self._finish(res, log, started_all, attempt, current, request)
            requested_out = sums["output_tokens"]
            is_haiku = request.model_id.startswith("claude-haiku")
            fallback = [
                name for name, u in side.items()
                if int(u.get("outputTokens") or 0) > 0
                and not (name.startswith("claude-haiku") and not is_haiku and requested_out > 0)
            ]
            if fallback or result.get("stop_reason") == "refusal":
                entry["outcome"] = "fallback" if fallback else "stop_reason_refusal"
                why = (f"another model {fallback} wrote output (the CLI's fallback)" if fallback
                       else "stop_reason refusal")
                res = CallResult("refusal", error=why, error_kind=entry["outcome"])
                return self._finish(res, log, started_all, attempt, current, request)
            answer, label, source, problems = pick_answer(parts, request.schema,
                                                          request.label_path, self.labels)
            if answer is None:
                entry["outcome"] = "invalid_answer"
                entry["error"] = "; ".join(problems)[:600]
                last = {"status": "parse_failed", "error_kind": "invalid_answer",
                        "error": entry["error"], **current}
                if attempt < self.max_attempts:
                    self._backoff(attempt, stop)
                continue
            entry["outcome"] = "ok"
            res = CallResult("ok", label=label, parse_status="ok", answer=answer)
            res.raw_response["answer_source"] = source
            return self._finish(res, log, started_all, attempt, current, request)
        # ---- out of attempts (or stopping): the last attempt decides
        status = last["status"]
        res = CallResult(status, error=last.get("error"), error_kind=last.get("error_kind"))
        if status == "parse_failed":
            res.parse_status = "failed"
        if attempt == 0:
            res.error, res.error_kind = "the run stopped before the call", "stopped"
        return self._finish(res, log, started_all, attempt,
                            {k: last[k] for k in ("parts", "out", "elapsed") if k in last} or None,
                            request)

    def _finish(self, res: CallResult, log: list[dict[str, Any]], started_all: float,
                attempt: int, current: dict[str, Any] | None = None,
                request: CallRequest | None = None) -> CallResult:
        """Fill the shared fields of a result from the attempt that decided it."""
        res.attempts = attempt
        res.attempt_log = log
        res.elapsed_s_total = round(time.monotonic() - started_all, 3)
        if not current or "parts" not in current:
            return res
        parts = current["parts"]
        res.latency_s = round(current["elapsed"], 3)
        res.raw_stream = current["out"]
        res.cli = cli_fields(parts)
        res.window_utilization = utilization_of(parts["rate_info"])
        res.raw_response.update({
            "result_event": parts["result"] or None,
            "structured_output_blocks": parts["structured_blocks"],
        })
        if request is not None and parts["result"].get("modelUsage"):
            sums, served, side = computed_usage(parts["result"], request.model_id)
            res.computed_usage, res.served_models, res.side_models = sums, served, side
            if sums["cache_write_5m"] > 0 and not self.allow_5m_cache_writes and not res.stop_signal:
                res.stop_signal = ("overage: a 5-minute cache write; Claude Code drops to the "
                                   "5-minute TTL when usage credits pay (kit D-59)")
        signal_text = overage_signal(parts["rate_info"])
        if signal_text and not res.stop_signal:
            res.stop_signal = f"overage: {signal_text}"
        return res
