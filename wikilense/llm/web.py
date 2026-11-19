"""The ``web`` condition: the claim alone, with Claude Code's WebSearch and WebFetch tools.

The question: how does Claude Haiku 5.5 score when a normal user lets it use the internet? The
FEVEROUS labels are public on the web (fever.ai, Zenodo, GitHub copies,
Hugging Face), so WebFetch is denied on those hosts, and every web row keeps a trace of what the
model searched and opened for the audit (``scripts/llm_web_audit.py``). ``wikilense/llm/NOTES.md``
lists each choice with its source.

- :func:`web_flags`: the flags a ``web`` call adds to the subscription backend's argv. The three
  passage conditions keep their argv byte for byte (``cli_backend.ClaudeCliBackend.argv``).
- :func:`load_deny_hosts`: ``web_deny_hosts.txt``, one host pattern per line, each turned into
  one ``WebFetch(domain:<pattern>)`` deny rule. The file's sha256 goes into the manifest.
- :func:`web_trace`: each WebSearch / WebFetch call of one stream, in order, with what came back.
- :func:`audit_rows`: the audit flags of one run's rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

WEB_TOOLS: tuple[str, ...] = ("WebSearch", "WebFetch")
#: ``init.tools`` of a web call must be exactly this set; anything else is fatal.
WEB_INIT_TOOLS = frozenset({"StructuredOutput", *WEB_TOOLS})
DENY_HOSTS_FILE = Path(__file__).resolve().parent / "web_deny_hosts.txt"
#: Turn cap of a web call.
DEFAULT_MAX_TURNS = 20
#: Per call. WebFetch has its own deadline per fetch; a web call makes several tool calls.
DEFAULT_TIMEOUT_S = 200.0
#: Attempts of an item whose call reached --max-turns without an answer: one retry at most.
MAX_TURN_CAP_ATTEMPTS = 2
#: The assistant "model" of messages the CLI writes itself (errors), not a model's output.
SYNTHETIC_MODEL = "<synthetic>"
_HOST_RE = re.compile(r"^(\*\.)?[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")


class WebConfigError(ValueError):
    """A deny list or web setting that cannot be used."""


# --------------------------------------------------------------------------------------
# Deny list and flags
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DenyList:
    """The WebFetch deny list: host patterns in file order, the file's path and sha256."""

    hosts: tuple[str, ...]
    sha256: str
    path: str

    def rules(self) -> list[str]:
        """Return one ``WebFetch(domain:<pattern>)`` rule per host pattern."""
        return [f"WebFetch(domain:{host})" for host in self.hosts]


def load_deny_hosts(path: Path | None = None) -> DenyList:
    """Read the deny list: one lower-case host pattern per line (``host`` or ``*.host``); blank
    lines and ``#`` comments are skipped. Refuses an empty list, a duplicate and anything that is
    not a host pattern (a URL, a path, a port, a bare ``*``)."""
    path = Path(path or DENY_HOSTS_FILE)
    raw = path.read_bytes()
    hosts: list[str] = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if not _HOST_RE.match(text):
            raise WebConfigError(f"{path}, line {n}: {text!r} is not a lower-case host pattern")
        if text in hosts:
            raise WebConfigError(f"{path}, line {n}: {text!r} is listed twice")
        hosts.append(text)
    if not hosts:
        raise WebConfigError(f"{path} lists no host")
    return DenyList(tuple(hosts), hashlib.sha256(raw).hexdigest(), str(path))


@dataclass(frozen=True)
class WebConfig:
    """What a ``web`` call needs besides the request: deny list, turn cap, time limit."""

    deny: DenyList
    max_turns: int = DEFAULT_MAX_TURNS
    timeout_s: float = DEFAULT_TIMEOUT_S
    turn_cap_attempts: int = MAX_TURN_CAP_ATTEMPTS

    def __post_init__(self) -> None:
        if not 1 <= self.max_turns <= 200:
            raise WebConfigError(f"max_turns {self.max_turns} outside 1..200")
        if not 1 <= self.turn_cap_attempts <= 2:
            raise WebConfigError("a call that reached the turn cap is retried at most once")

    def flags(self) -> list[str]:
        """Return the flags a web call adds after the base flags (see :func:`web_flags`)."""
        return web_flags(self.deny, self.max_turns)

    def describe(self) -> dict[str, Any]:
        """Return the manifest's ``web`` section."""
        return {"tools": list(WEB_TOOLS), "flags": self.flags(), "deny_hosts_file": self.deny.path,
                "deny_hosts_sha256": self.deny.sha256, "deny_hosts": list(self.deny.hosts),
                "max_turns": self.max_turns, "timeout_s": self.timeout_s,
                "turn_cap_attempts": self.turn_cap_attempts,
                "init_tools_required": sorted(WEB_INIT_TOOLS)}


def web_flags(deny: DenyList, max_turns: int) -> list[str]:
    """Return the flags of a web call, every value its own argv item (never a bare flag).

    - ``--tools WebSearch,WebFetch`` replaces ``--tools none`` (the form ``--help`` shows);
    - ``--allowedTools WebSearch WebFetch(domain:*)``: in ``-p`` mode a tool call that needs
      approval is otherwise denied; ``domain:*`` matches every host (CLI 2.1.322's matcher);
    - ``--disallowedTools mcp__* WebFetch(domain:<host>) ...``: deny rules win over allow rules;
    - ``--permission-prompts none``: anything that would still prompt is denied automatically;
    - ``--max-turns N``: a hidden option of 2.1.322 (defined in the binary, not in ``--help``).
    """
    return ["--tools", ",".join(WEB_TOOLS),
            "--allowedTools", "WebSearch", "WebFetch(domain:*)",
            "--disallowedTools", "mcp__*", *deny.rules(),
            "--permission-prompts", "none",
            "--max-turns", str(int(max_turns))]


# --------------------------------------------------------------------------------------
# The stream: assistant models and the web trace
# --------------------------------------------------------------------------------------
def _events(stdout: str) -> list[dict[str, Any]]:
    """Return the JSON objects of a stream-json output, one per line, in order."""
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _message(event: dict[str, Any]) -> dict[str, Any]:
    """Return an event's ``message`` when it is an object (some events carry a string there)."""
    message = event.get("message")
    return message if isinstance(message, dict) else {}


def _text_of(content: Any) -> str:
    """Return the text of a tool_result's content (a string or a list of text blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text_of(b.get("text") if isinstance(b, dict) else b) for b in content)
    return "" if content is None else str(content)


def assistant_messages(stdout: str) -> list[dict[str, Any]]:
    """Return ``{model, tools, structured}`` per assistant message of the main loop: its
    ``message.model``, the tool names it called and whether it holds a StructuredOutput call.
    Messages of a subagent (``parent_tool_use_id`` set) are kept and marked ``subagent``: the
    web tools start none, and the model check would rather see such a message than miss it."""
    found = []
    for event in _events(stdout):
        if event.get("type") != "assistant":
            continue
        message = _message(event)
        content = message.get("content") if isinstance(message.get("content"), list) else []
        tools = [b.get("name") for b in content
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        found.append({"model": message.get("model"), "tools": tools,
                      "structured": "StructuredOutput" in tools,
                      "subagent": bool(event.get("parent_tool_use_id"))})
    return found


def foreign_messages(messages: list[dict[str, Any]], served: list[str]) -> list[str]:
    """Return the models of assistant messages not written by a served name of the request
    (the CLI's own ``<synthetic>`` messages excepted, and a missing model counts as foreign)."""
    return [str(m["model"]) for m in messages
            if m["model"] != SYNTHETIC_MODEL and m["model"] not in served]


def web_trace(stdout: str, result_event: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return every WebSearch and WebFetch call of one stream, in order.

    Each entry: ``tool``, ``tool_use_id``, ``model`` (of the message that called it), ``query``
    (WebSearch) or ``url`` and ``prompt`` (WebFetch), ``result_chars`` (length of the text that
    came back; None when no result came back), ``is_error``, ``denied`` (named in the result
    event's ``permission_denials``, or an error result that says the permission was denied),
    ``error`` (the first 300 characters of an error result), and for WebSearch ``result_urls``
    (from the structured ``tool_use_result`` when present, else the URLs in the text). For
    WebFetch, ``http_code`` and ``bytes`` when the structured result has them.
    """
    denied_ids = {d.get("tool_use_id") for d in (result_event or {}).get("permission_denials") or []
                  if isinstance(d, dict)}
    trace: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for event in _events(stdout):
        message = _message(event)
        content = message.get("content") if isinstance(message.get("content"), list) else []
        if event.get("type") == "assistant":
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_use"
                        and block.get("name") in WEB_TOOLS):
                    continue
                args = block.get("input") if isinstance(block.get("input"), dict) else {}
                entry: dict[str, Any] = {"tool": block["name"], "tool_use_id": block.get("id"),
                                         "model": message.get("model")}
                if block["name"] == "WebSearch":
                    entry["query"] = args.get("query")
                    for key in ("allowed_domains", "blocked_domains"):
                        if args.get(key):
                            entry[key] = args[key]
                else:
                    entry["url"] = args.get("url")
                    entry["prompt"] = args.get("prompt")
                entry.update(result_chars=None, is_error=None,
                             denied=block.get("id") in denied_ids, error=None)
                trace.append(entry)
                by_id[str(block.get("id"))] = entry
        elif event.get("type") == "user":
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                entry = by_id.get(str(block.get("tool_use_id")))
                if entry is None:
                    continue
                text = _text_of(block.get("content"))
                entry["result_chars"] = len(text)
                entry["is_error"] = bool(block.get("is_error"))
                if entry["is_error"]:
                    entry["error"] = text[:300]
                    low = text.lower()
                    if "denied" in low or "permission" in low:
                        entry["denied"] = True
                structured = event.get("tool_use_result")
                if entry["tool"] == "WebSearch":
                    entry["result_urls"] = _search_urls(structured, text)
                elif isinstance(structured, dict):
                    for key, name in (("code", "http_code"), ("bytes", "bytes"), ("url", "final_url")):
                        if key in structured:
                            entry[name] = structured[key]
    return trace


def _search_urls(structured: Any, text: str) -> list[str]:
    """Return the result URLs of one WebSearch call, in order, without duplicates."""
    urls: list[str] = []
    if isinstance(structured, dict) and isinstance(structured.get("results"), list):
        for item in structured["results"]:
            for link in (item.get("content") if isinstance(item, dict) else None) or []:
                if isinstance(link, dict) and isinstance(link.get("url"), str):
                    urls.append(link["url"])
    if not urls:
        urls = _URL_RE.findall(text)
    return list(dict.fromkeys(urls))


def trace_counts(trace: list[dict[str, Any]], model_usage: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``n_web_search``, ``n_web_fetch``, ``n_denied`` and ``web_search_requests`` (the
    sum of ``webSearchRequests`` over every model in ``modelUsage``; None when no entry has it)."""
    usage = [u for u in (model_usage or {}).values() if isinstance(u, dict)]
    requests = [int(u["webSearchRequests"]) for u in usage
                if isinstance(u.get("webSearchRequests"), (int, float))]
    return {"n_web_search": sum(1 for t in trace if t["tool"] == "WebSearch"),
            "n_web_fetch": sum(1 for t in trace if t["tool"] == "WebFetch"),
            "n_denied": sum(1 for t in trace if t["denied"]),
            "web_search_requests": sum(requests) if requests else None}


# --------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------
#: What flags a row, by name: a host name only as a whole host (``hf.co`` not in ``shf.com``).
AUDIT_PATTERNS: dict[str, re.Pattern[str]] = {
    "fever.ai": re.compile(r"(?<![a-z0-9-])fever\.ai(?![a-z0-9-])", re.IGNORECASE),
    "feverous": re.compile(r"feverous", re.IGNORECASE),
    "zenodo": re.compile(r"zenodo", re.IGNORECASE),
    "huggingface": re.compile(r"hugging\s?face", re.IGNORECASE),
    "hf.co": re.compile(r"(?<![a-z0-9.-])hf\.co(?![a-z0-9-])", re.IGNORECASE),
    "raw.githubusercontent.com": re.compile(r"raw\.githubusercontent\.com", re.IGNORECASE),
}
_VERDICT_RE = re.compile(r"verdict\W{0,5}(supports|refutes|not enough info)", re.IGNORECASE)
_GITHUB_DATA_RE = re.compile(r"github[^\s\"']*\.(jsonl|json|csv)\b", re.IGNORECASE)


def host_of(url: str | None) -> str | None:
    """Return the lower-case host of a URL, or None."""
    if not url:
        return None
    try:
        host = urlsplit(url if "://" in url else f"https://{url}").hostname
    except ValueError:
        return None
    return host.lower().rstrip(".") if host else None


def host_matches(pattern: str, host: str) -> bool:
    """Return True when ``host`` falls under a deny pattern (``x`` exactly; ``*.x`` any
    subdomain of x, not x itself), as the CLI's WebFetch domain rules match."""
    if pattern.startswith("*."):
        return host.endswith("." + pattern[2:])
    return host == pattern


def _texts_of_row(row: dict[str, Any], raw_text: str | None) -> list[tuple[str, str]]:
    """Return (where, text) pairs of a web row to search: queries, URLs, prompts, result URLs,
    errors, and the tool results of the raw stream when it is given."""
    texts: list[tuple[str, str]] = []
    for n, entry in enumerate(row.get("web_trace") or []):
        for key in ("query", "url", "prompt", "error", "final_url"):
            if entry.get(key):
                texts.append((f"trace[{n}].{key}", str(entry[key])))
        for url in entry.get("result_urls") or []:
            texts.append((f"trace[{n}].result_urls", url))
    if raw_text:
        texts.append(("raw tool results", raw_text))
    return texts


def tool_result_text(stdout: str) -> str:
    """Return the text of every tool result of a stream, joined (what the model read)."""
    parts = []
    for event in _events(stdout):
        if event.get("type") != "user":
            continue
        content = _message(event).get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(_text_of(block.get("content")))
    return "\n".join(parts)


@dataclass
class AuditResult:
    """Hosts fetched, and the flagged rows with their reasons."""

    hosts: Counter = field(default_factory=Counter)
    search_hosts: Counter = field(default_factory=Counter)
    flagged: list[dict[str, Any]] = field(default_factory=list)
    n_rows: int = 0


def audit_row(row: dict[str, Any], deny: DenyList, raw_text: str | None = None) -> list[str]:
    """Return the reasons a web row is flagged (empty when it is not)."""
    reasons: list[str] = []
    for where, text in _texts_of_row(row, raw_text):
        for word, pattern in AUDIT_PATTERNS.items():
            if pattern.search(text):
                reasons.append(f"{where} mentions {word!r}")
        if _GITHUB_DATA_RE.search(text):
            reasons.append(f"{where} names a GitHub data file (.jsonl/.json/.csv)")
        if _VERDICT_RE.search(text):
            reasons.append(f"{where} has 'Verdict' next to a label")
        for url in _URL_RE.findall(text) if where != "raw tool results" else []:
            host = host_of(url)
            if host and any(host_matches(p, host) for p in deny.hosts):
                reasons.append(f"{where} names a denied host {host}")
    return list(dict.fromkeys(reasons))


def audit_rows(rows: list[dict[str, Any]], deny: DenyList,
               raw_texts: dict[str, str] | None = None) -> AuditResult:
    """Audit every web row; ``raw_texts`` maps a row's ``raw_stream_file`` to its tool-result
    text. Flagged rows are reported, never removed."""
    result = AuditResult()
    for row in rows:
        if row.get("condition") != "web":
            continue
        result.n_rows += 1
        for entry in row.get("web_trace") or []:
            if entry.get("tool") == "WebFetch":
                result.hosts[host_of(entry.get("url")) or "?"] += 1
            for url in entry.get("result_urls") or []:
                result.search_hosts[host_of(url) or "?"] += 1
        raw = (raw_texts or {}).get(row.get("raw_stream_file") or "")
        reasons = audit_row(row, deny, raw)
        if reasons:
            result.flagged.append({"claim_id": row.get("claim_id"), "key": row.get("key"),
                                   "status": row.get("status"), "label": row.get("label"),
                                   "gold_label": row.get("gold_label"), "reasons": reasons})
    return result
