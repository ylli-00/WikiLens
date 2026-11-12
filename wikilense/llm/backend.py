"""The backend interface: one request in, one result out, retries included.

The runner (:mod:`wikilense.llm.runner`) knows nothing about how a model is reached. A backend
takes a :class:`CallRequest` and returns a :class:`CallResult`. Two backends implement the
:class:`Backend` protocol, chosen per run with ``--backend``:

- ``subscription``: :class:`wikilense.llm.cli_backend.ClaudeCliBackend`, the Claude Code CLI on
  the owner's Claude subscription;
- ``api``: :class:`wikilense.llm.api_backend.AnthropicApiBackend`, the Anthropic Messages API
  with the owner's API key, billed per token.

Everything else (claims, prompts, conditions, retrieval, resume key, rows, guards) is the same
for both.

Result statuses and what the runner does with them:

==============  =======  ======================================================================
status          final?   meaning
==============  =======  ======================================================================
ok              yes      a schema-valid answer from the requested model; ``label`` is set
parse_failed    yes      the model answered, but no schema-valid answer after the retries
refusal         yes      usage-policy block, or another model wrote the answer (fallback)
error           no       timeouts, CLI or transient errors after the retries: asked again
window          no       the usage window is spent: the run stops, the item is asked again
money           no       API only: the run's --max-usd cap or the account's spend limit / credit
fatal           no       auth problem, API-key login, wrong model served: the run stops
==============  =======  ======================================================================

A label is never guessed: only a schema-valid structured answer gives one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

FINAL_STATUSES = frozenset({"ok", "parse_failed", "refusal"})
STATUSES = frozenset({"ok", "parse_failed", "refusal", "error", "window", "money", "fatal"})


@dataclass(frozen=True)
class CallRequest:
    """Everything one call sends. ``schema_json`` is the compact JSON of ``schema``.

    ``user_text`` is the whole user turn. ``cache_prefix`` is its start up to a
    ``<<<CACHE_BREAK>>>`` line in the template (empty without one): the API backend marks it for
    the prompt cache; the subscription backend sends ``user_text`` unchanged (it has no marks).
    """

    model_id: str
    effort: str
    system: str
    user_text: str
    schema: dict[str, Any]
    schema_json: str
    label_path: tuple[str, ...]
    cache_prefix: str = ""


@dataclass
class CallResult:
    """What one item's call ended as, with the CLI's own fields kept unchanged.

    ``cli`` (subscription) holds the CLI's fields verbatim: ``model_usage`` (the result event's ``modelUsage``),
    ``usage``, ``num_turns``, ``duration_ms``, ``duration_api_ms``, ``stop_reason``,
    ``list_price_usd_not_billed`` (the CLI's ``total_cost_usd``: the API list price of the call,
    which the subscription does not bill; never a cost), ``last_rate_limit_event`` and ``init``.
    ``api`` (API backend) holds the API's own fields verbatim: ``usage``, ``model``,
    ``stop_reason``, ``stop_details``, ``request_id``, ``ratelimit`` (headers). ``cost_usd`` (API
    only) is a LIST-PRICE computation over every billed attempt, not the invoice.
    ``computed_usage`` holds sums computed here (names say so). ``attempt_log`` keeps one entry
    per attempt, with that attempt's own usage, so retried calls are not lost from the count.
    """

    status: str
    label: str | None = None
    parse_status: str | None = None          # "ok" | "failed" | None (no answer to parse)
    answer: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None
    served_models: list[str] = field(default_factory=list)
    side_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    cli: dict[str, Any] = field(default_factory=dict)
    api: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    computed_usage: dict[str, Any] = field(default_factory=dict)
    window_utilization: dict[str, float | None] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)
    raw_stream: str = ""
    attempts: int = 0
    attempt_log: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float | None = None
    elapsed_s_total: float = 0.0
    stop_signal: str | None = None           # why the whole run should stop after this call

    @property
    def final(self) -> bool:
        """Return True when the item is done and is not asked again."""
        return self.status in FINAL_STATUSES


class Backend(Protocol):
    """What the runner needs from a backend."""

    name: str

    def describe(self) -> dict[str, Any]:
        """Return what the manifest records about the backend (version, binary, auth)."""
        ...

    def preflight(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        """Check that calls can be made (login, key, billing route); raise on a refusal."""
        ...

    def start(self, rows: list[dict[str, Any]]) -> None:
        """Called once before the calls with every row already in the results file."""
        ...

    def call(self, request: CallRequest, stop: threading.Event) -> CallResult:
        """Make one item's call with its retries; never raise for model or CLI errors."""
        ...

    def kill_all(self) -> None:
        """Kill every call in flight (used on a signal)."""
        ...
