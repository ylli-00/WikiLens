"""The API backend: one Anthropic Messages API request per item, with an API key.

``wikilense/llm/NOTES.md`` maps every choice to its source.

- **The key** comes from the repository's ``.env`` through python-dotenv's ``dotenv_values``,
  the way ``wikilense.config`` reads ``.env``: never from ``os.environ``. It is passed to the SDK
  client explicitly and is never printed, logged or written anywhere (rows, raw files, manifest).
  ``ANTHROPIC_BASE_URL`` or ``ANTHROPIC_AUTH_TOKEN`` in the environment would silently change
  what the SDK does, so the backend refuses to start with either set.
- **The request**: ``client.messages.stream(model, max_tokens, system, messages, thinking,
  output_config)``. ``system`` is one text block with a cache mark; with a ``<<<CACHE_BREAK>>>``
  prefix the user turn is two text blocks and the first carries a mark. ``output_config`` holds
  the effort and the structured-output schema (``format: json_schema``). Never sent:
  ``temperature``/``top_p``/``top_k``, ``tools``/``tool_choice`` (forced tool choice is a 400 on
  Sonnet 5.5 and Opus 5.5) and ``fallbacks``. A fallback model's answer would be counted as the
  requested model's and spoil the comparison between models, so another model answering is
  recorded as a refusal.
- **Errors**, in this order: 401/403/404/402 fatal; 400 = the account's spend limit or empty
  credit (``money``) or a bad request (error); 429 = the account's spend cap (``money``) or a
  per-minute limit (one gate shared by every worker, its own budget of 20); 5xx / overloaded =
  back off and retry; timeout = at most 2; network = back off.
- **Money**: ``--max-usd`` is required. Before every request the backend adds what this run spent
  and what every earlier row of the same results file cost; at the cap nothing more is sent.
  ``cost_usd`` is a LIST-PRICE computation (:data:`PRICES`), not the invoice.
- **Answer**: the reply's text blocks parsed as one JSON object and checked against the schema;
  thinking blocks are never read. Invalid answers are asked again up to ``max_attempts`` and end
  as ``parse_failed``; ``stop_reason: max_tokens`` is an error (truncated, asked again on the next
  run); ``stop_reason: refusal`` is a refusal with its ``stop_details`` category.
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wikilense.llm.backend import CallRequest, CallResult
from wikilense.llm.cli_backend import EFFORTS, dig, validate
from wikilense.llm.models import check_full_id

log = logging.getLogger("wikilense.llm.api")

API_BASE_URL = "https://api.anthropic.com"
PLACEHOLDER_KEYS = frozenset({"sk-ant-REPLACE_ME", "<the key>", ""})
#: Variables the SDK reads by itself even with an explicit key; refused.
REFUSED_ENV = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")


@dataclass(frozen=True)
class Price:
    """US$ per million tokens: input, output, cache read, 5-minute write, 1-hour write."""

    input: float
    output: float
    cache_read: float
    write_5m: float
    write_1h: float


#: List prices. Source: https://platform.claude.com/docs/en/about-claude/pricing, fetched
#: 2026-11-12 ("Model pricing" table). Haiku 5.5 is priced by prompt length: a request whose
#: prompt (all input tokens, cache reads and writes included) is over 100,000 tokens pays the
#: second set (same page, "Long context pricing").
PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.10, 2.50, 4.00),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00, 8.00),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125, 0.20),
}
LONG_PROMPT_PRICES: dict[str, tuple[int, Price]] = {
    "claude-haiku-5-5": (100_000, Price(0.50, 2.50, 0.05, 0.625, 1.00)),
}
#: "specifying US-only inference through the inference_geo parameter incurs a 1.1x multiplier"
#: (same page). Never requested here; applied if a response reports it.
US_GEO_MULTIPLIER = 1.1
PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-11-12"

#: max_tokens by effort; it caps thinking PLUS the answer.
MAX_TOKENS_BY_EFFORT = {"low": 4_096, "medium": 16_384, "high": 32_768, "xhigh": 65_536,
                        "max": 128_000}
CACHE_MARKS = {"5m": {"type": "ephemeral"}, "1h": {"type": "ephemeral", "ttl": "1h"}}
UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "pattern"})
RATE_LIMIT_MAX_ATTEMPTS = 20
RATE_LIMIT_DEFAULT_WAIT_S = 15.0
RATE_LIMIT_MAX_WAIT_S = 120.0
RATELIMIT_KINDS = ("requests", "tokens", "input-tokens", "output-tokens")
RETRYABLE_ERROR_TYPES = frozenset({"overloaded_error", "api_error"})
SPEND_LIMIT_CODE = "enforced_spend_limit_reached"
SPEND_LIMIT_MESSAGE = re.compile(
    r"reached your (?:specified )?(?:workspace )?api usage limits|credit balance is too low",
    re.IGNORECASE)
SNAPSHOT_SUFFIX = re.compile(r"-\d{8}")
MAX_TIMEOUT_ATTEMPTS = 2
NOISY_LOGGERS = ("anthropic", "httpx", "httpcore")


class ApiBackendError(RuntimeError):
    """The API backend cannot be used as configured (no SDK, no key, a refused variable)."""


class TimeLimit(Exception):
    """The answer was still streaming at the time limit; carries the usage billed so far."""

    def __init__(self, usage: dict[str, Any] | None) -> None:
        super().__init__("time limit")
        self.usage = usage or {}


# --------------------------------------------------------------------------------------
# The key and the environment
# --------------------------------------------------------------------------------------
def load_api_key(env_file: Path) -> str:
    """Return ``ANTHROPIC_API_KEY`` from ``env_file`` (dotenv_values), never from os.environ.

    Refuses a missing file or line, an empty value and placeholders. The error messages never
    contain the value.
    """
    from dotenv import dotenv_values

    if not env_file.is_file():
        raise ApiBackendError(f"no {env_file}: add ANTHROPIC_API_KEY=<the key> to the repo's .env")
    value = dotenv_values(env_file).get("ANTHROPIC_API_KEY")
    key = (value or "").strip()
    if key in PLACEHOLDER_KEYS:
        raise ApiBackendError(f"ANTHROPIC_API_KEY in {env_file} is missing, empty or a "
                              "placeholder")
    if not key.startswith("sk-ant-"):
        raise ApiBackendError(f"ANTHROPIC_API_KEY in {env_file} does not start with sk-ant-")
    return key


def check_environment(environ: dict[str, str] | None = None) -> list[str]:
    """Raise when the SDK would be redirected; return warnings (names only, never values)."""
    env = os.environ if environ is None else environ
    refused = [name for name in REFUSED_ENV if env.get(name)]
    if refused:
        raise ApiBackendError(f"{refused} set in the environment would change what the SDK "
                              "sends: unset them for the API backend")
    warnings = []
    if "ANTHROPIC_API_KEY" in env:
        warnings.append("ANTHROPIC_API_KEY is set in the shell environment: the API backend "
                        "ignores it (it reads .env), but remove it - the Claude Code CLI could "
                        "bill it instead of the subscription")
    return warnings


def quiet_sdk_loggers() -> None:
    """Keep the SDK's and httpx's debug logging off, whatever the runner's level."""
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# --------------------------------------------------------------------------------------
# Pure helpers (tested directly)
# --------------------------------------------------------------------------------------
def _num(value: Any) -> int:
    """A token count, or 0."""
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def token_split(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Return the computed token split of one response's ``usage``.

    ``input_tokens`` is the UNCACHED remainder only; cache reads and writes are separate, and
    writes split into 5-minute and 1-hour parts. ``output_tokens`` includes thinking.
    """
    u = usage or {}
    split = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
    written = _num(u.get("cache_creation_input_tokens"))
    w1h = _num(split.get("ephemeral_1h_input_tokens"))
    w5m = _num(split.get("ephemeral_5m_input_tokens")) if split else written
    uncached = _num(u.get("input_tokens"))
    read = _num(u.get("cache_read_input_tokens"))
    details = u.get("output_tokens_details") if isinstance(u.get("output_tokens_details"), dict) else {}
    return {
        "prompt_tokens": uncached + read + w5m + w1h,
        "input_tokens_uncached": uncached,
        "cache_read_tokens": read,
        "cache_creation_tokens": w5m + w1h,
        "cache_write_5m": w5m,
        "cache_write_1h": w1h,
        "output_tokens": _num(u.get("output_tokens")),
        "thinking_tokens": _num(details.get("thinking_tokens")),
        "side_model_tokens": 0,
    }


def price_for(model_id: str, prompt_tokens: int) -> Price:
    """Return the list price of a model for a request of ``prompt_tokens``; raise if unknown."""
    if model_id not in PRICES:
        raise ApiBackendError(f"no list price for {model_id}: add it to PRICES with its source")
    threshold, long_price = LONG_PROMPT_PRICES.get(model_id, (None, None))
    if threshold is not None and prompt_tokens > threshold:
        return long_price
    return PRICES[model_id]


def cost_of(usage: dict[str, Any] | None, model_id: str) -> tuple[float, dict[str, Any]]:
    """Return (US$ at list price, computed token split) of one response's ``usage``."""
    tokens = token_split(usage)
    price = price_for(model_id, tokens["prompt_tokens"])
    usd = (tokens["input_tokens_uncached"] * price.input
           + tokens["cache_read_tokens"] * price.cache_read
           + tokens["cache_write_5m"] * price.write_5m
           + tokens["cache_write_1h"] * price.write_1h
           + tokens["output_tokens"] * price.output) / 1e6
    if (usage or {}).get("inference_geo") == "us":
        usd *= US_GEO_MULTIPLIER
    return round(usd, 8), tokens


def sendable_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return (the schema as sent in output_config.format, the paths of removed keywords).

    A deep copy with the keywords the structured-output grammar does not support removed
    (minItems 0/1 kept) and ``additionalProperties: false`` added to objects lacking it; an
    object that allows extra properties is refused.
    """
    stripped: list[str] = []

    def walk(node: Any, path: str) -> Any:
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in UNSUPPORTED_SCHEMA_KEYWORDS:
                if key == "minItems" and value in (0, 1) and not isinstance(value, bool):
                    out[key] = value
                else:
                    stripped.append(f"{path}.{key}")
                continue
            if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                out[key] = {n: walk(s, f"{path}.{key}.{n}") for n, s in value.items()}
            elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
                out[key] = [walk(s, f"{path}.{key}[{i}]") for i, s in enumerate(value)]
            elif key == "items":
                out[key] = walk(value, f"{path}.items")
            else:
                out[key] = value
        if out.get("type") == "object" or "properties" in out:
            if "additionalProperties" not in out:
                out["additionalProperties"] = False
            elif out["additionalProperties"] is not False:
                raise ApiBackendError(f"structured output needs additionalProperties: false; "
                                      f"{path} allows extra properties")
        return out

    return walk(copy.deepcopy(schema), "$"), stripped


def same_model(served: str, requested: str) -> bool:
    """Exact, or the requested id plus a dated snapshot suffix (-YYYYMMDD); never a prefix."""
    if served == requested:
        return True
    return (served.startswith(requested)
            and SNAPSHOT_SUFFIX.fullmatch(served[len(requested):]) is not None)


def served_by_other_model(body: dict[str, Any], model_id: str) -> str | None:
    """Return the model that wrote the reply when it is not ``model_id``, else None.

    Defensive (no fallback is requested): the top-level model, a ``fallback_message`` entry in
    ``usage.iterations`` and a ``fallback`` content block are all checked.
    """
    served = body.get("model")
    if isinstance(served, str) and served and not same_model(served, model_id):
        return served
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    for entry in usage.get("iterations") or []:
        if isinstance(entry, dict) and entry.get("type") == "fallback_message":
            return entry.get("model") or "a fallback model"
    for block in body.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "fallback":
            target = block.get("to")
            return (target.get("model") if isinstance(target, dict) else None) or "a fallback model"
    return None


def json_from_text(content: list[Any]) -> dict[str, Any] | None:
    """The reply's text blocks joined, as ONE JSON object, or None (thinking is never read)."""
    joined = "".join(b.get("text", "") for b in content or []
                     if isinstance(b, dict) and b.get("type") == "text").strip()
    if not joined:
        return None
    try:
        payload = json.loads(joined)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_rfc3339(value: str) -> datetime | None:
    """An RFC 3339 timestamp as an aware datetime, or None."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def rate_limit_wait(headers: Any, now: datetime | None = None) -> float | None:
    """Seconds a 429 says to wait: retry-after, else the anthropic-ratelimit-*-reset headers
    (latest among limits at 0, else the soonest), clamped to [1, 120] s; None if they say nothing.
    """
    if headers is None:
        return None

    def clamp(seconds: float) -> float:
        return min(max(seconds, 1.0), RATE_LIMIT_MAX_WAIT_S)

    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            seconds = float(retry_after)
        except (TypeError, ValueError):
            seconds = math.nan
        if math.isfinite(seconds):
            return clamp(seconds)
    now = now or datetime.now(timezone.utc)
    spent, future = [], []
    for kind in RATELIMIT_KINDS:
        reset = headers.get(f"anthropic-ratelimit-{kind}-reset")
        when = _parse_rfc3339(reset) if isinstance(reset, str) else None
        if when is None:
            continue
        seconds = (when - now).total_seconds()
        if seconds <= 0:
            continue
        future.append(seconds)
        remaining = headers.get(f"anthropic-ratelimit-{kind}-remaining")
        if isinstance(remaining, str) and remaining.strip() == "0":
            spent.append(seconds)
    if spent:
        return clamp(max(spent))
    if future:
        return clamp(min(future))
    return None


def ratelimit_snapshot(headers: Any) -> dict[str, Any]:
    """The anthropic-ratelimit-*-limit / -remaining headers of a response, as ints."""
    out: dict[str, Any] = {}
    if headers is None:
        return out
    for kind in ("requests", "input-tokens", "output-tokens", "tokens"):
        for part in ("limit", "remaining"):
            value = headers.get(f"anthropic-ratelimit-{kind}-{part}")
            if value is not None:
                try:
                    out[f"{kind}_{part}"] = int(value)
                except (TypeError, ValueError):
                    out[f"{kind}_{part}"] = value
    return out


def error_text(exc: Any) -> str:
    """The API's own error message, else the exception's text. Never request headers."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message if isinstance(message, str) and message else str(getattr(exc, "message", exc))


def account_limit_reason(exc: Any) -> str | None:
    """The message when ``exc`` is the ACCOUNT's spend limit or empty credit, else None."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    code = details.get("error_code") if isinstance(details, dict) else None
    text = error_text(exc)
    if code == SPEND_LIMIT_CODE or SPEND_LIMIT_MESSAGE.search(text):
        return text
    return None


class RateGate:
    """One pause shared by every worker: a 429 seen by one closes it for the wait it names."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open_at = 0.0
        self.closures = 0

    def close_for(self, seconds: float) -> None:
        """Close the gate for ``seconds`` from now (never shortens a longer closure)."""
        with self._lock:
            target = time.monotonic() + max(0.0, seconds)
            if target > self._open_at:
                self._open_at = target
                self.closures += 1

    def wait(self, stop: threading.Event) -> None:
        """Block until the gate is open (or the run stops), with up to 1 s of jitter."""
        with self._lock:
            delay = self._open_at - time.monotonic()
        if delay > 0:
            stop.wait(delay + random.uniform(0.0, 1.0))


class SpendGuard:
    """The run's US$ cap: earlier rows of the results file plus this run, checked before every
    request. Calls in flight finish and bill, so a run can end up to (workers x one call) over."""

    def __init__(self, cap_usd: float) -> None:
        self.cap = float(cap_usd)
        self.already = 0.0
        self.here = 0.0
        self._lock = threading.Lock()

    def add(self, usd: float) -> None:
        with self._lock:
            self.here += max(0.0, usd)

    @property
    def spent(self) -> float:
        with self._lock:
            return self.already + self.here

    def over(self) -> bool:
        return self.spent >= self.cap


def http_error_classes(anthropic_mod: Any) -> tuple[tuple[type, ...], tuple[type, ...]]:
    """(timeout classes, network classes): the SDK's and its HTTP library's."""
    timeouts: list[type] = [anthropic_mod.APITimeoutError]
    network: list[type] = [anthropic_mod.APIConnectionError]
    for name in ("httpx2", "httpx"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        timeouts.append(mod.TimeoutException)
        network.extend([mod.TransportError, mod.StreamError])
    return tuple(timeouts), tuple(network)


# --------------------------------------------------------------------------------------
# The backend
# --------------------------------------------------------------------------------------
class AnthropicApiBackend:
    """The Anthropic Messages API with an API key, billed per token (list price in cost_usd)."""

    name = "anthropic_api"

    def __init__(
        self,
        env_file: Path,
        labels: tuple[str, ...],
        max_usd: float,
        timeout_s: float = 480.0,
        max_attempts: int = 4,
        backoff_base_s: float = 10.0,
        cache_ttl: str = "5m",
        thinking_display: str = "omitted",
        max_tokens: int | None = None,
        client_factory: Any = None,
        api_key: str | None = None,
        rate_limit_attempts: int = RATE_LIMIT_MAX_ATTEMPTS,
    ) -> None:
        """``api_key`` is used instead of reading ``env_file`` (the web page's fact-check passes
        the key it resolved; the runner never does). It is kept only on the instance.
        ``rate_limit_attempts`` is the number of 429 answers after which one call gives up;
        each earlier one is waited out and the request sent again (the web page's fact-check
        takes 1: the first 429 ends the call)."""
        if max_usd is None or not max_usd > 0:
            raise ApiBackendError("--max-usd is required for the API backend (a positive US$ cap)")
        if cache_ttl not in ("5m", "1h", "off"):
            raise ApiBackendError(f"cache_ttl {cache_ttl!r} not in 5m, 1h, off")
        if thinking_display not in ("omitted", "summarized"):
            raise ApiBackendError(f"thinking_display {thinking_display!r}")
        self.env_file = Path(env_file)
        self.labels = labels
        self.guard = SpendGuard(max_usd)
        self.timeout_s = float(timeout_s)
        self.max_attempts = int(max_attempts)
        self.backoff_base_s = float(backoff_base_s)
        self.cache_ttl = cache_ttl
        self.thinking_display = thinking_display
        self.max_tokens = max_tokens
        self.gate = RateGate()
        self._client_factory = client_factory
        self._key: str | None = api_key
        self.rate_limit_attempts = int(rate_limit_attempts)
        self._local = threading.local()
        self._active: set[Any] = set()
        self._active_lock = threading.Lock()
        self._killing = threading.Event()
        self._anthropic: Any = None

    # ----------------------------------------------------------------------------------
    def _sdk(self) -> Any:
        """Import the SDK (lazily: dry runs and the subscription route need none)."""
        if self._anthropic is None:
            try:
                self._anthropic = importlib.import_module("anthropic")
            except ImportError as exc:
                raise ApiBackendError("the anthropic package is not installed in this "
                                      "environment (pip install anthropic==0.121.0)") from exc
        return self._anthropic

    def sdk_version(self) -> str | None:
        """The installed SDK's version, or None when it is not installed."""
        try:
            return str(self._sdk().__version__)
        except ApiBackendError:
            return None

    def max_tokens_for(self, effort: str) -> int:
        """``max_tokens`` for an effort level (thinking plus the answer)."""
        return self.max_tokens or MAX_TOKENS_BY_EFFORT[effort]

    def describe(self) -> dict[str, Any]:
        """The manifest entry. Nothing about the key, not even whether it was read."""
        version = self.sdk_version()
        return {
            "backend": self.name,
            "version": f"anthropic {version}" if version else None,
            "sdk_version": version,
            "base_url": API_BASE_URL,
            "prices_usd_per_mtok": {m: vars(p) for m, p in PRICES.items()},
            "long_prompt_prices_usd_per_mtok": {m: {"over_prompt_tokens": t, **vars(p)}
                                                for m, (t, p) in LONG_PROMPT_PRICES.items()},
            "price_source": PRICE_SOURCE,
            "cost_usd_note": "list-price computation from each response's usage, not the invoice",
            "max_usd": self.guard.cap,
            "cache_ttl": self.cache_ttl,
            "thinking": {"type": "adaptive", "display": self.thinking_display},
            "max_tokens": self.max_tokens or dict(MAX_TOKENS_BY_EFFORT),
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_attempts,
            "fallbacks": "never sent: another model's answer is recorded as a refusal",
        }

    def start(self, rows: list[dict[str, Any]]) -> None:
        """Count what earlier rows of the same results file already cost against --max-usd."""
        self.guard.already = sum(float(r.get("cost_usd") or 0.0) for r in rows)
        log.info("earlier rows of this results file cost $%.4f (list price) of the $%.2f cap",
                 self.guard.already, self.guard.cap)

    def preflight(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        """Check the environment, the SDK and the key; ask the free Models API for each model.

        ``models.retrieve`` is not billed. A 401 (bad key) or 404 (unknown id) is raised here,
        before any billed call. Returns facts for the manifest, never the key.
        """
        for warning in check_environment():
            log.warning("%s", warning)
        self._sdk()
        self._key = load_api_key(self.env_file)
        quiet_sdk_loggers()
        found: dict[str, Any] = {}
        sdk = self._sdk()
        for model_id in model_ids or []:
            check_full_id(model_id)
            try:
                info = self.client().models.retrieve(model_id)
                found[model_id] = getattr(info, "id", None)
            except sdk.AuthenticationError as exc:
                raise ApiBackendError(f"401: the API key in .env is invalid or revoked: "
                                      f"{error_text(exc)[:200]}") from None
            except sdk.NotFoundError:
                raise ApiBackendError(f"404: model {model_id!r} not found for this key") from None
        return {"key_source": "repo .env (dotenv_values)", "models_api": found}

    def client(self) -> Any:
        """This thread's SDK client: explicit key and base URL, SDK retries off."""
        client = getattr(self._local, "client", None)
        if client is None:
            if self._key is None:
                self._key = load_api_key(self.env_file)
            if self._client_factory is not None:
                client = self._client_factory(self._key, self.timeout_s + 30.0)
            else:
                client = self._sdk().Anthropic(api_key=self._key, base_url=API_BASE_URL,
                                               max_retries=0, timeout=self.timeout_s + 30.0)
            self._local.client = client
        return client

    def request_kwargs(self, request: CallRequest) -> dict[str, Any]:
        """The ``messages.stream`` keyword arguments of one request (no key in them)."""
        check_full_id(request.model_id)
        if request.effort not in EFFORTS:
            raise ApiBackendError(f"effort {request.effort!r} not in {EFFORTS}")
        mark = CACHE_MARKS.get(self.cache_ttl)
        system: list[dict[str, Any]] = [{"type": "text", "text": request.system}]
        if mark:
            system[0]["cache_control"] = dict(mark)
        prefix = request.cache_prefix
        if prefix and mark and request.user_text.startswith(prefix):
            content: Any = [{"type": "text", "text": prefix, "cache_control": dict(mark)}]
            tail = request.user_text[len(prefix):]
            if tail:
                content.append({"type": "text", "text": tail})
        else:
            content = request.user_text
        schema_sent, _ = sendable_schema(request.schema)
        return {
            "model": request.model_id,
            "max_tokens": self.max_tokens_for(request.effort),
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive", "display": self.thinking_display},
            "output_config": {"effort": request.effort,
                              "format": {"type": "json_schema", "schema": schema_sent}},
        }

    def kill_all(self) -> None:
        """Close every open stream and keep new requests from starting."""
        self._killing.set()
        with self._active_lock:
            streams = list(self._active)
        for stream in streams:
            try:
                stream.close()
            except Exception:
                log.debug("closing a stream failed", exc_info=True)

    # ----------------------------------------------------------------------------------
    def _stream(self, kwargs: dict[str, Any]) -> tuple[Any, Any, str | None]:
        """One streamed request read to its final message within the time limit.

        A watchdog CLOSES the stream at the deadline (a read timeout fires only after silence,
        and the API keeps streams alive with pings). Raises TimeLimit with the usage billed so
        far when it cut the stream.
        """
        limit = self.timeout_s
        started = time.monotonic()
        with self.client().messages.stream(**kwargs) as stream:
            with self._active_lock:
                self._active.add(stream)
            try:
                fired = threading.Event()

                def cut() -> None:
                    fired.set()
                    try:
                        stream.close()
                    except Exception:
                        log.debug("watchdog close failed", exc_info=True)

                timer = threading.Timer(limit, cut)
                timer.daemon = True
                timer.start()
                finished = False
                try:
                    for _event in stream:
                        if time.monotonic() - started >= limit:
                            break
                    else:
                        finished = True
                except Exception:
                    if not fired.is_set() and time.monotonic() - started < limit:
                        raise
                finally:
                    timer.cancel()
                headers = getattr(getattr(stream, "response", None), "headers", None)
                rid = getattr(stream, "request_id", None)
                if finished:
                    try:
                        message = stream.get_final_message()
                    except Exception:  # noqa: BLE001 - closed before a message existed
                        message = None
                    if message is not None and getattr(message, "stop_reason", None):
                        return message, headers, rid
                usage = None
                try:
                    dump = stream.current_message_snapshot.model_dump(mode="json")
                    usage = dump.get("usage") if isinstance(dump, dict) else None
                except Exception:  # noqa: BLE001 - nothing received yet
                    usage = None
                raise TimeLimit(usage)
            finally:
                with self._active_lock:
                    self._active.discard(stream)

    def _backoff(self, attempt: int, stop: threading.Event) -> None:
        """10, 20, 40 ... s (capped at 300) +-20%, or less if the run stops."""
        stop.wait(min(300.0, self.backoff_base_s * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2))

    def call(self, request: CallRequest, stop: threading.Event) -> CallResult:
        """One item's request with the retry policy. Never raises for API errors."""
        sdk = self._sdk()
        timeout_classes, network_classes = http_error_classes(sdk)
        kwargs = self.request_kwargs(request)
        log_entries: list[dict[str, Any]] = []
        started_all = time.monotonic()
        total_cost = 0.0
        last: dict[str, Any] = {"status": "error", "error_kind": "not_called", "error": None}
        attempt = rate_limited = timeouts = 0
        keep: dict[str, Any] = {}

        def finish(res: CallResult) -> CallResult:
            res.attempts = sum(1 for e in log_entries if e.get("sent"))
            res.attempt_log = log_entries
            res.cost_usd = round(total_cost, 8) if res.attempts else None
            res.elapsed_s_total = round(time.monotonic() - started_all, 3)
            for name, value in keep.items():
                setattr(res, name, value)
            return res

        while attempt < self.max_attempts:
            if stop.is_set() or self._killing.is_set():
                break
            if self.guard.over():
                return finish(CallResult(
                    "money", error_kind="max_usd", error=f"--max-usd ${self.guard.cap:.2f} "
                    f"reached (${self.guard.spent:.4f} at list price)",
                    stop_signal="money: --max-usd reached"))
            attempt += 1
            self.gate.wait(stop)
            if stop.is_set() or self._killing.is_set() or self.guard.over():
                attempt -= 1
                continue
            entry: dict[str, Any] = {"attempt": attempt, "sent": True}
            log_entries.append(entry)
            started = time.monotonic()
            try:
                message, headers, rid = self._stream(kwargs)
            except TimeLimit as exc:
                usd, _ = cost_of(exc.usage, request.model_id)
                self.guard.add(usd)
                total_cost += usd
                entry.update(outcome="time_limit", cost_usd=usd, usage=exc.usage)
                last = {"status": "error", "error_kind": "time_limit",
                        "error": f"still streaming at {self.timeout_s:.0f} s (cut and billed)"}
                break
            except sdk.AuthenticationError as exc:
                entry.update(outcome="401")
                return finish(CallResult("fatal", error_kind="auth",
                                         error=f"401: the API key is invalid or revoked: "
                                               f"{error_text(exc)[:200]}"))
            except sdk.PermissionDeniedError as exc:
                entry.update(outcome="403")
                return finish(CallResult("fatal", error_kind="permission",
                                         error=f"403: {error_text(exc)[:200]}"))
            except sdk.NotFoundError as exc:
                entry.update(outcome="404")
                return finish(CallResult("fatal", error_kind="not_found",
                                         error=f"404 for {request.model_id!r}: "
                                               f"{error_text(exc)[:200]}"))
            except sdk.BadRequestError as exc:
                reason = account_limit_reason(exc)
                entry.update(outcome="400", error=error_text(exc)[:300])
                if reason:
                    return finish(CallResult("money", error_kind="account_limit",
                                             error=reason[:300],
                                             stop_signal="money: the account's spend limit "
                                                         "or credit"))
                last = {"status": "error", "error_kind": "bad_request",
                        "error": f"400: {error_text(exc)[:400]}"}
                break
            except sdk.APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                body = getattr(exc, "body", None)
                etype = getattr(exc, "type", None)
                if etype is None and isinstance(body, dict) and isinstance(body.get("error"), dict):
                    etype = body["error"].get("type")
                entry.update(outcome=str(status), error=error_text(exc)[:300])
                if status == 402:
                    return finish(CallResult("fatal", error_kind="billing",
                                             error=f"402: {error_text(exc)[:200]}"))
                if status == 429 or etype == "rate_limit_error":
                    reason = account_limit_reason(exc)
                    if reason:
                        return finish(CallResult("money", error_kind="account_limit",
                                                 error=reason[:300],
                                                 stop_signal="money: the account's spend cap"))
                    entry["sent"] = False            # a 429 is not billed and is not an attempt
                    entry["outcome"] = "429"
                    attempt -= 1
                    rate_limited += 1
                    if rate_limited >= self.rate_limit_attempts:
                        last = {"status": "error", "error_kind": "rate_limit",
                                "error": f"429 {rate_limited} times: per-minute limits"}
                        break
                    response = getattr(exc, "response", None)
                    wait = rate_limit_wait(getattr(response, "headers", None))
                    wait = RATE_LIMIT_DEFAULT_WAIT_S if wait is None else wait
                    log.warning("429 (per-minute limit): every worker pauses %.0f s", wait)
                    self.gate.close_for(wait)
                    continue
                if (status is not None and status >= 500) or etype in RETRYABLE_ERROR_TYPES:
                    last = {"status": "error", "error_kind": f"http_{status or etype}",
                            "error": f"{status} {etype or ''}: {error_text(exc)[:200]}"}
                    if attempt < self.max_attempts:
                        self._backoff(attempt, stop)
                    continue
                last = {"status": "error", "error_kind": f"http_{status}",
                        "error": f"{status}: {error_text(exc)[:400]}"}
                break
            except timeout_classes as exc:
                timeouts += 1
                entry.update(outcome="timeout")
                last = {"status": "error", "error_kind": "timeout", "error": f"timeout: {exc}"}
                if timeouts >= MAX_TIMEOUT_ATTEMPTS:
                    break
                if attempt < self.max_attempts:
                    self._backoff(attempt, stop)
                continue
            except network_classes as exc:
                entry.update(outcome="network")
                last = {"status": "error", "error_kind": "network",
                        "error": f"{type(exc).__name__}: {exc}"}
                if stop.is_set() or self._killing.is_set():
                    break
                if attempt < self.max_attempts:
                    self._backoff(attempt, stop)
                continue
            except sdk.APIError as exc:
                entry.update(outcome="api_error")
                last = {"status": "error", "error_kind": "api_error",
                        "error": f"{type(exc).__name__}: {error_text(exc)[:400]}"}
                break

            # ---- a reply arrived: bill it, then read it
            elapsed = time.monotonic() - started
            raw = message.model_dump(mode="json")
            usd, tokens = cost_of(raw.get("usage"), request.model_id)
            self.guard.add(usd)
            total_cost += usd
            entry.update(cost_usd=usd, usage=raw.get("usage"), stop_reason=raw.get("stop_reason"),
                         elapsed_s=round(elapsed, 3), request_id=rid)
            served = raw.get("model")
            keep = {
                "latency_s": round(elapsed, 3),
                "raw_stream": json.dumps(raw, ensure_ascii=False),
                "raw_response": {"message": raw},
                "computed_usage": tokens,
                "served_models": [served] if isinstance(served, str) else [],
                "side_models": {},
                "api": {"model": served, "usage": raw.get("usage"),
                        "stop_reason": raw.get("stop_reason"),
                        "stop_details": raw.get("stop_details"), "request_id": rid,
                        "message_id": raw.get("id"), "ratelimit": ratelimit_snapshot(headers)},
            }
            other = served_by_other_model(raw, request.model_id)
            if other is not None:
                entry["outcome"] = "other_model"
                return finish(CallResult("refusal", error_kind="other_model",
                                         error=f"answered by {other}, not {request.model_id}"))
            if raw.get("stop_reason") == "refusal":
                details = raw.get("stop_details") if isinstance(raw.get("stop_details"), dict) else {}
                entry["outcome"] = "refusal"
                return finish(CallResult("refusal", error_kind="refusal",
                                         error=f"refusal (category {details.get('category')!r})"))
            if raw.get("stop_reason") == "max_tokens":
                entry["outcome"] = "max_tokens"
                last = {"status": "error", "error_kind": "max_tokens",
                        "error": f"truncated at max_tokens={kwargs['max_tokens']}"}
                break
            payload = json_from_text(raw.get("content") or [])
            problems = (["the reply's text is not one JSON object"] if payload is None
                        else validate(payload, request.schema))
            label = dig(payload, request.label_path) if payload is not None and not problems else None
            if not problems and label not in self.labels:
                problems = [f"label {label!r} not allowed"]
            if problems:
                entry["outcome"] = "invalid_answer"
                last = {"status": "parse_failed", "error_kind": "invalid_answer",
                        "error": "; ".join(problems)[:600]}
                if attempt < self.max_attempts:
                    self._backoff(attempt, stop)
                continue
            entry["outcome"] = "ok"
            res = CallResult("ok", label=label, parse_status="ok", answer=payload)
            finished = finish(res)
            finished.raw_response["answer_source"] = "text"
            return finished
        res = CallResult(last["status"], error=last.get("error"), error_kind=last.get("error_kind"))
        if last["status"] == "parse_failed":
            res.parse_status = "failed"
        if not any(e.get("sent") for e in log_entries):
            res.error = res.error or "the run stopped before the call"
            res.error_kind = res.error_kind if res.error_kind != "not_called" else "stopped"
        return finish(res)
