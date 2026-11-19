"""A fake Anthropic Messages API for the API backend's offline tests (no network, no key).

The real ``anthropic``
SDK talks to this object through ``httpx.MockTransport``, so the request the SDK builds and the
streamed (SSE) replies it parses are the real code paths. What a ``POST /v1/messages`` does is
chosen by a TRIGGER_* word in the user turn:

  (none)            a streamed answer {"result": {"label": "SUPPORTS"...}} (LABEL_REFUTES / LABEL_NEI)
  TRIGGER_AUTH      401                        TRIGGER_CREDIT     400 "credit balance is too low"
  TRIGGER_SPENDCAP  429 enforced_spend_limit   TRIGGER_RATE429    429 retry-after 1, then OK
  TRIGGER_OVERLOAD  529 overloaded, then OK    TRIGGER_BADREQ     400 (not retryable)
  TRIGGER_REFUSE    stop_reason refusal        TRIGGER_OTHERMODEL the reply names another model
  TRIGGER_TRUNC     stop_reason max_tokens     TRIGGER_BADSCHEMA  label outside the enum
  TRIGGER_BADONCE   label outside the enum on the first call, valid after
  TRIGGER_SLOW      the stream stalls 2.5 s after message_start

``GET /v1/models/<id>`` (the free preflight) answers for every id not in ``unknown_models``,
and 401 when ``auth_fails`` is set. Every request body and its x-api-key header are recorded.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx

USAGE = {"input_tokens": 12, "output_tokens": 1, "cache_read_input_tokens": 1000,
         "cache_creation_input_tokens": 300,
         "cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 0},
         "service_tier": "standard", "inference_geo": "global"}
OUTPUT_TOKENS = 50
#: List price of one default reply at Sonnet 5.5's prices: 12 x 2 + 1000 x 0.10 + 300 x 2.50
#: + 50 x 10, per million.
SONNET_COST = (12 * 2.00 + 1000 * 0.10 + 300 * 2.50 + OUTPUT_TOKENS * 10.00) / 1e6


def answer_text(label: str = "SUPPORTS") -> str:
    """The JSON text a normal reply carries."""
    return json.dumps({"result": {"label": label, "justification": "fake reply"}})


def sse(model: str, text: str, stop_reason: str = "end_turn", stop_details: Any = None,
        stall_s: float = 0.0) -> Any:
    """A streamed reply in the API's SSE shape (a thinking block, a ping, a text block)."""
    start = [{"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None, "usage": dict(USAGE)}},
        {"type": "ping"}]
    delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    rest = [
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": delta, "usage": {"output_tokens": OUTPUT_TOKENS}},
        {"type": "message_stop"},
    ]

    def enc(events: list[dict[str, Any]]) -> bytes:
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()

    if not stall_s:
        return enc(start + rest)

    def gen() -> Any:
        yield enc(start)
        time.sleep(stall_s)
        yield enc(rest)
    return gen()


def error(status: int, etype: str, message: str, headers: dict[str, str] | None = None,
          details: dict[str, Any] | None = None) -> httpx.Response:
    """An API error response."""
    err: dict[str, Any] = {"type": etype, "message": message}
    if details:
        err["details"] = details
    return httpx.Response(status, headers=headers or {}, json={"type": "error", "error": err})


class FakeAPI:
    """The mocked endpoints. Records every Messages request body and its API key header."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.keys: list[str | None] = []
        self.model_lookups: list[str] = []
        self.lock = threading.Lock()
        self.delay_s = delay_s
        self.seen: set[str] = set()
        self.unknown_models: set[str] = set()
        self.auth_fails = False

    @staticmethod
    def user_text(body: dict[str, Any]) -> str:
        """The user turn's text, whether a string or text blocks."""
        content = body["messages"][0]["content"]
        return content if isinstance(content, str) else "".join(b["text"] for b in content)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.startswith("/v1/models/"):
            model = request.url.path.rsplit("/", 1)[1]
            with self.lock:
                self.model_lookups.append(model)
            if self.auth_fails:
                return error(401, "authentication_error", "invalid x-api-key")
            if model in self.unknown_models:
                return error(404, "not_found_error", f"model: {model}")
            return httpx.Response(200, json={"id": model, "type": "model", "display_name": model,
                                             "created_at": "2026-01-01T00:00:00Z"})
        body = json.loads(request.content)
        text = self.user_text(body)
        with self.lock:
            self.calls.append(body)
            self.keys.append(request.headers.get("x-api-key"))
            first = text not in self.seen
            self.seen.add(text)
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.respond(body, text, first)

    def respond(self, body: dict[str, Any], text: str, first: bool) -> httpx.Response:
        """The reply chosen by the trigger word."""
        model = body["model"]
        hdr = {"content-type": "text/event-stream", "request-id": "req_test123",
               "anthropic-ratelimit-input-tokens-limit": "2000000",
               "anthropic-ratelimit-input-tokens-remaining": "1999000"}
        label = "REFUTES" if "LABEL_REFUTES" in text else (
            "NOT ENOUGH INFO" if "LABEL_NEI" in text else "SUPPORTS")
        if "TRIGGER_AUTH" in text:
            return error(401, "authentication_error", "invalid x-api-key")
        if "TRIGGER_CREDIT" in text:
            return error(400, "invalid_request_error",
                         "Your credit balance is too low to access the Anthropic API.")
        if "TRIGGER_SPENDCAP" in text:
            return error(429, "rate_limit_error", "monthly cap reached",
                         details={"error_code": "enforced_spend_limit_reached"})
        if "TRIGGER_RATE429" in text and first:
            return error(429, "rate_limit_error", "rate limited", headers={"retry-after": "1"})
        if "TRIGGER_OVERLOAD" in text and first:
            return error(529, "overloaded_error", "Overloaded")
        if "TRIGGER_BADREQ" in text:
            return error(400, "invalid_request_error", "messages.0: something is wrong")
        if "TRIGGER_REFUSE" in text:
            return httpx.Response(200, headers=hdr, content=sse(
                model, "", "refusal", {"type": "refusal", "category": "cyber",
                                       "explanation": "declined"}))
        if "TRIGGER_OTHERMODEL" in text:
            return httpx.Response(200, headers=hdr, content=sse("claude-sonnet-5", answer_text()))
        if "TRIGGER_TRUNC" in text:
            return httpx.Response(200, headers=hdr,
                                  content=sse(model, '{"result": {"label": "SUPP', "max_tokens"))
        if "TRIGGER_BADSCHEMA" in text or ("TRIGGER_BADONCE" in text and first):
            return httpx.Response(200, headers=hdr, content=sse(model, answer_text("MAYBE")))
        if "TRIGGER_SLOW" in text:
            return httpx.Response(200, headers=hdr,
                                  content=sse(model, answer_text(label), stall_s=2.5))
        return httpx.Response(200, headers=hdr, content=sse(model, answer_text(label)))


def client_factory(fake: FakeAPI) -> Any:
    """A factory for AnthropicApiBackend(client_factory=...) whose HTTP goes to ``fake``."""
    import anthropic

    def make(api_key: str, timeout_s: float) -> Any:
        return anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com",
                                   max_retries=0, timeout=timeout_s,
                                   http_client=httpx.Client(transport=httpx.MockTransport(fake)))
    return make
