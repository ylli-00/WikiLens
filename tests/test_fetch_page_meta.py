"""Tests for scripts/fetch_page_meta.py without the network: the API client's politeness rules,
the category extraction, the choice of the revision and the following of a moved title."""

from __future__ import annotations

import contextlib
import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from wikilense.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "fetch_page_meta.py"

SENTENCE_0 = "The river flows through the old town before it joins the sea."
SENTENCE_1 = "Its water is used by three power stations along the valley."
PAGE = {
    "title": "River",
    "order": ["sentence_0", "section_0", "sentence_1", "sentence_2"],
    "sentence_0": "The [[River_(disambiguation)|river]] flows through the old town before it joins "
    "the sea.",
    "section_0": {"value": "Use", "level": 2},
    "sentence_1": SENTENCE_1,
    "sentence_2": "Too short.",
}
FULL_TEXT = f"The '''river''' flows through the old town<ref>A source.</ref> before it joins the sea. {SENTENCE_1}"
HALF_TEXT = SENTENCE_0


@pytest.fixture(scope="module")
def fpm() -> Any:
    spec = importlib.util.spec_from_file_location("fetch_page_meta", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    """A fake monotonic clock: ``sleep`` records the wait and advances the time."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _ok(data: dict, headers: dict | None = None) -> tuple[int, dict, bytes]:
    return 200, headers or {}, json.dumps(data).encode("utf-8")


def _rev(revid: int, timestamp: str, text: str) -> dict:
    return {"revid": revid, "timestamp": timestamp, "text": text}


class FakeApi:
    """Answers the revision and move-log queries of fetch_page from a fixed page history."""

    def __init__(self, history: dict[str, list[dict]], moves: dict[str, str] | None = None):
        self.history = history
        self.moves = moves or {}
        self.calls: list[dict] = []
        self.requests = 0

    def get(self, **params: str) -> dict:
        self.calls.append(params)
        self.requests += 1
        if params.get("list") == "logevents":
            target = self.moves.get(params["letitle"])
            events = [{"params": {"target_title": target}}] if target else []
            return {"query": {"logevents": events}}
        revisions = self.history.get(params["titles"])
        if revisions is None:
            return {"query": {"pages": [{"title": params["titles"], "missing": True}]}}
        if params["rvdir"] == "older":
            chosen = [r for r in revisions if r["timestamp"] <= params["rvstart"]][-1:]
        else:
            chosen = [r for r in revisions if params["rvstart"] <= r["timestamp"] <= params["rvend"]]
        return {"query": {"pages": [{"title": params["titles"], "revisions": [
            {"revid": r["revid"], "timestamp": r["timestamp"],
             "slots": {"main": {"content": r["text"]}}}
            for r in chosen
        ]}]}}  # fmt: skip


# ---------------------------------------------------------------------------------------------
# the API client
# ---------------------------------------------------------------------------------------------


def test_requests_are_spaced_and_carry_the_policy_parameters(fpm: Any) -> None:
    clock = Clock()
    urls: list[str] = []

    def opener(url: str) -> tuple[int, dict, bytes]:
        urls.append(url)
        clock.now += 0.05
        return _ok({"query": {}})

    api = fpm.Api(opener=opener, sleep=clock.sleep, clock=clock.clock)
    api.get(action="query", titles="A")
    api.get(action="query", titles="B")
    assert api.requests == 2
    assert clock.sleeps == [pytest.approx(fpm.MIN_INTERVAL)]  # one request at a time, <= 4/s
    for part in ("maxlag=5", "format=json", "formatversion=2", "titles=A"):
        assert part in urls[0]


def test_429_and_maxlag_are_retried_after_retry_after(fpm: Any) -> None:
    clock = Clock()
    answers = [
        (429, {"retry-after": "7"}, b""),
        _ok({"error": {"code": "maxlag", "info": "lagged"}}, {"retry-after": "3"}),
        _ok({"query": {"x": 1}}),
    ]
    api = fpm.Api(opener=lambda url: answers.pop(0), sleep=clock.sleep, clock=clock.clock)
    assert api.get(action="query") == {"query": {"x": 1}}
    assert api.requests == 3
    assert 7.0 in clock.sleeps and 3.0 in clock.sleeps


def test_refusals_errors_and_other_statuses_raise(fpm: Any) -> None:
    clock = Clock()
    api = fpm.Api(opener=lambda url: (503, {}, b""), sleep=clock.sleep, clock=clock.clock)
    with pytest.raises(fpm.ApiError, match="still refused"):
        api.get(action="query")
    assert api.requests == fpm.MAX_RETRIES + 1
    api = fpm.Api(opener=lambda url: _ok({"error": {"code": "badvalue", "info": "no"}}),
                  sleep=clock.sleep, clock=clock.clock)
    with pytest.raises(fpm.ApiError, match="badvalue"):
        api.get(action="query")
    api = fpm.Api(opener=lambda url: (404, {}, b""), sleep=clock.sleep, clock=clock.clock)
    with pytest.raises(fpm.ApiError, match="HTTP 404"):
        api.get(action="query")


def test_a_slow_request_is_followed_by_the_pause(fpm: Any) -> None:
    clock = Clock()

    def opener(url: str) -> tuple[int, dict, bytes]:
        clock.now += 1.5
        return _ok({})

    fpm.Api(opener=opener, sleep=clock.sleep, clock=clock.clock).get(action="query")
    assert clock.sleeps == [fpm.SLOW_PAUSE]


def test_urlopen_sends_the_user_agent_without_an_email_and_decodes_gzip(
    fpm: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class Response:
        def __init__(self) -> None:
            self.status = 200
            self.headers = {"Content-Encoding": "gzip"}

        def read(self) -> bytes:
            return gzip.compress(b'{"ok": true}')

    def fake_urlopen(request: Any, timeout: float) -> contextlib.nullcontext[Response]:
        seen["agent"] = request.get_header("User-agent")
        seen["encoding"] = request.get_header("Accept-encoding")
        return contextlib.nullcontext(Response())

    monkeypatch.setattr(fpm.urllib.request, "urlopen", fake_urlopen)
    status, headers, body = fpm._urlopen("https://example.org/w/api.php")
    assert (status, body) == (200, b'{"ok": true}')
    assert headers == {"content-encoding": "gzip"}
    assert seen["agent"].startswith("WikiLense/") and fpm.REPOSITORY_URL in seen["agent"]
    assert "@" not in seen["agent"]  # the contact is the repository, never an email address
    assert seen["encoding"] == "gzip"


# ---------------------------------------------------------------------------------------------
# text: categories and sentence probes
# ---------------------------------------------------------------------------------------------


def test_categories_are_read_from_the_wikitext_as_mediawiki_names_them(fpm: Any) -> None:
    text = (
        "Text.\n[[Category:Rivers of Switzerland]] [[category:aare basin|Aare]]\n"
        "[[Category: Rivers_of_the  canton of Bern ]] [[:Category:Not a member]]\n"
        "<!-- [[Category:Commented out]] --> [[Category:Rivers of Switzerland]] [[Kategorie:X]]"
    )
    assert fpm.categories(text) == [
        "Rivers of Switzerland", "Aare basin", "Rivers of the canton of Bern"
    ]
    assert fpm.categories("No categories here.") == []


def test_probes_reduce_links_skip_short_sentences_and_match_plain_wikitext(fpm: Any) -> None:
    assert fpm.probes(PAGE) == [
        "The river flows through the old", "Its water is used by three"
    ]
    assert fpm.matched(fpm.probes(PAGE), FULL_TEXT) == 2  # bold quotes and the ref removed
    assert fpm.matched(fpm.probes(PAGE), HALF_TEXT) == 1
    assert fpm.plain_wikitext("a [[B|c]] [[D]] <!-- e --> '''f''' g<ref name=x/>") == "a c D f g"


# ---------------------------------------------------------------------------------------------
# choosing the revision
# ---------------------------------------------------------------------------------------------


def test_without_december_edits_the_revision_live_at_the_end_of_2020_is_taken(fpm: Any) -> None:
    api = FakeApi({"River": [
        _rev(1, "2019-03-01T00:00:00Z", HALF_TEXT + " [[Category:Old]]"),
        _rev(2, "2020-11-17T02:44:45Z", FULL_TEXT + " [[Category:Rivers]] [[Category:Old towns]]"),
        _rev(3, "2021-05-01T00:00:00Z", "Rewritten. [[Category:Changed]]"),
    ]})  # fmt: skip
    assert fpm.fetch_page(api, PAGE) == {
        "title": "River", "wiki_title": "River", "method": "last_before_end", "revision_id": 2,
        "last_edited": "2020-11-17T02:44:45Z", "december_edits": 0, "probes": 2,
        "probes_matched": 2, "categories": ["Rivers", "Old towns"],
    }  # fmt: skip


def test_with_december_edits_the_best_text_match_is_taken_and_a_tie_goes_to_the_later(
    fpm: Any,
) -> None:
    api = FakeApi({"River": [
        _rev(1, "2020-11-01T00:00:00Z", HALF_TEXT),
        _rev(2, "2020-12-05T00:00:00Z", FULL_TEXT + " [[Category:A]]"),
        _rev(3, "2020-12-20T00:00:00Z", HALF_TEXT + " [[Category:B]]"),
        _rev(4, "2021-01-02T00:00:00Z", FULL_TEXT + " [[Category:C]]"),
    ]})  # fmt: skip
    record = fpm.fetch_page(api, PAGE)
    assert (record["method"], record["revision_id"], record["december_edits"]) == (
        "best_text_match", 2, 2
    )
    assert record["probes_matched"] == 2 and record["categories"] == ["A"]
    tie = FakeApi({"River": [
        _rev(1, "2020-11-01T00:00:00Z", FULL_TEXT),
        _rev(2, "2020-12-05T00:00:00Z", FULL_TEXT),
    ]})  # fmt: skip
    assert fpm.fetch_page(tie, PAGE)["revision_id"] == 2


def test_a_title_moved_after_2020_is_followed_and_an_unknown_page_is_missing(fpm: Any) -> None:
    api = FakeApi(
        {
            "River": [_rev(9, "2024-02-21T21:07:33Z", "#REDIRECT [[River (Asia)]]")],
            "River (Asia)": [_rev(5, "2020-10-01T00:00:00Z", FULL_TEXT + " [[Category:Asia]]")],
        },
        moves={"River": "River (Asia)"},
    )
    record = fpm.fetch_page(api, PAGE)
    assert (record["wiki_title"], record["revision_id"], record["method"]) == (
        "River (Asia)", 5, "last_before_end"
    )
    assert record["categories"] == ["Asia"]
    missing = fpm.fetch_page(FakeApi({}), PAGE)
    assert missing["method"] == "missing" and missing["revision_id"] is None
    assert missing["last_edited"] is None and missing["categories"] == []


def test_main_writes_the_records_and_the_summary(
    fpm: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = [PAGE, dict(PAGE, title="Lost")]
    (tmp_path / "pages.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in pages), encoding="utf-8"
    )
    history = {"River": [_rev(2, "2020-11-17T02:44:45Z", FULL_TEXT + " [[Category:Rivers]]")]}
    monkeypatch.setattr(fpm, "Api", lambda: FakeApi(history))
    assert fpm.main(["--corpus-dir", str(tmp_path)]) == 0
    lines = (tmp_path / "page_meta.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["method"] for line in lines] == ["last_before_end", "missing"]
    summary = (tmp_path / fpm.SUMMARY_FILE).read_text(encoding="utf-8")
    assert "| pages | 2 |" in summary and "| revision found | 1 |" in summary
    assert "| Rivers | 1 |" in summary
    # another --out writes no summary file but prints it; --limit takes the first pages only
    other = tmp_path / "trial.jsonl"
    (tmp_path / fpm.SUMMARY_FILE).unlink()
    assert fpm.main(["--corpus-dir", str(tmp_path), "--out", str(other), "--limit", "1"]) == 0
    assert len(other.read_text(encoding="utf-8").splitlines()) == 1
    assert not (tmp_path / fpm.SUMMARY_FILE).exists()
    assert "| pages | 1 |" in capsys.readouterr().out
