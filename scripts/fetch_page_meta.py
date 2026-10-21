#!/usr/bin/env python3
"""Fetch each corpus page's December 2020 revision, its timestamp and its categories from Wikipedia.

FEVEROUS scraped English Wikipedia in December 2020 (paper, Sec 7.7) and released the pages
without a revision id, an edit date or categories. For every page of data/corpus/pages.jsonl
this script finds, through the MediaWiki Action API, the Wikipedia revision whose text the
corpus holds, and writes data/corpus/page_meta.jsonl (one JSON object per page, in corpus
order) and a summary, data/corpus/PAGE_META.md. The ingest loads the file into
page.revision_id, page.last_edited, category and page_category, which the "edited since that
date" and "only articles in this category" filters use.

How the revision is chosen (``method`` in each record):

- ``last_before_end``: the page had no edit in December 2020, so the revision live at
  2020-12-31T23:59:59Z is the one FEVEROUS saw, whichever day of the month it scraped the page.
- ``best_text_match``: the page was edited during December 2020. Of the revisions live at
  some moment of the month (the last one before 1 December and every December edit), the one
  whose wikitext contains the most of the page's FEVEROUS sentences (the first six words of
  each, links reduced to their shown text) is chosen; a tie goes to the later revision.
- ``missing``: no revision of the title, nor of a title it was moved to after 2020, existed at
  the end of 2020. revision_id and last_edited are null and categories is empty.

A page renamed after 2020 is followed through the move log; ``wiki_title`` then names the title
the revision was found under. ``probes_matched`` / ``probes`` record how many of the page's
FEVEROUS sentence probes the chosen revision contains, a check that it is the right text.
``last_edited`` is the chosen revision's timestamp: when the page was last edited as the corpus
knows it, not its last edit today.

The categories are the ``[[Category:...]]`` links of the chosen revision's wikitext (HTML
comments removed; a ``[[:Category:...]]`` link points at a category without adding the page to
it). Categories that templates add when the page is rendered are not in the wikitext; most of
those are hidden maintenance categories.

Politeness (Wikimedia Robot Policy and API:Etiquette): one request at a time and at most four
per second, a User-Agent naming the project and its repository, gzip, ``maxlag=5``,
``Retry-After`` honoured on HTTP 429 or 503 and on maxlag errors, and a 5 s pause after any
request that took longer than 1 s. The 100 pages of the corpus take a few hundred requests.

Usage (from the project root):
    python3 scripts/fetch_page_meta.py                      # -> data/corpus/page_meta.jsonl
    python3 scripts/fetch_page_meta.py --limit 3 --out /tmp/trial.jsonl
"""

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from wikilense import __version__
from wikilense.corpus import DEFAULT_CORPUS_DIR, PAGE_META_FILE, iter_pages, nfc

API_URL = "https://en.wikipedia.org/w/api.php"
REPOSITORY_URL = "https://github.com/ylli-00/WikiLens"
USER_AGENT = (
    f"WikiLense/{__version__} ({REPOSITORY_URL}) "
    f"python-urllib/{sys.version_info[0]}.{sys.version_info[1]}"
)
DECEMBER_START = "2020-12-01T00:00:00Z"
BEFORE_DECEMBER = "2020-11-30T23:59:59Z"
DECEMBER_END = "2020-12-31T23:59:59Z"
MIN_INTERVAL = 0.25  # seconds from one request to the next: at most four per second
SLOW_REQUEST = 1.0  # a request slower than this ...
SLOW_PAUSE = 5.0  # ... is followed by this pause (Robot Policy, expensive requests)
MAX_RETRIES = 5
TIMEOUT = 60  # seconds per request
PROBE_WORDS = 6
MAX_MOVES = 3  # renames followed for one title
SUMMARY_FILE = "PAGE_META.md"

_FEVEROUS_LINK = re.compile(r"\[\[[^|\]]*\|([^\]]*)\]\]")  # FEVEROUS: [[Target|shown]]
_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")  # wikitext: [[Target]] or [[Target|shown]]
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CATEGORY = re.compile(r"\[\[\s*Category\s*:\s*([^\]|\n]+?)\s*(?:\|[^\]\n]*)?\]\]", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


class ApiError(RuntimeError):
    """The API answered with an error, or kept refusing after the retries."""


def _retry_after(headers):
    """Return the Retry-After header in seconds, or None when it is absent or not a number."""
    try:
        return max(0.0, float(headers.get("retry-after", "")))
    except ValueError:
        return None


def _urlopen(url):
    """GET ``url`` with the User-Agent and gzip; return (status, lower-case headers, body bytes)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            status, headers, body = response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        status, headers, body = exc.code, exc.headers, exc.read()
    headers = {key.lower(): value for key, value in headers.items()}
    if headers.get("content-encoding", "").lower() == "gzip":
        body = gzip.decompress(body)
    return status, headers, body


class Api:
    """The MediaWiki Action API, one request at a time within the politeness limits above.

    ``opener(url) -> (status, headers, body)``, ``sleep`` and ``clock`` are replaceable so that
    the tests run without the network or real waiting.
    """

    def __init__(self, url=API_URL, opener=_urlopen, sleep=time.sleep, clock=time.monotonic):
        self.url = url
        self.opener = opener
        self.sleep = sleep
        self.clock = clock
        self.requests = 0
        self._last = None

    def get(self, **params):
        """Return the decoded JSON of one API call; retries 429, 503 and maxlag, raises ApiError."""
        query = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        url = self.url + "?" + urllib.parse.urlencode(query)
        for attempt in range(MAX_RETRIES + 1):
            if self._last is not None:
                wait = MIN_INTERVAL - (self.clock() - self._last)
                if wait > 0:
                    self.sleep(wait)
            start = self.clock()
            status, headers, body = self.opener(url)
            self._last = self.clock()
            self.requests += 1
            retry_after = _retry_after(headers)
            data = json.loads(body) if status == 200 else None
            lagged = data is not None and (data.get("error") or {}).get("code") == "maxlag"
            if status in (429, 503) or lagged:
                if attempt == MAX_RETRIES:
                    raise ApiError(f"still refused after {MAX_RETRIES} retries: {url}")
                self.sleep(retry_after if retry_after is not None else 5.0 * (attempt + 1))
                continue
            if status != 200:
                raise ApiError(f"HTTP {status} for {url}")
            if "error" in data:
                error = data["error"]
                raise ApiError(f"API error {error.get('code')}: {error.get('info')}")
            if self._last - start > SLOW_REQUEST:
                self.sleep(SLOW_PAUSE)
            return data
        raise AssertionError("unreachable")


def _revisions(data):
    """Return the revisions of the one page in a ``prop=revisions`` answer (empty if none)."""
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or pages[0].get("invalid"):
        return []
    return [
        {
            "revid": int(rev["revid"]),
            "timestamp": rev["timestamp"],
            "text": nfc(rev.get("slots", {}).get("main", {}).get("content", "")),
        }
        for rev in pages[0].get("revisions", [])
    ]


def revision_at(api, title, moment):
    """Return the revision of ``title`` live at ``moment`` (the last one at or before it) with its
    wikitext, or None when the title had no revision then."""
    data = api.get(
        action="query", prop="revisions", titles=title, rvlimit="1", rvdir="older",
        rvstart=moment, rvprop="ids|timestamp|content", rvslots="main",
    )  # fmt: skip
    revisions = _revisions(data)
    return revisions[0] if revisions else None


def december_revisions(api, title):
    """Return every revision of ``title`` saved during December 2020, oldest first, with text."""
    revisions, more = [], {}
    while True:
        data = api.get(
            action="query", prop="revisions", titles=title, rvlimit="50", rvdir="newer",
            rvstart=DECEMBER_START, rvend=DECEMBER_END, rvprop="ids|timestamp|content",
            rvslots="main", **more,
        )  # fmt: skip
        revisions.extend(_revisions(data))
        if "continue" not in data:
            return revisions
        more = dict(data["continue"])


def moved_to(api, title):
    """Return the title that ``title`` was moved to after 2020 (the first such move), or None."""
    data = api.get(
        action="query", list="logevents", letype="move", letitle=title, ledir="newer",
        lestart=DECEMBER_END, lelimit="10", leprop="title|details|timestamp",
    )  # fmt: skip
    for event in data.get("query", {}).get("logevents", []):
        target = (event.get("params") or {}).get("target_title")
        if target:
            return target
    return None


def probes(page):
    """Return the first PROBE_WORDS words of every FEVEROUS sentence that has that many words,
    links reduced to their shown text (``[[Target|shown]]`` -> ``shown``)."""
    found = []
    for key in page.get("order", []):
        if key.startswith("sentence_"):
            words = _FEVEROUS_LINK.sub(r"\1", nfc(page.get(key, ""))).split()
            if len(words) >= PROBE_WORDS:
                found.append(" ".join(words[:PROBE_WORDS]))
    return found


def plain_wikitext(text):
    """Return wikitext with comments, references, link brackets and bold/italic quotes removed."""
    text = _COMMENT.sub(" ", text)
    text = _REF.sub(" ", text)
    text = _WIKI_LINK.sub(r"\1", text)
    return _SPACE.sub(" ", text.replace("'''", "").replace("''", "")).strip()


def matched(probe_list, text):
    """Return how many of ``probe_list`` occur in the plain form of the wikitext ``text``."""
    plain = plain_wikitext(text)
    return sum(1 for probe in probe_list if probe in plain)


def categories(text):
    """Return the category names of a page's wikitext, in order, without duplicates.

    Each ``[[Category:Name]]`` or ``[[Category:Name|sort key]]`` link (any case of "Category")
    gives ``Name`` with underscores as spaces, runs of spaces collapsed and the first letter
    upper-cased, as MediaWiki stores it. HTML comments are removed first.
    """
    names = []
    for raw in _CATEGORY.findall(_COMMENT.sub(" ", text)):
        name = _SPACE.sub(" ", raw.replace("_", " ")).strip()
        if not name:
            continue
        name = nfc(name[0].upper() + name[1:])
        if name not in names:
            names.append(name)
    return names


def fetch_page(api, page):
    """Return the page_meta.jsonl record of one FEVEROUS page (see the module docstring)."""
    title = nfc(page["title"])
    probe_list = probes(page)
    record = {
        "title": title, "wiki_title": title, "method": "missing", "revision_id": None,
        "last_edited": None, "december_edits": 0, "probes": len(probe_list),
        "probes_matched": 0, "categories": [],
    }  # fmt: skip
    wiki_title = title
    revision = revision_at(api, wiki_title, DECEMBER_END)
    for _ in range(MAX_MOVES):
        if revision is not None:
            break
        target = moved_to(api, wiki_title)
        if target is None:
            break
        wiki_title = target
        revision = revision_at(api, wiki_title, DECEMBER_END)
    if revision is None:
        return record
    december = december_revisions(api, wiki_title)
    if december:
        before = revision_at(api, wiki_title, BEFORE_DECEMBER)
        candidates = ([before] if before else []) + december
        revision = max(candidates, key=lambda r: (matched(probe_list, r["text"]), r["timestamp"]))
        method = "best_text_match"
    else:
        method = "last_before_end"
    record.update(
        wiki_title=wiki_title,
        method=method,
        revision_id=revision["revid"],
        last_edited=revision["timestamp"],
        december_edits=len(december),
        probes_matched=matched(probe_list, revision["text"]),
        categories=categories(revision["text"]),
    )
    return record


def _by_count(item):
    """Sort key of a (name, count) pair: the larger count first, then the name."""
    return (-item[1], item[0])


def _share(part, whole):
    """Return "part of whole (pct%)", or "part of whole" when whole is 0."""
    return f"{part} of {whole}" + (f" ({100 * part / whole:.1f}%)" if whole else "")


def summary(records, n_requests, fetched_at):
    """Return the Markdown summary of the records (PAGE_META.md)."""
    methods = Counter(r["method"] for r in records)
    found = [r for r in records if r["revision_id"] is not None]
    moved = sum(1 for r in found if r["wiki_title"] != r["title"])
    with_categories = sum(1 for r in records if r["categories"])
    pairs = sum(len(r["categories"]) for r in records)
    distinct = len({name for r in records for name in r["categories"]})
    shared = Counter(name for r in records for name in r["categories"])
    probes_total = sum(r["probes"] for r in found)
    probes_hit = sum(r["probes_matched"] for r in found)
    full = sum(1 for r in found if r["probes"] and r["probes_matched"] == r["probes"])
    dates = sorted(r["last_edited"] for r in found)
    years = Counter(d[:4] for d in dates)
    rows = [
        ("pages", len(records)),
        ("revision found", len(found)),
        ("method last_before_end (no edit in December 2020)", methods["last_before_end"]),
        ("method best_text_match (edited in December 2020)", methods["best_text_match"]),
        ("method missing (no revision at the end of 2020)", methods["missing"]),
        ("found under a title it was moved to after 2020", moved),
        ("pages with at least one category", with_categories),
        ("page-category pairs", pairs),
        ("distinct categories", distinct),
        ("categories shared by two or more pages", sum(1 for n in shared.values() if n > 1)),
        ("FEVEROUS sentence probes found in the chosen revision", _share(probes_hit, probes_total)),
        ("pages whose every probe was found", f"{full} of {len(found)}"),
        ("last_edited, earliest", dates[0] if dates else "-"),
        ("last_edited, latest", dates[-1] if dates else "-"),
    ]
    lines = [
        "# Page metadata",
        "",
        (
            f"Written by `scripts/fetch_page_meta.py` on {fetched_at} from the MediaWiki API "
            f"({API_URL}) with {n_requests} requests. Each record names the Wikipedia revision "
            "whose text the corpus holds (the one live in December 2020, when FEVEROUS scraped "
            "the page), its timestamp and its categories; the script's docstring explains how "
            "the revision is chosen. The revisions lie in the past, so a re-run finds the same "
            "ones unless a page was deleted or moved since."
        ),
        "",
        "| Measure | Value |",
        "|---|---|",
        *(f"| {name} | {value} |" for name, value in rows),
        "",
        "## Last edit before the snapshot, by year",
        "",
        "| Year | Pages |",
        "|---|---|",
        *(f"| {year} | {n} |" for year, n in sorted(years.items())),
        "",
        "## Categories with the most pages",
        "",
        "| Category | Pages |",
        "|---|---|",
        *(f"| {name} | {n} |" for name, n in sorted(shared.items(), key=_by_count)[:15]),
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch each corpus page's December 2020 Wikipedia revision, its timestamp and "
        "its categories (see the module docstring)."
    )
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR),
                        help="directory with pages.jsonl (default data/corpus)")
    parser.add_argument("--out", help=f"output file (default <corpus-dir>/{PAGE_META_FILE}; the "
                        f"summary {SUMMARY_FILE} is written next to the default file only)")
    parser.add_argument("--limit", type=int, help="fetch only the first N pages (a trial run)")
    args = parser.parse_args(argv)
    corpus_dir = Path(args.corpus_dir)
    default_out = corpus_dir / PAGE_META_FILE
    out = Path(args.out) if args.out else default_out
    pages = list(iter_pages(corpus_dir / "pages.jsonl"))
    if args.limit is not None:
        pages = pages[: args.limit]
    api = Api()
    records = []
    for number, page in enumerate(pages, start=1):
        try:
            record = fetch_page(api, page)
        except (ApiError, OSError, ValueError) as exc:
            sys.exit(f"fetch_page_meta: {page['title']!r}: {exc}")
        records.append(record)
        print(
            f"[{number}/{len(pages)}] {record['title']}: {record['method']}, "
            f"{record['last_edited']}, {len(record['categories'])} categories, "
            f"probes {record['probes_matched']}/{record['probes']}",
            file=sys.stderr,
        )
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    temporary.replace(out)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = summary(records, api.requests, fetched_at)
    if out.resolve() == default_out.resolve():
        (corpus_dir / SUMMARY_FILE).write_text(text, encoding="utf-8")
    else:
        print(text)
    print(f"wrote {out}: {len(records)} pages, {api.requests} API requests", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
