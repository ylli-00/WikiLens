"""Minimal FastAPI query page: ``GET /`` is one self-contained HTML form, ``GET /api/search`` is JSON.

``create_app(settings=None, embedder=None, database=None, warm_up=True)`` builds the app. The
embedder is created once per app and, with ``warm_up``, loads its model at creation by
embedding one warm-up text, so the first request's ``timing_ms.embed`` is the embedding alone
and not the model load; the milliseconds that warm-up took are kept as
``app.state.model_load_ms`` and returned as ``timing_ms.model_load`` (``null`` without
warm-up: then the first request carries the load in ``embed``). A database connection is
opened per request and closed in a ``finally`` clause.

``/api/search`` takes every field of ``search.Filters`` (``min_words``, ``max_words``,
``heading``, ``path``, ``linked_from``, ``links_to``, ``titles`` (repeated once per title),
``category``, ``edited_since`` and ``edited_before`` (``YYYY-MM-DD``, UTC);
:data:`API_FILTER_FIELDS` maps the names), the strategy (the choices are
``search.STRATEGIES``, as the ``Literal`` :data:`Strategy`), ``overfetch``, ``ef_search`` and
``sentences``; the accepted names are :data:`API_PARAMETER_NAMES`, read from the endpoint's
signature, and any other query parameter is rejected with 400 (FastAPI would ignore it, and a
mistyped filter would search without it). ``ef_search`` defaults to ``settings.ef_search``
(``WIKILENSE_EF_SEARCH``); ``0`` leaves the server's session value alone, as ``--ef-search 0``
does on the command line. The response is ``{"hits": [...], "sql": "...",
"explain": [...], "ef_search": N, "timing_ms": {"model_load": ..., "embed": ..., "sql": ...}}``
where ``hits`` are ``search.Hit`` dicts (with ``sentences=1`` each also carries
``"sentences": [{"element_key", "text"}, ...]``, the chunk's rows of ``chunk_sentence`` joined to
``sentence`` in page order), ``sql`` is the statement that ran (values are bound as parameters,
so it shows ``%s`` placeholders), ``explain`` its ``EXPLAIN`` rows, ``ef_search`` the effective
``mhnsw_ef_search`` and the timings are in milliseconds. ``search()`` always gets
``query_text=<the query>`` (``rrf`` needs it; the other strategies ignore it).

The interactive API documentation stays enabled on purpose so that a reader can try the API
from the browser: Swagger UI at :data:`DOCS_URL` and the schema at :data:`OPENAPI_URL`; the
page footer links to both. The page has no JavaScript framework and no external assets: the
HTML, CSS and the few lines of script that call the API live in :data:`INDEX_HTML`, which
:func:`render_index` completes with the strategy list and the ``ef_search`` default.

Databases: the page offers at most two (:func:`offered_databases`), the demo database (the
``database`` given to :func:`create_app`, else ``settings.db_name``), always, and the scaled
database :data:`SCALED_DATABASE` when it exists on the server, has the schema and holds chunks.
The ``database`` parameter of every endpoint accepts only those names (anything else is a 422),
so the API never reaches another database; each request opens one connection to the chosen one.
``GET /api/databases`` lists the choices with their sizes, the note shown when S1 is missing, and
the ready-made examples (``picker.EXAMPLES``).

Claims: ``GET /api/claims`` lists the claims the picker offers on a database (the rule is in
``wikilense.picker``); ``GET /api/pages`` lists a database's page titles. ``claim_id`` on
``/api/search`` adds ``claim`` (the claim's details and gold evidence), ``marks`` (one per hit:
gold page, gold evidence, the gold sentences' element keys) and ``summary`` (the claim at ``k`` by
the evaluation's definitions); ``q`` must then be the claim's text. They are computed after the
timed statement, so ``sql`` and ``timing_ms.sql`` are the search's alone.

Repeated timing: ``repeat=N`` (1 to :data:`MAX_REPEAT`) runs the search statement N times on the
same connection, one after another, with the query embedded once. The hits are those of the
first run and ``timing_ms.sql`` is its time; the response gains ``repeat`` with every run's SQL
time, their median and range, and whether every run returned the first run's hits.

Without ``database``, ``claim_id`` and ``repeat`` a response is as it was without them.

Errors: invalid parameters are 422 (FastAPI validation, an inconsistent filter pair, a filter
given with strategy ``none``, an argument ``search`` refuses, a database that is not offered, or
a ``q`` that is not the picked claim's text), 404 (a claim that is not offered on the database)
or 400 (a blank query, an unknown parameter); a server that cannot be reached, a database without schema, without
chunks or without the FULLTEXT index that ``rrf`` needs, a model that cannot be loaded, or one
whose vectors do not have the ``chunk.embedding`` dimension are 503 with a one-sentence
``detail``.
"""

from __future__ import annotations

import html
import inspect
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import numpy as np
import pymysql
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from wikilense import __version__, db, picker, search
from wikilense.config import Settings, load_settings
from wikilense.embedding import Embedder
from wikilense.evaluate import latency_stats
from wikilense.search import EF_SEARCH_VARIABLE, STRATEGIES, Filters

logger = logging.getLogger(__name__)

DEFAULT_K = 5
MAX_K = 100
"""Largest ``k`` the API accepts (the page shows at most this many hits)."""
DEFAULT_OVERFETCH = 10
DEFAULT_STRATEGY = "inline"

WARM_UP_TEXT = "warm-up"

DOCS_URL = "/docs"
OPENAPI_URL = "/openapi.json"

#: The scaled database the page offers next to the demo when it is loaded (never S0, never a test
#: database).
SCALED_DATABASE = "wikilense_s1"
MAX_REPEAT = 10
"""Largest ``repeat`` the API accepts."""
PAGE_REPEAT = 5
"""The page's "repeat" checkbox sends ``repeat`` with this value."""

S1_MISSING_NOTE = (
    f"The scaled database S1 ({SCALED_DATABASE}) is not loaded on this server. docs/SETUP.md, "
    'section "Get the scaled database (S1)", has the two steps: make fetch-s1, then make load-s1.'
)
EXAMPLES_NOTE = (
    "These examples use the same claims on both databases, so you can switch and compare."
)

CHUNK_COUNT_SQL = "SELECT COUNT(*) FROM chunk"
ER_DBACCESS_DENIED = 1044  # MariaDB: "Access denied for user ... to database ..." (also unknown)
ER_BAD_DB = 1049  # MariaDB: "Unknown database ..."
ER_NO_SUCH_TABLE = 1146  # MariaDB: "Table ... doesn't exist"
ER_FT_MATCHING_KEY_NOT_FOUND = 1191  # MariaDB: "Can't find FULLTEXT index matching the column list"
ER_BAD_FIELD_ERROR = 1054  # MariaDB: "Unknown column ..." (a database ingested with an older schema)

#: The strategy that runs without filters; a filter given with it is a 422.
STRATEGY_NO_FILTERS = "none"

#: The accepted values of the ``strategy`` parameter: a ``Literal`` over ``search.STRATEGIES``,
#: so a strategy added to the search module is accepted (and listed in /docs) without a change.
Strategy = Literal[STRATEGIES]  # type: ignore[valid-type]

#: The ``/api/search`` filter parameters as (query parameter name, ``search.Filters`` field).
API_FILTER_FIELDS: tuple[tuple[str, str], ...] = (
    ("min_words", "min_words"),
    ("max_words", "max_words"),
    ("heading", "heading_like"),
    ("path", "path_like"),
    ("linked_from", "linked_from"),
    ("links_to", "links_to"),
    ("titles", "titles"),
    ("category", "category"),
    ("edited_since", "edited_since"),
    ("edited_before", "edited_before"),
)

#: One line per known strategy for the page's select box; an unknown one shows its name.
STRATEGY_LABELS: dict[str, str] = {
    "inline": "inline: filters inside the index-driven statement (always k rows)",
    "overfetch": "overfetch: filter k x overfetch index candidates (bounded, may return fewer)",
    "none": "none: plain nearest neighbours, no filters allowed",
    "rrf": "rrf: vector top-N fused with a full-text top-N by reciprocal rank",
}


class QueryEmbedder(Protocol):
    """What the app needs from an embedder (``embedding.Embedder`` or a test double)."""

    def embed_queries(
        self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)`` with unit rows."""


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>WikiLense</title>
<style>
  :root {
    --fg: #1b1b1b; --bg: #fafafa; --muted: #5f6368; --line: #d9d9d9;
    --accent: #1f5fa8; --panel: #ffffff;
    --page-bg: #fff1c2; --page-fg: #5c4400; --evidence-bg: #d6f3dc; --evidence-fg: #0f5a1e;
    --mark: #fff09a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fg: #e8e8e8; --bg: #141414; --muted: #a0a0a0; --line: #333333;
      --accent: #8ab8f0; --panel: #1e1e1e;
      --page-bg: #4a3b00; --page-fg: #ffe08a; --evidence-bg: #12391b; --evidence-fg: #9be3a8;
      --mark: #5a4a00;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body {
    margin: 0 auto; max-width: 64rem; padding: 1.5rem 1rem 3rem;
    font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--fg); background: var(--bg);
  }
  h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
  .sub { margin: 0 0 1.25rem; color: var(--muted); }
  a { color: var(--accent); }
  code, pre { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .9em; }
  form {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: .75rem 1rem;
    padding: 1rem; background: var(--panel); border: 1px solid var(--line); border-radius: .5rem;
  }
  form label { display: flex; flex-direction: column; font-size: .85rem; color: var(--muted); }
  .wide { grid-column: 1 / -1; }
  form label.check { flex-direction: row; align-items: center; gap: .5rem; align-self: end; }
  form input, form select, form textarea {
    margin-top: .25rem; padding: .45rem .55rem; font: inherit; color: var(--fg);
    background: var(--bg); border: 1px solid var(--line); border-radius: .35rem;
  }
  form textarea { resize: vertical; }
  form label.check input { margin: 0; }
  form button {
    align-self: end; padding: .5rem 1rem; font: inherit; font-weight: 600; color: #fff;
    background: var(--accent); border: 0; border-radius: .35rem; cursor: pointer;
  }
  .note { margin: 0; color: var(--muted); font-size: .9rem; }
  #claim-list { width: 100%; font-size: .85rem; }
  #free-note details { margin-top: .25rem; }
  #page-list { max-height: 14rem; overflow: auto; columns: 14rem; margin: .5rem 0 0;
               font-size: .85rem; }
  .buttons { display: flex; flex-wrap: wrap; gap: .5rem; margin: .5rem 0 1rem; }
  a.example { padding: .3rem .8rem; border: 1px solid var(--accent); border-radius: 1rem;
              font-size: .9rem; text-decoration: none; }
  #claim-box { margin: 1rem 0; padding: .75rem 1rem; background: var(--panel);
               border: 1px solid var(--line); border-left: 4px solid var(--accent);
               border-radius: .5rem; }
  #claim-box h2 { font-size: 1.05rem; margin: 0 0 .25rem; }
  #claim-box p { margin: .25rem 0; }
  .set-head { font-weight: 600; }
  ul.evidence { margin: .25rem 0 .5rem; padding-left: 1.25rem; font-size: .9rem; }
  #summary { font-weight: 600; }
  .badge { padding: 0 .45rem; border-radius: .6rem; font-size: .75rem; font-weight: 600;
           background: var(--page-bg); color: var(--page-fg); }
  .badge.evidence { background: var(--evidence-bg); color: var(--evidence-fg); }
  mark { background: var(--mark); color: inherit; }
  #notice { margin: 0 0 1rem; padding: .5rem .75rem; border: 1px solid var(--accent);
            border-radius: .35rem; background: var(--panel); }
  #status { min-height: 1.5rem; color: var(--muted); }
  #hits { padding-left: 1.5rem; }
  #hits li { margin: 0 0 1.25rem; }
  .head { display: flex; flex-wrap: wrap; gap: .5rem; align-items: baseline; }
  .dist { font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--accent); }
  .path, .meta { color: var(--muted); font-size: .9rem; }
  .text { margin: .25rem 0 0; }
  ul.sentences { margin: .35rem 0 0; padding-left: 1rem; font-size: .9rem; }
  #hits ul.sentences li { margin: 0 0 .15rem; }
  ul.sentences code { color: var(--muted); }
  details { margin-top: 1.5rem; }
  summary { cursor: pointer; font-weight: 600; }
  pre { white-space: pre-wrap; word-break: break-word; padding: .75rem; background: var(--panel);
        border: 1px solid var(--line); border-radius: .35rem; }
  table { border-collapse: collapse; font-size: .85rem; width: 100%; }
  th, td { text-align: left; padding: .25rem .5rem; border-bottom: 1px solid var(--line);
           vertical-align: top; }
  footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line);
           font-size: .9rem; color: var(--muted); }
</style>
</head>
<body>
<h1>WikiLense</h1>
<p class="sub">Semantic search over Wikipedia inside MariaDB: <code>VEC_DISTANCE_COSINE</code>
on a <code>VECTOR INDEX</code>, combined with SQL filters in the same statement. Every filter
keeps only chunks whose page or section passes; filters combine with AND.</p>

<section id="examples" hidden>
  <p class="note">__EXAMPLES_NOTE__</p>
  <div id="example-buttons" class="buttons"></div>
</section>
<p id="notice" role="alert" hidden></p>

<form id="search-form">
  <label class="wide" id="database-field" hidden title="The database to search. The free-text note, the page list, the claim picker and the marks follow it.">Database
    <select name="database" id="database"></select></label>
  <p class="note wide" id="s1-note" hidden></p>
  <div class="wide" id="picker" hidden>
    <label title="A FEVEROUS benchmark claim: the search runs on its text, and the hits are marked against its gold evidence.">Benchmark claim (optional; search by text or id)
      <input id="claim-search" type="search" autocomplete="off" placeholder="claim text or id"></label>
    <select id="claim-list" size="5" aria-label="matching claims"></select>
    <p class="note" id="claim-count"></p>
    <input type="hidden" name="claim_id" id="claim-id">
  </div>
  <label class="wide">Query
    <input name="q" required autofocus placeholder="Aare river in Switzerland"></label>
  <div class="wide note" id="free-note">
    <p>Free-text search: there is no answer key, so the page cannot tell whether a hit is right.
    The hits are the closest passages among the <span id="page-count">pages</span> of this
    database, not all of Wikipedia. A question about something outside them still returns its
    nearest passages, which may look unrelated.</p>
    <details id="page-box"><summary>Which pages does it hold?</summary>
      <input id="page-filter" type="search" placeholder="filter the titles" aria-label="filter the page titles">
      <p class="note" id="page-count-line"></p>
      <ol id="page-list"></ol></details>
  </div>
  <label title="Number of hits to return (1 to __MAX_K__).">k (hits)
    <input name="k" type="number" value="__DEFAULT_K__" min="1" max="__MAX_K__"></label>
  <label title="Keep only chunks of pages with at least N words (page.n_words >= N).">Min words on page
    <input name="min_words" type="number" min="0" placeholder="any"></label>
  <label title="Keep only chunks of pages with at most N words (page.n_words <= N).">Max words on page
    <input name="max_words" type="number" min="0" placeholder="any"></label>
  <label title="Keep only chunks whose section heading matches this SQL LIKE pattern (case- and accent-insensitive).">Heading LIKE
    <input name="heading" placeholder="%History%"></label>
  <label title="Keep only chunks whose section path ('Heading > Subheading') matches this SQL LIKE pattern.">Section path LIKE
    <input name="path" placeholder="Geography%"></label>
  <label title="Keep only chunks of pages that this page links to (exact title).">Linked from page
    <input name="linked_from" placeholder="exact page title"></label>
  <label title="Keep only chunks of pages that link to this page (exact title).">Links to page
    <input name="links_to" placeholder="exact page title"></label>
  <label class="wide" title="Keep only chunks of these pages (exact titles, one per line; sent as one titles parameter per line).">Only these pages
    <textarea name="titles" rows="2" placeholder="Aare&#10;Bern"></textarea></label>
  <label title="Keep only chunks of pages in this Wikipedia category (exact name, without 'Category:').">Category
    <input name="category" placeholder="Rivers of Switzerland"></label>
  <label title="Keep only chunks of pages whose stored revision was edited on or after this day (UTC).">Edited since
    <input name="edited_since" type="date"></label>
  <label title="Keep only chunks of pages whose stored revision was edited before this day (UTC).">Edited before
    <input name="edited_before" type="date"></label>
  <label title="How the filters are applied; see the option text.">Strategy
    <select name="strategy">
__STRATEGY_OPTIONS__
    </select></label>
  <label title="overfetch and rrf only: the index returns k x this many candidates before the filters run.">Overfetch factor
    <input name="overfetch" type="number" min="1" placeholder="__DEFAULT_OVERFETCH__"></label>
  <label title="Candidates the HNSW index keeps while searching (more = closer to the exact ranking, slower); 0 leaves the server's session value.">mhnsw_ef_search
    <input name="ef_search" type="number" min="0" placeholder="__EF_SEARCH_DEFAULT__ (default); 0 = server value"></label>
  <label class="check" title="Also list, under each hit, the sentences of the chunk: chunk_sentence joined to sentence, in page order.">
    <input name="sentences" type="checkbox" value="1"> show the chunk's sentences</label>
  <label class="check" title="Run the search statement __PAGE_REPEAT__ times on the same connection, one after another, and show the median SQL time and the range; the query is embedded once.">
    <input name="repeat" type="checkbox" value="__PAGE_REPEAT__"> <span id="repeat-text">repeat __PAGE_REPEAT__ times</span></label>
  <button type="submit">Search</button>
</form>

<section id="claim-box" hidden>
  <h2 id="claim-head"></h2>
  <p id="claim-text"></p>
  <p class="meta" id="claim-source"></p>
  <div id="claim-evidence"></div>
  <p id="summary"></p>
</section>
<p id="status"></p>
<ol id="hits"></ol>
<details id="sql-box" hidden>
  <summary>SQL that ran and its EXPLAIN</summary>
  <pre id="sql"></pre>
  <table id="explain"></table>
</details>

<footer>
<p>The form calls <code>GET /api/search</code> with the same parameter names (<code>sentences=1</code>
for the sentence list, <code>titles</code> once per title, <code>claim_id</code> for a picked
claim, <code>repeat</code> for the repeated timing, <code>database</code> for the database); an
unknown parameter is rejected with 400. <code>GET /api/databases</code>,
<code>/api/claims</code> and <code>/api/pages</code> list the databases, the claims and the
page titles. A picked claim's marks and summary use the evaluation's definitions and are
computed after the timed statement. The interactive API
documentation is left on so the API can be tried from the browser:
<a href="__DOCS_URL__">__DOCS_URL__</a> (OpenAPI schema at <a href="__OPENAPI_URL__">__OPENAPI_URL__</a>).
Model load, query embedding and SQL time are reported separately in the status line.</p>
</footer>

<script>
(function () {
  var MAX_LISTED_CLAIMS = 50;
  var MAX_LISTED_PAGES = 1000;
  var MAX_REPEAT = __MAX_REPEAT__;
  var form = document.getElementById('search-form');
  var status = document.getElementById('status');
  var list = document.getElementById('hits');
  var box = document.getElementById('sql-box');
  var sqlEl = document.getElementById('sql');
  var explainEl = document.getElementById('explain');
  var query = form.elements.namedItem('q');
  var dbField = document.getElementById('database-field');
  var dbSelect = document.getElementById('database');
  var s1Note = document.getElementById('s1-note');
  var picker = document.getElementById('picker');
  var claimSearch = document.getElementById('claim-search');
  var claimList = document.getElementById('claim-list');
  var claimCount = document.getElementById('claim-count');
  var claimIdField = document.getElementById('claim-id');
  var freeNote = document.getElementById('free-note');
  var pageCount = document.getElementById('page-count');
  var pageBox = document.getElementById('page-box');
  var pageFilter = document.getElementById('page-filter');
  var pageCountLine = document.getElementById('page-count-line');
  var pageList = document.getElementById('page-list');
  var claimBox = document.getElementById('claim-box');
  var examplesBox = document.getElementById('examples');
  var exampleButtons = document.getElementById('example-buttons');
  var noticeEl = document.getElementById('notice');

  var databases = [];
  var examples = [];
  var s1Name = null;
  var claims = [];
  var claimsById = {};
  var picked = null;
  var lastBody = null;
  var pageTitles = null;

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function fmt(n) { return Number(n).toLocaleString('en-US'); }

  // A message about the link or the database choice, shown above the form where it is seen.
  function notice(text) {
    noticeEl.textContent = text;
    noticeEl.hidden = false;
  }

  function describe(detail) {
    if (Array.isArray(detail)) {
      return detail.map(function (d) {
        return (d.loc || []).slice(1).join('.') + ': ' + d.msg;
      }).join('; ');
    }
    return String(detail);
  }

  function getJSON(url) {
    return fetch(url).then(function (resp) {
      return resp.json().then(function (body) {
        if (!resp.ok) throw new Error('Error ' + resp.status + ': ' + describe(body.detail));
        return body;
      });
    });
  }

  function selectedDatabase() { return dbSelect.value; }

  function currentDatabase() {
    for (var i = 0; i < databases.length; i++) {
      if (databases[i].name === selectedDatabase()) return databases[i];
    }
    return null;
  }

  function dbQuery(path) {
    var name = selectedDatabase();
    return name ? path + '?database=' + encodeURIComponent(name) : path;
  }

  // --- the free-text note and the page list ------------------------------------------------

  function showFreeNote() { freeNote.hidden = picked !== null; }

  function updatePageCount() {
    var d = currentDatabase();
    pageCount.textContent = d ? fmt(d.n_pages) + ' pages' : 'pages';
  }

  function renderPages() {
    pageList.replaceChildren();
    if (pageTitles === null) return;
    var term = pageFilter.value.trim().toLowerCase();
    var shown = 0, matching = 0;
    pageTitles.forEach(function (title) {
      if (term && title.toLowerCase().indexOf(term) === -1) return;
      matching++;
      if (shown < MAX_LISTED_PAGES) { pageList.appendChild(el('li', null, title)); shown++; }
    });
    pageCountLine.textContent = fmt(matching) + ' of ' + fmt(pageTitles.length) + ' titles' +
      (shown < matching ? '; the first ' + fmt(shown) + ' are listed, filter to narrow' : '');
  }

  function loadPages() {
    getJSON(dbQuery('/api/pages')).then(function (body) {
      pageTitles = body.titles;
      renderPages();
    }).catch(function (err) { pageCountLine.textContent = String(err.message || err); });
  }

  pageBox.addEventListener('toggle', function () {
    if (pageBox.open && pageTitles === null) loadPages();
  });
  pageFilter.addEventListener('input', renderPages);

  // --- the claim picker ----------------------------------------------------------------------

  function renderClaimList() {
    var term = claimSearch.value.trim().toLowerCase();
    var matching = 0;
    claimList.replaceChildren();
    claims.forEach(function (c) {
      if (term && String(c.id).indexOf(term) !== 0 && c.text.toLowerCase().indexOf(term) === -1) return;
      matching++;
      if (matching <= MAX_LISTED_CLAIMS) {
        claimList.appendChild(new Option(c.id + ' \\u00b7 ' + c.label + ' \\u00b7 ' + c.text, String(c.id)));
      }
    });
    claimCount.textContent = fmt(claims.length) + ' claims offered on this database, ' +
      fmt(matching) + ' matching' +
      (matching > MAX_LISTED_CLAIMS ? '; the first ' + MAX_LISTED_CLAIMS + ' are listed' : '') +
      (picked ? '. Picked: claim ' + picked.id + '.' : '.');
  }

  function pickClaim(c) {
    picked = c;
    claimIdField.value = String(c.id);
    query.value = c.text;
    showFreeNote();
    renderClaimList();
  }

  function formParams() {
    var params = new URLSearchParams();
    new FormData(form).forEach(function (value, name) {
      if (name === 'titles') {
        value.split('\\n').forEach(function (title) {
          title = title.trim();
          if (title !== '') params.append('titles', title);
        });
      } else if (value !== '') {
        params.set(name, value);
      }
    });
    return params;
  }

  // The address bar always holds the form, so a search can be bookmarked or sent to someone.
  function updateUrl(params) {
    var text = params.toString();
    window.history.replaceState(null, '', text ? '/?' + text : '/');
  }

  function dropClaim() {
    picked = null;
    claimIdField.value = '';
    claimBox.hidden = true;
    showFreeNote();
    renderClaimList();
    if (lastBody) {
      lastBody.marks = undefined;
      renderHits(lastBody);
    }
    updateUrl(formParams());
  }

  claimSearch.addEventListener('input', renderClaimList);
  claimList.addEventListener('change', function () {
    var c = claimsById[claimList.value];
    if (c) { pickClaim(c); form.requestSubmit(); }
  });
  // Editing the query text drops the picked claim: its marks belong to its own text only.
  query.addEventListener('input', function () {
    if (picked && query.value !== picked.text) dropClaim();
  });

  function loadClaims() {
    return getJSON(dbQuery('/api/claims')).then(function (body) {
      claims = body.claims;
      claimsById = {};
      claims.forEach(function (c) { claimsById[String(c.id)] = c; });
      picker.hidden = claims.length === 0;
      if (picked && !claimsById[String(picked.id)]) {
        var gone = picked.id;
        dropClaim();
        notice('Claim ' + gone + ' is not offered on ' + body.database + '; it was dropped.');
      }
      renderClaimList();
      renderExamples();
    });
  }

  // --- the ready-made examples ---------------------------------------------------------------

  function renderExamples() {
    exampleButtons.replaceChildren();
    var name = selectedDatabase();
    examples.forEach(function (ex) {
      if (ex.s1_only && name !== s1Name) return;
      if (ex.params.claim_id !== undefined && !claimsById[ex.params.claim_id]) return;
      var params = new URLSearchParams();
      if (name) params.set('database', name);
      Object.keys(ex.params).forEach(function (key) { params.set(key, ex.params[key]); });
      var link = el('a', 'example', ex.label);
      link.href = '/?' + params.toString();
      exampleButtons.appendChild(link);
    });
    examplesBox.hidden = exampleButtons.childNodes.length === 0;
  }

  // --- the results ---------------------------------------------------------------------------

  function renderExplain(rows) {
    explainEl.replaceChildren();
    if (!rows.length) return;
    var cols = Object.keys(rows[0]);
    var head = el('tr');
    cols.forEach(function (c) { head.appendChild(el('th', null, c)); });
    explainEl.appendChild(head);
    rows.forEach(function (row) {
      var tr = el('tr');
      cols.forEach(function (c) {
        tr.appendChild(el('td', null, row[c] === null ? 'NULL' : String(row[c])));
      });
      explainEl.appendChild(tr);
    });
  }

  function renderSentences(rows, goldKeys) {
    var ul = el('ul', 'sentences');
    if (!rows.length) {
      ul.appendChild(el('li', null, 'no chunk_sentence rows'));
      return ul;
    }
    rows.forEach(function (s) {
      var li = el('li');
      li.appendChild(el('code', null, s.element_key));
      li.appendChild(document.createTextNode(' '));
      if (goldKeys.indexOf(s.element_key) !== -1) li.appendChild(el('mark', null, s.text));
      else li.appendChild(document.createTextNode(s.text));
      ul.appendChild(li);
    });
    return ul;
  }

  function renderHits(body) {
    list.replaceChildren();
    body.hits.forEach(function (h, i) {
      var mark = body.marks ? body.marks[i] : null;
      var li = el('li');
      var head = el('div', 'head');
      head.appendChild(el('span', 'dist', h.distance.toFixed(4)));
      head.appendChild(el('strong', null, h.title));
      if (h.section_path) head.appendChild(el('span', 'path', '> ' + h.section_path));
      var meta = '(chunk_id ' + h.chunk_id + ', ' + h.n_words + ' words';
      if (h.score !== undefined && h.score !== null) meta += ', rrf score ' + h.score.toFixed(4);
      head.appendChild(el('span', 'meta', meta + ')'));
      if (mark && mark.gold_page) head.appendChild(el('span', 'badge', 'gold page'));
      if (mark && mark.gold_evidence) head.appendChild(el('span', 'badge evidence', 'gold evidence'));
      li.appendChild(head);
      li.appendChild(el('p', 'text', h.text));
      if (h.sentences) li.appendChild(renderSentences(h.sentences, mark ? mark.gold_sentence_keys : []));
      list.appendChild(li);
    });
  }

  function summaryText(s) {
    var parts = [];
    parts.push('At k = ' + s.k + ': ' + (s.gold_page_found ?
      'gold page found, first at rank ' + s.gold_page_rank + '.' :
      'gold page not found in the ' + s.n_hits + (s.n_hits === 1 ? ' hit.' : ' hits.')));
    if (!s.evidence_recall_applies) {
      parts.push('Evidence recall does not apply: evidence in a table (not indexed).');
    } else {
      parts.push('Evidence recall: ' + (s.evidence_recall ? 'yes' : 'no') + ', ' +
        s.units_covered + ' of ' + s.units_total + ' gold units covered' +
        (s.n_evidence_sets > 1 ? ' (the best-covered evidence set).' : '.'));
      if (s.table_sets) {
        parts.push(s.table_sets + (s.table_sets === 1 ? ' other evidence set is' : ' other evidence sets are') +
          ' in a table (not indexed).');
      }
    }
    parts.push('Definitions as in the evaluation (docs/EVALUATION.md).');
    return parts.join(' ');
  }

  function renderClaim(body) {
    if (!body.claim) { claimBox.hidden = true; return; }
    var c = body.claim;
    document.getElementById('claim-head').textContent =
      'Claim ' + c.id + ' \\u00b7 ' + c.label + ' \\u00b7 needs ' + c.kind_text;
    document.getElementById('claim-text').textContent = c.text;
    document.getElementById('claim-source').textContent =
      'From ' + c.sources.join(' and ') + ' (FEVEROUS ' + c.split + ')';
    var evidence = document.getElementById('claim-evidence');
    evidence.replaceChildren();
    c.evidence_sets.forEach(function (set, i) {
      var title = 'Gold evidence' +
        (c.evidence_sets.length > 1 ? ', set ' + (i + 1) + ' of ' + c.evidence_sets.length : '') +
        (set.sentence_only ? '' : ': evidence in a table (not indexed)');
      evidence.appendChild(el('p', 'set-head', title));
      var ul = el('ul', 'evidence');
      set.elements.forEach(function (e) {
        var li = el('li');
        li.appendChild(el('strong', null, e.page));
        li.appendChild(document.createTextNode(e.in_table ?
          ': ' + e.element_key + ' (a table element, not indexed)' : ': ' + e.text));
        ul.appendChild(li);
      });
      evidence.appendChild(ul);
    });
    document.getElementById('summary').textContent = summaryText(body.summary);
    claimBox.hidden = false;
  }

  function render(body) {
    lastBody = body;
    renderClaim(body);
    renderHits(body);
    sqlEl.textContent = body.sql;
    renderExplain(body.explain);
    box.hidden = false;
    var t = body.timing_ms;
    var loaded = t.model_load === null ? 'model loaded lazily' :
      'model loaded at start-up in ' + t.model_load.toFixed(0) + ' ms';
    var sql = 'SQL ' + t.sql.toFixed(1) + ' ms';
    if (body.repeat) {
      var r = body.repeat;
      sql = 'SQL median ' + r.median_ms.toFixed(1) + ' ms, range ' + r.min_ms.toFixed(1) +
        '\\u2013' + r.max_ms.toFixed(1) + ' ms over ' + r.runs + ' runs (query embedded once)';
      if (!r.same_hits) {
        sql += '; run ' + r.differing_runs.join(', ') + ' returned other hits than run 1, shown here';
      }
    }
    status.textContent = body.hits.length + (body.hits.length === 1 ? ' hit' : ' hits') +
      (body.database ? ' from ' + body.database : '') +
      '; embedding ' + t.embed.toFixed(1) + ' ms, ' + sql +
      '; mhnsw_ef_search ' + body.ef_search + ' (' + loaded + ')';
  }

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    var params = formParams();
    updateUrl(params);
    status.textContent = 'Searching...';
    list.replaceChildren();
    box.hidden = true;
    claimBox.hidden = true;
    fetch('/api/search?' + params.toString()).then(function (resp) {
      return resp.json().then(function (body) {
        if (!resp.ok) {
          status.textContent = 'Error ' + resp.status + ': ' + describe(body.detail);
          return;
        }
        render(body);
      });
    }).catch(function (err) {
      status.textContent = 'Request failed: ' + err;
    });
  });

  // The page count, the page list, the claims and the results follow the database: a search
  // already shown runs again on the new database (a claim it does not offer is dropped first).
  dbSelect.addEventListener('change', function () {
    updatePageCount();
    pageTitles = null;
    renderPages();
    if (pageBox.open) loadPages();
    noticeEl.hidden = true;
    var shown = lastBody !== null;
    lastBody = null;
    list.replaceChildren();
    box.hidden = true;
    claimBox.hidden = true;
    status.textContent = '';
    loadClaims().then(function () {
      if (shown && query.value.trim() !== '') form.requestSubmit();
      else updateUrl(formParams());
    }).catch(function (err) { notice(String(err.message || err)); });
  });

  // --- a link with the form's parameters fills the form and runs the search ----------------
  // (/?q=...&k=3&category=..., /?database=...&claim_id=...&repeat=5), so a search can be
  // bookmarked or sent to someone; an unknown claim or database is reported, never replaced.

  var initial = new URLSearchParams(window.location.search);

  function applyUrl() {
    initial.forEach(function (value, name) {
      if (name === 'database' || name === 'claim_id') return;
      var field = form.elements.namedItem(name);
      if (!field) return;
      if (name === 'titles') field.value = initial.getAll('titles').join('\\n');
      else if (name === 'repeat') applyRepeat(field, value);
      else if (field.type === 'checkbox') field.checked = value === field.value;
      else field.value = value;
    });
  }

  // A link may ask for any repeat count the API accepts; the checkbox then carries that count.
  function applyRepeat(field, value) {
    var n = Number(value);
    if (Number.isInteger(n) && n >= 1 && n <= MAX_REPEAT) {
      field.value = String(n);
      field.checked = true;
      document.getElementById('repeat-text').textContent = 'repeat ' + n + ' times';
    } else {
      notice('repeat=' + value + ' in the link is not a whole number from 1 to ' + MAX_REPEAT +
        '; the search runs once.');
    }
  }

  function runFromUrl() {
    var wantedClaim = initial.get('claim_id');
    if (wantedClaim !== null) {
      var c = claimsById[wantedClaim];
      if (!c) {
        notice('Claim ' + wantedClaim + ' is not offered on database ' +
          (selectedDatabase() || 'of this server') + ' (it is in neither claim file, or its ' +
          'gold pages are not all in this database); nothing was searched.');
        return;
      }
      applyUrl();
      pickClaim(c);
      form.requestSubmit();
    } else if (initial.get('q')) {
      applyUrl();
      form.requestSubmit();
    }
  }

  getJSON('/api/databases').then(function (body) {
    databases = body.databases;
    examples = body.examples;
    s1Name = body.s1;
    dbSelect.replaceChildren();
    databases.forEach(function (d) { dbSelect.appendChild(new Option(d.label, d.name)); });
    dbField.hidden = false;
    if (body.s1_note) { s1Note.textContent = body.s1_note; s1Note.hidden = false; }
    var wanted = initial.get('database');
    if (wanted !== null && !databases.some(function (d) { return d.name === wanted; })) {
      notice('Database "' + wanted + '" is not offered on this server (offered: ' +
        databases.map(function (d) { return d.name; }).join(', ') + '); nothing was searched.');
      updatePageCount();
      return loadClaims();
    }
    if (wanted) dbSelect.value = wanted;
    updatePageCount();
    return loadClaims().then(runFromUrl);
  }).catch(function (err) {
    // Without the database list the page still runs a free-text link, as the API allows.
    notice(String(err.message || err));
    if (initial.get('claim_id') === null && initial.get('q')) {
      var wanted = initial.get('database');
      if (wanted) dbSelect.replaceChildren(new Option(wanted, wanted));  // the API checks it
      applyUrl();
      form.requestSubmit();
    }
  });
})();
</script>
</body>
</html>
"""


def render_index(settings: Settings) -> str:
    """Return the query page: :data:`INDEX_HTML` with the strategy options and the defaults filled in.

    The ``<select>`` lists ``search.STRATEGIES`` in order (labels from
    :data:`STRATEGY_LABELS`); the ``ef_search`` placeholder shows ``settings.ef_search``.
    """
    options = "\n".join(
        f'      <option value="{html.escape(name)}">'
        f"{html.escape(STRATEGY_LABELS.get(name, name))}</option>"
        for name in STRATEGIES
    )
    replacements = {
        "__STRATEGY_OPTIONS__": options,
        "__EF_SEARCH_DEFAULT__": str(settings.ef_search),
        "__DEFAULT_K__": str(DEFAULT_K),
        "__MAX_K__": str(MAX_K),
        "__DEFAULT_OVERFETCH__": str(DEFAULT_OVERFETCH),
        "__DOCS_URL__": DOCS_URL,
        "__OPENAPI_URL__": OPENAPI_URL,
        "__PAGE_REPEAT__": str(PAGE_REPEAT),
        "__MAX_REPEAT__": str(MAX_REPEAT),
        "__EXAMPLES_NOTE__": html.escape(EXAMPLES_NOTE),
    }
    page = INDEX_HTML
    for marker, value in replacements.items():
        page = page.replace(marker, value)
    return page


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _chunk_count(conn: pymysql.Connection) -> int:
    """Return the number of rows in ``chunk``."""
    with conn.cursor() as cur:
        cur.execute(CHUNK_COUNT_SQL)
        return int(cur.fetchone()[0])


def _open_connection(settings: Settings, database: str | None) -> pymysql.Connection:
    """Return a connection to ``database`` (default ``settings.db_name``); 503 when it fails."""
    try:
        return db.connect(settings, database=database)
    except pymysql.err.OperationalError as exc:
        message = exc.args[1] if len(exc.args) >= 2 else str(exc)
        raise HTTPException(
            status_code=503,
            detail=(
                f"cannot connect to MariaDB at {settings.db_host}:{settings.db_port}: {message}"
            ),
        ) from exc


def warm_up_embedder(embedder: QueryEmbedder) -> float:
    """Embed :data:`WARM_UP_TEXT` once and return the milliseconds it took (the model load).

    A model that cannot be loaded raises ``OSError`` from sentence-transformers; the caller
    (``cli.cmd_serve``) reports it.
    """
    start = time.perf_counter()
    embedder.embed_queries([WARM_UP_TEXT])
    return (time.perf_counter() - start) * 1000.0


# ---------------------------------------------------------------------------------------------
# the database choice and the claim catalogs (cached on app.state, built under its lock)
# ---------------------------------------------------------------------------------------------

_UNPROBED = object()


def demo_database(state: Any) -> str:
    """Return the name of the demo database: the app's ``database``, else ``settings.db_name``."""
    return state.database or state.settings.db_name


def _db_error_code(exc: BaseException) -> int | None:
    """Return the MariaDB error number of a PyMySQL error, or None."""
    return exc.args[0] if exc.args and isinstance(exc.args[0], int) else None


def probe_s1(settings: Settings) -> tuple[int, int] | None:
    """Return ``(pages, chunks)`` of :data:`SCALED_DATABASE`, or None when it is not offered.

    None means the database does not exist (or this user may not open it), has no schema, or
    holds no chunks. A server that cannot be reached raises ``pymysql.err.OperationalError``.
    """
    try:
        conn = db.connect(settings, database=SCALED_DATABASE)
    except pymysql.err.OperationalError as exc:
        if _db_error_code(exc) in (ER_DBACCESS_DENIED, ER_BAD_DB):
            return None
        raise
    try:
        n_pages, n_chunks = picker.database_size(conn)
    except pymysql.err.ProgrammingError as exc:
        if _db_error_code(exc) == ER_NO_SUCH_TABLE:
            return None
        raise
    finally:
        conn.close()
    return (n_pages, n_chunks) if n_chunks > 0 else None


def _s1_size(state: Any) -> tuple[int, int] | None:
    """Return S1's size when the app offers it (probed once, then cached), else None."""
    if not state.offer_s1 or demo_database(state) == SCALED_DATABASE:
        return None
    with state.lock:
        if state.s1_probe is _UNPROBED:
            try:
                state.s1_probe = probe_s1(state.settings)
            except pymysql.err.OperationalError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"cannot check database {SCALED_DATABASE} on MariaDB at "
                    f"{state.settings.db_host}:{state.settings.db_port}: {exc}",
                ) from exc
        return state.s1_probe


def offered_names(state: Any) -> list[str]:
    """Return the database names the app offers: the demo, then S1 when it is loaded."""
    names = [demo_database(state)]
    if _s1_size(state) is not None:
        names.append(SCALED_DATABASE)
    return names


def choose_database(state: Any, requested: str | None) -> str:
    """Return the database a request uses: ``requested`` when offered, the demo for None; else 422.

    The demo needs no check; S1 is offered only after :func:`probe_s1` found it loaded; any other
    name is refused without a connection.
    """
    demo = demo_database(state)
    if requested is None or requested == demo:
        return demo
    if requested == SCALED_DATABASE and _s1_size(state) is not None:
        return requested
    raise HTTPException(
        status_code=422,
        detail=f"database {requested!r} is not offered by this server; offered: "
        + ", ".join(offered_names(state)),
    )


def catalog(state: Any, name: str) -> picker.Catalog:
    """Return the claim catalog of an offered database, built at its first use and then cached.

    The claim files are read once per app. The catalog reflects the database when it was built,
    so a database ingested again needs a restart of the server.
    """
    with state.lock:
        cached = state.catalogs.get(name)
        if cached is not None:
            return cached
        if state.claims is None:
            try:
                state.claims = picker.read_claim_files(state.claim_files)
            except picker.ClaimFileError as exc:
                raise HTTPException(
                    status_code=503, detail=f"the claim files cannot be read: {exc}"
                ) from exc
        conn = _open_connection(state.settings, name)
        try:
            try:
                built = picker.build_catalog(conn, name, state.claims)
            except pymysql.err.ProgrammingError as exc:
                if _db_error_code(exc) == ER_NO_SUCH_TABLE:
                    raise HTTPException(
                        status_code=503,
                        detail=f"the database {name} has no schema or an older one: run "
                        "'wikilense ingest'",
                    ) from exc
                raise
        finally:
            conn.close()
        state.catalogs[name] = built
        return built


@dataclass(frozen=True)
class DatabaseChoice:
    """One entry of the page's database selector."""

    name: str
    label: str
    n_pages: int
    n_chunks: int


def database_label(name: str, n_pages: int, n_chunks: int) -> str:
    """Return the selector label, e.g. ``Demo: 100 pages, 4,598 chunks``."""
    kind = "Scaled run (S1)" if name == SCALED_DATABASE else "Demo"
    return f"{kind}: {n_pages:,} pages, {n_chunks:,} chunks"


def database_size(state: Any, name: str) -> tuple[int, int]:
    """Return ``(pages, chunks)`` of an offered database, read once and then cached."""
    with state.lock:
        cached = state.sizes.get(name)
        if cached is not None:
            return cached
        conn = _open_connection(state.settings, name)
        try:
            try:
                size = picker.database_size(conn)
            except pymysql.err.ProgrammingError as exc:
                if _db_error_code(exc) == ER_NO_SUCH_TABLE:
                    raise HTTPException(
                        status_code=503,
                        detail=f"the database {name} has no schema: run 'wikilense ingest'",
                    ) from exc
                raise
        finally:
            conn.close()
        state.sizes[name] = size
        return size


def offered_databases(state: Any) -> list[DatabaseChoice]:
    """Return the offered databases with their sizes, read from each database.

    The sizes do not depend on the claim catalogs, so a claim file that cannot be read leaves
    the database choice and free-text search working.
    """
    choices = []
    for name in offered_names(state):
        n_pages, n_chunks = database_size(state, name)
        choices.append(
            DatabaseChoice(name, database_label(name, n_pages, n_chunks), n_pages, n_chunks)
        )
    return choices


def prepare(app: FastAPI) -> list[str]:
    """Probe S1 and build every offered database's claim catalog now; return one line per database.

    ``wikilense serve`` calls this at start-up, so the membership check runs before the first
    request; without it every catalog is built at its first use. An ``HTTPException`` (the
    server, or a database without schema) propagates.
    """
    lines = []
    for choice in offered_databases(app.state):
        cat = catalog(app.state, choice.name)
        own = ", ".join(cat.own_files) if cat.own_files else "none (the claim picker is hidden)"
        lines.append(
            f"{choice.label} ({choice.name}); claim file: {own}; {len(cat.claims):,} claims "
            f"offered, {len(cat.dropped)} left out"
        )
    if app.state.offer_s1 and SCALED_DATABASE not in offered_names(app.state):
        lines.append(f"{SCALED_DATABASE} is not loaded (make fetch-s1, then make load-s1)")
    return lines


# ---------------------------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------------------------


def index(request: Request) -> str:
    """Return the query page for the app's settings."""
    return render_index(request.app.state.settings)


def api_search(
    request: Request,
    q: Annotated[str, Query(min_length=1, description="the query text")],
    k: Annotated[int, Query(ge=1, le=MAX_K, description="number of hits")] = DEFAULT_K,
    min_words: Annotated[
        int | None,
        Query(ge=0, description="keep only chunks of pages with at least N words (page.n_words >= N)"),
    ] = None,
    max_words: Annotated[
        int | None,
        Query(ge=0, description="keep only chunks of pages with at most N words (page.n_words <= N)"),
    ] = None,
    heading: Annotated[
        str | None,
        Query(
            description="keep only chunks whose section heading matches this SQL LIKE pattern, "
            "e.g. %History% (case- and accent-insensitive)"
        ),
    ] = None,
    path: Annotated[
        str | None,
        Query(
            description="keep only chunks whose section path ('Heading > Subheading') matches "
            "this SQL LIKE pattern, e.g. Geography%"
        ),
    ] = None,
    linked_from: Annotated[
        str | None,
        Query(description="keep only chunks of pages that the page with this exact title links to"),
    ] = None,
    links_to: Annotated[
        str | None,
        Query(description="keep only chunks of pages that link to the page with this exact title"),
    ] = None,
    titles: Annotated[
        list[str] | None,
        Query(
            description="keep only chunks of the pages with these exact titles; repeat the "
            "parameter once per title (titles=Aare&titles=Bern)"
        ),
    ] = None,
    category: Annotated[
        str | None,
        Query(
            description="keep only chunks of pages in the Wikipedia category with this exact "
            "name, without 'Category:' (the categories of the page's December 2020 revision)"
        ),
    ] = None,
    edited_since: Annotated[
        date | None,
        Query(
            description="keep only chunks of pages whose stored revision was edited on or after "
            "this day, YYYY-MM-DD in UTC (page.last_edited >= day)"
        ),
    ] = None,
    edited_before: Annotated[
        date | None,
        Query(
            description="keep only chunks of pages whose stored revision was edited before this "
            "day, YYYY-MM-DD in UTC (page.last_edited < day)"
        ),
    ] = None,
    strategy: Annotated[
        Strategy,
        Query(
            description="how the filters are applied: inline (in the index-driven statement, "
            "always k rows), overfetch (filter k x overfetch index candidates, may return "
            "fewer), none (no filters allowed), rrf (vector and full-text ranks fused)"
        ),
    ] = DEFAULT_STRATEGY,
    overfetch: Annotated[
        int,
        Query(
            ge=1,
            le=search.MAX_OVERFETCH,
            description="overfetch and rrf only: the index returns k x overfetch candidates "
            "before the filters run",
        ),
    ] = DEFAULT_OVERFETCH,
    ef_search: Annotated[
        int | None,
        Query(
            ge=search.EF_SEARCH_SERVER,
            le=search.MAX_EF_SEARCH,
            description="mhnsw_ef_search for this query (candidates the HNSW index keeps); "
            "default WIKILENSE_EF_SEARCH; 0 leaves the server's session value",
        ),
    ] = None,
    sentences: Annotated[
        bool,
        Query(
            description="also return, per hit, the sentences of the chunk (chunk_sentence "
            "joined to sentence, in page order)"
        ),
    ] = False,
    claim_id: Annotated[
        int | None,
        Query(
            ge=0,
            description="a benchmark claim offered on the database (GET /api/claims); q must "
            "be its text, and the response gains the claim, a mark per hit and the summary",
        ),
    ] = None,
    database: Annotated[
        str | None,
        Query(
            description="the database: the demo database (the default) or "
            f"{SCALED_DATABASE} when it is loaded; GET /api/databases lists the offered ones"
        ),
    ] = None,
    repeat: Annotated[
        int | None,
        Query(
            ge=1,
            le=MAX_REPEAT,
            description="run the search statement this many times on the same connection "
            "(the query is embedded once); the response gains every run's SQL time, the median "
            "and the range",
        ),
    ] = None,
) -> dict[str, Any]:
    """Return the hits for ``q`` with the SQL that ran, its EXPLAIN rows, ``ef_search`` and timings."""
    state = request.app.state
    settings: Settings = state.settings
    embedder: QueryEmbedder = state.embedder
    text = q.strip()
    if not text:
        raise HTTPException(status_code=400, detail="q must not be blank")
    # Without the parameter the app's database is used as before (None: settings.db_name).
    target: str | None = state.database if database is None else choose_database(state, database)
    offered: picker.OfferedClaim | None = None
    if claim_id is not None:
        name = target or settings.db_name
        offered = catalog(state, name).claims.get(claim_id)
        if offered is None:
            raise HTTPException(
                status_code=404,
                detail=f"claim {claim_id} is not offered on database {name}: it is in neither "
                "claim file, its gold pages are not all in this database, or GET /api/claims "
                "lists it as left out",
            )
        if text != offered.claim.text.strip():
            raise HTTPException(
                status_code=422,
                detail=f"q is not the text of claim {claim_id}; send its text as q, or leave "
                "out claim_id for a free-text search",
            )
    strategy_name: str = strategy
    # An empty string (a blank form field) is no filter; blank titles are dropped the same way.
    given_titles = [t.strip() for t in titles or [] if t.strip()]
    try:
        filters = Filters(
            min_words=min_words,
            max_words=max_words,
            heading_like=heading or None,
            path_like=path or None,
            linked_from=linked_from or None,
            links_to=links_to or None,
            titles=given_titles or None,
            category=category or None,
            edited_since=edited_since,
            edited_before=edited_before,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if strategy_name == STRATEGY_NO_FILTERS and not filters.is_empty():
        given = [
            name for name, field in API_FILTER_FIELDS if getattr(filters, field) is not None
        ]
        others = ", ".join(s for s in STRATEGIES if s != STRATEGY_NO_FILTERS)
        raise HTTPException(
            status_code=422,
            detail=f"strategy '{STRATEGY_NO_FILTERS}' runs without filters, but "
            f"{', '.join(given)} given: drop the filter or choose strategy {others}",
        )
    ef = search.resolve_ef_search(ef_search, settings.ef_search)

    start = time.perf_counter()
    try:
        qvec = embedder.embed_queries([text])[0]
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail=f"cannot load the embedding model: {exc}"
        ) from exc
    embed_ms = (time.perf_counter() - start) * 1000.0
    if qvec.shape != (db.VECTOR_DIM,):  # a server misconfiguration, not a bad request
        raise HTTPException(
            status_code=503,
            detail=f"the embedding model gives vectors of {qvec.shape[-1]} dimensions, but "
            f"chunk.embedding is VECTOR({db.VECTOR_DIM}): set WIKILENSE_EMBEDDING_MODEL to the "
            "model the corpus was ingested with",
        )

    conn = _open_connection(settings, target)
    try:
        try:
            # The timed part: the search statement alone, ``repeat`` times on this connection.
            sql_times: list[float] = []
            run_ids: list[list[int]] = []
            hits: list[search.Hit] = []
            for _ in range(repeat or 1):
                start = time.perf_counter()
                try:
                    run_hits = search.search(
                        conn,
                        qvec,
                        k=k,
                        filters=filters,
                        strategy=strategy_name,
                        overfetch=overfetch,
                        ef_search=ef,
                        query_text=text,
                    )
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                sql_times.append((time.perf_counter() - start) * 1000.0)
                if not run_ids:
                    hits = run_hits
                run_ids.append([hit.chunk_id for hit in run_hits])
                if len(run_ids) < (repeat or 1):
                    # One index search per transaction, as for a single search: a REPEATABLE READ
                    # snapshot that spans several searches can leave the index stale
                    # (docs/VECTOR_INDEX.md, MDEV-41321). Outside the timed part.
                    conn.rollback()
            sql_ms = sql_times[0]
            ef_effective = db.get_session_var(conn, EF_SEARCH_VARIABLE) if ef is None else ef
            if not hits and _chunk_count(conn) == 0:
                raise HTTPException(
                    status_code=503, detail="the database is empty: run 'wikilense ingest'"
                )
            sql, _params = search.search_statement(
                qvec,
                k,
                filters,
                strategy_name,
                overfetch,
                query_text=text,
            )
            explain = search.explain_search(
                conn,
                qvec,
                k,
                filters,
                strategy_name,
                overfetch,
                query_text=text,
            )
            hit_dicts = [asdict(hit) for hit in hits]
            if sentences:
                rows = search.hit_sentences(conn, [hit.chunk_id for hit in hits])
                for hit_dict in hit_dicts:
                    hit_dict["sentences"] = [
                        {"element_key": key, "text": sentence_text}
                        for key, sentence_text in rows.get(hit_dict["chunk_id"], [])
                    ]
            # Outside the timed part: the claim's marks and summary need one more lookup
            # (chunk_sentence), which neither the SQL box nor timing_ms.sql includes.
            if offered is not None:
                marks, summary = picker.search_marks(conn, offered, hits, k)
        except pymysql.err.ProgrammingError as exc:
            if exc.args and exc.args[0] == ER_NO_SUCH_TABLE:
                raise HTTPException(
                    status_code=503,
                    detail="the database has no schema or an older one: run 'wikilense "
                    "ingest'",
                ) from exc
            raise
        except pymysql.err.OperationalError as exc:
            # PyMySQL has no mapping for 1191 and raises OperationalError for it (errno >= 1000)
            if exc.args and exc.args[0] == ER_FT_MATCHING_KEY_NOT_FOUND:
                raise HTTPException(
                    status_code=503,
                    detail="the chunk table has no FULLTEXT index (ingested with an older "
                    "schema): re-run 'wikilense ingest' before using strategy rrf",
                ) from exc
            # 1054 has no PyMySQL mapping either: a column of a newer schema is missing
            # (page.last_edited before the page metadata), so the ingest is older than the code
            if exc.args and exc.args[0] == ER_BAD_FIELD_ERROR:
                raise HTTPException(
                    status_code=503,
                    detail="the database was ingested with an older schema: re-run "
                    "'wikilense ingest'",
                ) from exc
            raise
    finally:
        conn.close()
    body: dict[str, Any] = {
        "hits": hit_dicts,
        "sql": sql,
        "explain": explain,
        "ef_search": ef_effective,
        "timing_ms": {
            "model_load": state.model_load_ms,
            "embed": round(embed_ms, 3),
            "sql": round(sql_ms, 3),
        },
    }
    if database is not None:
        body["database"] = target
    if repeat is not None:
        stats = latency_stats(sql_times)
        differing = [n for n, ids in enumerate(run_ids, start=1) if ids != run_ids[0]]
        body["repeat"] = {
            "runs": len(sql_times),
            "sql_ms": [round(ms, 3) for ms in sql_times],
            "median_ms": round(stats.p50_ms, 3),
            "min_ms": round(stats.min_ms, 3),
            "max_ms": round(stats.max_ms, 3),
            "same_hits": not differing,
            "differing_runs": differing,
        }
    if offered is not None:
        body["claim"] = offered.details()
        body["marks"] = marks
        body["summary"] = summary
    return body


def api_databases(request: Request) -> dict[str, Any]:
    """Return the offered databases with their sizes, the note when S1 is missing, and the examples."""
    state = request.app.state
    choices = offered_databases(state)
    s1_missing = state.offer_s1 and SCALED_DATABASE not in {c.name for c in choices}
    return {
        "default": demo_database(state),
        "databases": [asdict(c) for c in choices],
        "s1": SCALED_DATABASE,
        "s1_note": S1_MISSING_NOTE if s1_missing else None,
        "examples_note": EXAMPLES_NOTE,
        "examples": [asdict(e) for e in picker.EXAMPLES],
    }


def api_claims(
    request: Request,
    database: Annotated[
        str | None,
        Query(
            description="the database: the demo database (the default) or "
            f"{SCALED_DATABASE} when it is loaded; GET /api/databases lists the offered ones"
        ),
    ] = None,
) -> dict[str, Any]:
    """Return the claims the picker offers on the database, the claim files that belong to it,
    and the claims left out with the reason."""
    state = request.app.state
    name = choose_database(state, database)
    cat = catalog(state, name)
    return {
        "database": name,
        "own_files": list(cat.own_files),
        "claims": [c.listing() for c in cat.claims.values()],
        "dropped": [{"claim_id": cid, "reason": reason} for cid, reason in cat.dropped],
    }


def api_pages(
    request: Request,
    database: Annotated[
        str | None,
        Query(
            description="the database: the demo database (the default) or "
            f"{SCALED_DATABASE} when it is loaded; GET /api/databases lists the offered ones"
        ),
    ] = None,
) -> dict[str, Any]:
    """Return every page title of the database, sorted."""
    state = request.app.state
    name = choose_database(state, database)
    conn = _open_connection(state.settings, name)
    try:
        try:
            titles = picker.page_titles(conn)
        except pymysql.err.ProgrammingError as exc:
            if _db_error_code(exc) == ER_NO_SUCH_TABLE:
                raise HTTPException(
                    status_code=503,
                    detail=f"the database {name} has no schema: run 'wikilense ingest'",
                ) from exc
            raise
    finally:
        conn.close()
    return {"database": name, "n_pages": len(titles), "titles": titles}


def _parameter_names(endpoint: Callable[..., Any]) -> tuple[str, ...]:
    """Return an endpoint's query parameter names, in signature order (``request`` excluded)."""
    return tuple(name for name in inspect.signature(endpoint).parameters if name != "request")


#: The query parameters ``/api/search`` accepts, in signature order (``request`` excluded).
API_PARAMETER_NAMES: tuple[str, ...] = _parameter_names(api_search)
API_PARAMETERS = frozenset(API_PARAMETER_NAMES)


def rejecting_unknown(names: Sequence[str]) -> Callable[[Request], None]:
    """Return a dependency that raises 400 naming every query parameter not in ``names``.

    FastAPI ignores unknown query parameters by default, so ``?heading_like=...`` would search
    without the filter and the caller would not notice.
    """
    accepted = frozenset(names)

    def check(request: Request) -> None:
        unknown = sorted({name for name in request.query_params if name not in accepted})
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown query parameter(s): {', '.join(unknown)}; accepted: "
                + (", ".join(names) if names else "none"),
            )

    return check


_reject_unknown_query_params = rejecting_unknown(API_PARAMETER_NAMES)


# ---------------------------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    embedder: QueryEmbedder | None = None,
    database: str | None = None,
    warm_up: bool = True,
    offer_s1: bool = True,
    claim_files: Sequence[Path] = picker.CLAIM_FILES,
) -> FastAPI:
    """Return the FastAPI app serving ``GET /`` (the page) and ``GET /api/search`` (JSON).

    ``settings`` defaults to ``load_settings()``. ``embedder`` defaults to
    ``embedding.Embedder(settings.embedding_model)`` and is shared by every request; tests pass
    a fake. With ``warm_up`` (the default) the model is loaded here by :func:`warm_up_embedder` and the
    milliseconds it took are kept as ``app.state.model_load_ms`` (``None`` otherwise); an
    ``OSError`` from a model that cannot be loaded propagates. ``database`` overrides
    ``settings.db_name`` (the test suite passes ``settings.test_db_name``); it is the demo
    database of the selector. With ``offer_s1`` the selector also offers
    :data:`SCALED_DATABASE` when it is loaded; ``claim_files`` are the picker's claim files. The
    settings, the embedder, the database name and the caches of the S1 probe and the claim
    catalogs are kept on ``app.state``. The Swagger page (:data:`DOCS_URL`) and the schema
    (:data:`OPENAPI_URL`) are enabled.
    """
    if settings is None:
        settings = load_settings()
    if embedder is None:
        embedder = Embedder(settings.embedding_model)
    app = FastAPI(
        title="WikiLense",
        version=__version__,
        description=(
            "Semantic search over Wikipedia inside MariaDB: VEC_DISTANCE_COSINE on a VECTOR "
            "INDEX combined with SQL filters in the same statement. GET /api/search takes the "
            "query and the filters; the page at / is a form over it."
        ),
        docs_url=DOCS_URL,
        openapi_url=OPENAPI_URL,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.database = database
    app.state.offer_s1 = offer_s1
    app.state.claim_files = tuple(Path(p) for p in claim_files)
    app.state.lock = threading.RLock()
    app.state.s1_probe = _UNPROBED
    app.state.claims = None
    app.state.catalogs = {}
    app.state.sizes = {}
    app.state.model_load_ms = round(warm_up_embedder(embedder), 3) if warm_up else None
    app.add_api_route("/", index, response_class=HTMLResponse, include_in_schema=False)
    app.add_api_route(
        "/api/search",
        api_search,
        methods=["GET"],
        dependencies=[Depends(_reject_unknown_query_params)],
        summary="Search the chunks",
    )
    for path, endpoint, summary in (
        ("/api/databases", api_databases, "List the offered databases and the examples"),
        ("/api/claims", api_claims, "List the claims the picker offers on a database"),
        ("/api/pages", api_pages, "List a database's page titles"),
    ):
        app.add_api_route(
            path,
            endpoint,
            methods=["GET"],
            dependencies=[Depends(rejecting_unknown(_parameter_names(endpoint)))],
            summary=summary,
        )
    return app
