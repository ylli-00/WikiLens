"""Minimal FastAPI query page: ``GET /`` is one self-contained HTML form, ``GET /api/search`` is JSON.

``create_app(settings=None, embedder=None, database=None, warm_up=True)`` builds the app. The
embedder is created once per app and, with ``warm_up``, loads its model at creation by
embedding one warm-up text, so the first request's ``timing_ms.embed`` is the embedding alone
and not the model load; the milliseconds that warm-up took are kept as
``app.state.model_load_ms`` and returned as ``timing_ms.model_load`` (``null`` without
warm-up: then the first request carries the load in ``embed``). A database connection is
opened per request and closed in a ``finally`` clause.

``/api/search`` takes every field of ``search.Filters`` (``min_words``, ``max_words``,
``heading``, ``path``, ``linked_from``, ``links_to`` and ``titles``, the last one repeated
once per title; :data:`API_FILTER_FIELDS` maps the names), the strategy (the choices are
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

Errors: invalid parameters are 422 (FastAPI validation, an inconsistent filter pair, a filter
given with strategy ``none``, or an argument ``search`` refuses) or 400 (a blank query, an
unknown parameter); a server that cannot be reached, a database without schema, without
chunks or without the FULLTEXT index that ``rrf`` needs, or a model that cannot be loaded are
503 with a one-sentence ``detail``.
"""

from __future__ import annotations

import html
import inspect
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import Annotated, Any, Literal, Protocol

import numpy as np
import pymysql
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from wikilense import __version__, db, search
from wikilense.config import Settings, load_settings
from wikilense.embedding import Embedder
from wikilense.search import EF_SEARCH_VARIABLE, STRATEGIES, Filters

DEFAULT_K = 5
MAX_K = 100
"""Largest ``k`` the API accepts (the page shows at most this many hits)."""
DEFAULT_OVERFETCH = 10
DEFAULT_STRATEGY = "inline"

WARM_UP_TEXT = "warm-up"

DOCS_URL = "/docs"
OPENAPI_URL = "/openapi.json"

CHUNK_COUNT_SQL = "SELECT COUNT(*) FROM chunk"
ER_NO_SUCH_TABLE = 1146  # MariaDB: "Table ... doesn't exist"
ER_FT_MATCHING_KEY_NOT_FOUND = 1191  # MariaDB: "Can't find FULLTEXT index matching the column list"

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
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fg: #e8e8e8; --bg: #141414; --muted: #a0a0a0; --line: #333333;
      --accent: #8ab8f0; --panel: #1e1e1e;
    }
  }
  * { box-sizing: border-box; }
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
  form label.wide { grid-column: 1 / -1; }
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
  #status { min-height: 1.5rem; color: var(--muted); }
  #hits { padding-left: 1.5rem; }
  #hits li { margin: 0 0 1.25rem; }
  .head { display: flex; flex-wrap: wrap; gap: .5rem; align-items: baseline; }
  .dist { font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--accent); }
  .path, .meta { color: var(--muted); font-size: .9rem; }
  .text { margin: .25rem 0 0; }
  ul.sentences { margin: .35rem 0 0; padding-left: 1rem; font-size: .9rem; }
  ul.sentences li { margin: 0 0 .15rem; }
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

<form id="search-form">
  <label class="wide">Query
    <input name="q" required autofocus placeholder="Aare river in Switzerland"></label>
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
  <button type="submit">Search</button>
</form>

<p id="status"></p>
<ol id="hits"></ol>
<details id="sql-box" hidden>
  <summary>SQL that ran and its EXPLAIN</summary>
  <pre id="sql"></pre>
  <table id="explain"></table>
</details>

<footer>
<p>The form calls <code>GET /api/search</code> with the same parameter names (<code>sentences=1</code>
for the sentence list, <code>titles</code> once per title); an unknown parameter is rejected
with 400. The interactive API
documentation is left on so the API can be tried from the browser:
<a href="__DOCS_URL__">__DOCS_URL__</a> (OpenAPI schema at <a href="__OPENAPI_URL__">__OPENAPI_URL__</a>).
Model load, query embedding and SQL time are reported separately in the status line.</p>
</footer>

<script>
(function () {
  var form = document.getElementById('search-form');
  var status = document.getElementById('status');
  var list = document.getElementById('hits');
  var box = document.getElementById('sql-box');
  var sqlEl = document.getElementById('sql');
  var explainEl = document.getElementById('explain');

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function describe(detail) {
    if (Array.isArray(detail)) {
      return detail.map(function (d) {
        return (d.loc || []).slice(1).join('.') + ': ' + d.msg;
      }).join('; ');
    }
    return String(detail);
  }

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

  function renderSentences(rows) {
    var ul = el('ul', 'sentences');
    if (!rows.length) {
      ul.appendChild(el('li', null, 'no chunk_sentence rows'));
      return ul;
    }
    rows.forEach(function (s) {
      var li = el('li');
      li.appendChild(el('code', null, s.element_key));
      li.appendChild(document.createTextNode(' ' + s.text));
      ul.appendChild(li);
    });
    return ul;
  }

  function render(body) {
    list.replaceChildren();
    body.hits.forEach(function (h) {
      var li = el('li');
      var head = el('div', 'head');
      head.appendChild(el('span', 'dist', h.distance.toFixed(4)));
      head.appendChild(el('strong', null, h.title));
      if (h.section_path) head.appendChild(el('span', 'path', '> ' + h.section_path));
      var meta = '(chunk_id ' + h.chunk_id + ', ' + h.n_words + ' words';
      if (h.score !== undefined && h.score !== null) meta += ', rrf score ' + h.score.toFixed(4);
      head.appendChild(el('span', 'meta', meta + ')'));
      li.appendChild(head);
      li.appendChild(el('p', 'text', h.text));
      if (h.sentences) li.appendChild(renderSentences(h.sentences));
      list.appendChild(li);
    });
    sqlEl.textContent = body.sql;
    renderExplain(body.explain);
    box.hidden = false;
    var t = body.timing_ms;
    var loaded = t.model_load === null ? 'model loaded lazily' :
      'model loaded at start-up in ' + t.model_load.toFixed(0) + ' ms';
    status.textContent = body.hits.length + ' hits; embedding ' + t.embed.toFixed(1) +
      ' ms, SQL ' + t.sql.toFixed(1) + ' ms; mhnsw_ef_search ' + body.ef_search + ' (' + loaded + ')';
  }

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
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
    status.textContent = 'Searching...';
    list.replaceChildren();
    box.hidden = true;
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
) -> dict[str, Any]:
    """Return the hits for ``q`` with the SQL that ran, its EXPLAIN rows, ``ef_search`` and timings."""
    settings: Settings = request.app.state.settings
    embedder: QueryEmbedder = request.app.state.embedder
    database: str | None = request.app.state.database
    text = q.strip()
    if not text:
        raise HTTPException(status_code=400, detail="q must not be blank")
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

    conn = _open_connection(settings, database)
    try:
        try:
            start = time.perf_counter()
            try:
                hits = search.search(
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
            sql_ms = (time.perf_counter() - start) * 1000.0
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
        except pymysql.err.ProgrammingError as exc:
            if exc.args and exc.args[0] == ER_NO_SUCH_TABLE:
                raise HTTPException(
                    status_code=503,
                    detail="the database has no schema: run 'wikilense init-db' and "
                    "'wikilense ingest'",
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
            raise
    finally:
        conn.close()
    return {
        "hits": hit_dicts,
        "sql": sql,
        "explain": explain,
        "ef_search": ef_effective,
        "timing_ms": {
            "model_load": request.app.state.model_load_ms,
            "embed": round(embed_ms, 3),
            "sql": round(sql_ms, 3),
        },
    }


#: The query parameters ``/api/search`` accepts, in signature order (``request`` excluded).
API_PARAMETER_NAMES: tuple[str, ...] = tuple(
    name for name in inspect.signature(api_search).parameters if name != "request"
)
API_PARAMETERS = frozenset(API_PARAMETER_NAMES)


def _reject_unknown_query_params(request: Request) -> None:
    """Raise 400 naming every query parameter that is not in :data:`API_PARAMETERS`.

    FastAPI ignores unknown query parameters by default, so ``?heading_like=...`` would search
    without the filter and the caller would not notice. Returns None when all are known.
    """
    unknown = sorted({name for name in request.query_params if name not in API_PARAMETERS})
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown query parameter(s): {', '.join(unknown)}; accepted: "
            + ", ".join(API_PARAMETER_NAMES),
        )


# ---------------------------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    embedder: QueryEmbedder | None = None,
    database: str | None = None,
    warm_up: bool = True,
) -> FastAPI:
    """Return the FastAPI app serving ``GET /`` (the page) and ``GET /api/search`` (JSON).

    ``settings`` defaults to ``load_settings()``. ``embedder`` defaults to
    ``embedding.Embedder(settings.embedding_model)`` and is shared by every request; tests pass
    a fake. With ``warm_up`` (the default) the model is loaded here by :func:`warm_up_embedder` and the
    milliseconds it took are kept as ``app.state.model_load_ms`` (``None`` otherwise); an
    ``OSError`` from a model that cannot be loaded propagates. ``database`` overrides
    ``settings.db_name`` (the test suite passes ``settings.test_db_name``). The settings, the
    embedder and the database name are kept on ``app.state``. The Swagger page
    (:data:`DOCS_URL`) and the schema (:data:`OPENAPI_URL`) are enabled.
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
    app.state.model_load_ms = round(warm_up_embedder(embedder), 3) if warm_up else None
    app.add_api_route("/", index, response_class=HTMLResponse, include_in_schema=False)
    app.add_api_route(
        "/api/search",
        api_search,
        methods=["GET"],
        dependencies=[Depends(_reject_unknown_query_params)],
        summary="Search the chunks",
    )
    return app
