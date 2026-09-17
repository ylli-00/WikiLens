"""Minimal FastAPI query page: ``GET /`` is one self-contained HTML form, ``GET /api/search`` is JSON.

``create_app(settings=None)`` builds the app. The embedder is created once per app and loads
its model on the first query; a database connection is opened per request and closed in a
``finally`` clause. ``/api/search`` returns ``{"hits": [...], "sql": "...", "explain": [...],
"timing_ms": {"embed": ..., "sql": ...}}`` where ``hits`` are ``search.Hit`` dicts, ``sql`` is
the statement that ran (values are bound as parameters, so it shows ``%s`` placeholders),
``explain`` its ``EXPLAIN`` rows and the two timings the query embedding and the SQL round trip
in milliseconds. The page has no JavaScript framework and no external assets: the HTML, CSS
and the few lines of script that call the API live in :data:`INDEX_HTML`.

Errors: invalid parameters are 422 (FastAPI validation) or 400 (a blank query); a server that
cannot be reached, a database without schema or without chunks, or a model that cannot be
loaded are 503 with a one-sentence ``detail``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import Annotated, Any, Literal, Protocol

import numpy as np
import pymysql
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from wikilense import __version__, db, search
from wikilense.config import Settings, load_settings
from wikilense.embedding import Embedder
from wikilense.search import Filters

DEFAULT_K = 5
MAX_K = 100
"""Largest ``k`` the API accepts (the page shows at most this many hits)."""

CHUNK_COUNT_SQL = "SELECT COUNT(*) FROM chunk"
ER_NO_SUCH_TABLE = 1146  # MariaDB: "Table ... doesn't exist"

#: The accepted values of the ``strategy`` parameter; tests/test_web.py checks that they equal
#: ``search.STRATEGIES``.
Strategy = Literal["inline", "overfetch", "none"]


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
  code, pre { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .9em; }
  form {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: .75rem 1rem;
    padding: 1rem; background: var(--panel); border: 1px solid var(--line); border-radius: .5rem;
  }
  form label { display: flex; flex-direction: column; font-size: .85rem; color: var(--muted); }
  form label.wide { grid-column: 1 / -1; }
  form input, form select {
    margin-top: .25rem; padding: .45rem .55rem; font: inherit; color: var(--fg);
    background: var(--bg); border: 1px solid var(--line); border-radius: .35rem;
  }
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
  details { margin-top: 1.5rem; }
  summary { cursor: pointer; font-weight: 600; }
  pre { white-space: pre-wrap; word-break: break-word; padding: .75rem; background: var(--panel);
        border: 1px solid var(--line); border-radius: .35rem; }
  table { border-collapse: collapse; font-size: .85rem; width: 100%; }
  th, td { text-align: left; padding: .25rem .5rem; border-bottom: 1px solid var(--line);
           vertical-align: top; }
</style>
</head>
<body>
<h1>WikiLense</h1>
<p class="sub">Semantic search over Wikipedia inside MariaDB: <code>VEC_DISTANCE_COSINE</code>
on a <code>VECTOR INDEX</code>, combined with SQL filters in the same statement.</p>

<form id="search-form">
  <label class="wide">Query
    <input name="q" required autofocus placeholder="Aare river in Switzerland"></label>
  <label>k (hits)
    <input name="k" type="number" value="5" min="1" max="100"></label>
  <label>Min words on page
    <input name="min_words" type="number" min="0" placeholder="any"></label>
  <label>Heading LIKE
    <input name="heading" placeholder="%History%"></label>
  <label>Linked from page
    <input name="linked_from" placeholder="page title"></label>
  <label>Strategy
    <select name="strategy">
      <option value="inline">inline (filters in the statement)</option>
      <option value="overfetch">overfetch (filter k x 10 candidates)</option>
      <option value="none">none (ignore filters)</option>
    </select></label>
  <label>mhnsw_ef_search
    <input name="ef_search" type="number" min="1" placeholder="server default"></label>
  <button type="submit">Search</button>
</form>

<p id="status"></p>
<ol id="hits"></ol>
<details id="sql-box" hidden>
  <summary>SQL that ran and its EXPLAIN</summary>
  <pre id="sql"></pre>
  <table id="explain"></table>
</details>

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

  function render(body) {
    list.replaceChildren();
    body.hits.forEach(function (h) {
      var li = el('li');
      var head = el('div', 'head');
      head.appendChild(el('span', 'dist', h.distance.toFixed(4)));
      head.appendChild(el('strong', null, h.title));
      if (h.section_path) head.appendChild(el('span', 'path', '> ' + h.section_path));
      head.appendChild(el('span', 'meta', '(chunk_id ' + h.chunk_id + ', ' + h.n_words + ' words)'));
      li.appendChild(head);
      li.appendChild(el('p', 'text', h.text));
      list.appendChild(li);
    });
    sqlEl.textContent = body.sql;
    renderExplain(body.explain);
    box.hidden = false;
    status.textContent = body.hits.length + ' hits; embedding ' + body.timing_ms.embed.toFixed(1) +
      ' ms, SQL ' + body.timing_ms.sql.toFixed(1) + ' ms';
  }

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    var params = new URLSearchParams();
    new FormData(form).forEach(function (value, name) {
      if (value !== '') params.set(name, value);
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


def create_app(
    settings: Settings | None = None,
    *,
    embedder: QueryEmbedder | None = None,
    database: str | None = None,
) -> FastAPI:
    """Return the FastAPI app serving ``GET /`` (the page) and ``GET /api/search`` (JSON).

    ``settings`` defaults to ``load_settings()``. ``embedder`` defaults to
    ``embedding.Embedder(settings.embedding_model)`` and is shared by every request (the model
    loads on the first query); tests pass a fake. ``database`` overrides ``settings.db_name``
    (the test suite passes ``settings.test_db_name``). The three are kept on ``app.state``.
    """
    if settings is None:
        settings = load_settings()
    if embedder is None:
        embedder = Embedder(settings.embedding_model)
    app = FastAPI(title="WikiLense", version=__version__, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.database = database

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        """Return the query page."""
        return INDEX_HTML

    @app.get("/api/search")
    def api_search(
        q: Annotated[str, Query(min_length=1, description="query text")],
        k: Annotated[int, Query(ge=1, le=MAX_K, description="number of hits")] = DEFAULT_K,
        min_words: Annotated[
            int | None, Query(ge=0, description="only pages with at least N words")
        ] = None,
        heading: Annotated[
            str | None, Query(description="SQL LIKE pattern on section.heading")
        ] = None,
        linked_from: Annotated[
            str | None, Query(description="only pages linked from the page with this title")
        ] = None,
        strategy: Annotated[Strategy, Query(description="inline, overfetch or none")] = "inline",
        ef_search: Annotated[
            int | None, Query(ge=1, description="mhnsw_ef_search for this query")
        ] = None,
    ) -> dict[str, Any]:
        """Return the hits for ``q`` with the SQL that ran, its EXPLAIN rows and the timings."""
        text = q.strip()
        if not text:
            raise HTTPException(status_code=400, detail="q must not be blank")
        filters = Filters(
            min_words=min_words, heading_like=heading or None, linked_from=linked_from or None
        )
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
                hits = search.search(
                    conn, qvec, k=k, filters=filters, strategy=strategy, ef_search=ef_search
                )
                sql_ms = (time.perf_counter() - start) * 1000.0
                if not hits and _chunk_count(conn) == 0:
                    raise HTTPException(
                        status_code=503, detail="the database is empty: run 'wikilense ingest'"
                    )
                sql, _params = search.search_statement(qvec, k, filters, strategy)
                explain = search.explain_search(conn, qvec, k, filters, strategy)
            except pymysql.err.ProgrammingError as exc:
                if exc.args and exc.args[0] == ER_NO_SUCH_TABLE:
                    raise HTTPException(
                        status_code=503,
                        detail="the database has no schema: run 'wikilense init-db' and "
                        "'wikilense ingest'",
                    ) from exc
                raise
        finally:
            conn.close()
        return {
            "hits": [asdict(hit) for hit in hits],
            "sql": sql,
            "explain": explain,
            "timing_ms": {"embed": round(embed_ms, 3), "sql": round(sql_ms, 3)},
        }

    return app
