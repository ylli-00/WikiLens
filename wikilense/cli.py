"""The ``wikilense`` command: ``init-db``, ``ingest``, ``query``, ``eval`` and ``serve``.

``main(argv=None) -> int`` is the console-script entry point (pyproject.toml). Every subcommand
reads its settings with ``config.load_settings()`` (environment and ``.env``), connects with
``db.connect`` and closes the connection in a ``finally`` clause. The embedding model is loaded
only by the commands that need it (``query``, ``eval``, ``serve``) and only after the database
has been checked, so an unreachable or empty database fails before the model is touched.

Exit codes: 0 success; 1 an expected failure (settings missing, server unreachable, database
empty or without schema, ingest refused, model not loadable), reported as one line on stderr
without a traceback; 2 bad arguments; 130 interrupted. ``query`` writes its result to stdout
(text, or JSON with ``--json``) and one timing line to stderr, so ``--json`` output can be piped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pymysql

from wikilense import __version__, db, evaluate, ingest, search
from wikilense.config import Settings, SettingsError, load_settings
from wikilense.corpus import DEFAULT_CORPUS_DIR
from wikilense.embedding import Embedder
from wikilense.search import STRATEGIES, Filters, Hit

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

DEFAULT_K = 5
DEFAULT_OVERFETCH = 10
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

TEXT_WIDTH = 100
"""Column at which ``query`` wraps chunk texts and the SQL."""
INDENT = "    "

CHUNK_COUNT_SQL = "SELECT COUNT(*) FROM chunk"
ER_NO_SUCH_TABLE = 1146  # MariaDB: "Table ... doesn't exist"


class CliError(Exception):
    """An expected failure: ``main`` prints the message as one stderr line and returns ``exit_code``."""

    def __init__(self, message: str, exit_code: int = EXIT_FAILURE) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors become a ``CliError`` (exit 2) instead of ``SystemExit``."""

    def error(self, message: str) -> NoReturn:
        raise CliError(f"{message} (try '{self.prog} --help')", exit_code=EXIT_USAGE)


# ---------------------------------------------------------------------------------------------
# argument types
# ---------------------------------------------------------------------------------------------


def positive_int(text: str) -> int:
    """Return ``text`` as an int >= 1; raises ``argparse.ArgumentTypeError`` otherwise."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value}")
    return value


def non_negative_int(text: str) -> int:
    """Return ``text`` as an int >= 0; raises ``argparse.ArgumentTypeError`` otherwise."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {value}")
    return value


def port_number(text: str) -> int:
    """Return ``text`` as a TCP port (1..65535); raises ``argparse.ArgumentTypeError`` otherwise."""
    value = positive_int(text)
    if value > 65535:
        raise argparse.ArgumentTypeError(f"expected a port between 1 and 65535, got {value}")
    return value


def file_stem(text: str) -> str:
    """Return ``text`` when it is a plain file name without directories; ArgumentTypeError otherwise."""
    if not text or Path(text).name != text or text in (".", ".."):
        raise argparse.ArgumentTypeError(f"expected a plain file name without directories, got {text!r}")
    return text


def k_list(text: str) -> tuple[int, ...]:
    """Return ``"1,3,5"`` as ``(1, 3, 5)``: distinct integers >= 1, in the given order.

    Raises ``argparse.ArgumentTypeError`` for anything else.
    """
    try:
        values = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers such as 1,3,5, got {text!r}"
        ) from None
    if any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(f"every k must be >= 1, got {text!r}")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"k values must be distinct, got {text!r}")
    return values


# ---------------------------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the ``wikilense`` argument parser with its five subcommands.

    Each subparser sets ``func`` to the command function, which takes the parsed namespace and
    returns the exit code.
    """
    parser = _Parser(
        prog="wikilense",
        description=(
            "Semantic search over Wikipedia inside MariaDB: VECTOR column, VECTOR INDEX and "
            "VEC_DISTANCE_COSINE combined with SQL filters."
        ),
    )
    parser.add_argument("--version", action="version", version=f"wikilense {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p_init = commands.add_parser("init-db", help="create the tables from sql/schema.sql")
    p_init.add_argument("--reset", action="store_true", help="drop the tables first")
    p_init.set_defaults(func=cmd_init_db)

    p_ingest = commands.add_parser(
        "ingest", help="load data/corpus into the database and embed the chunks"
    )
    p_ingest.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="directory with pages.jsonl and claims.jsonl (default: data/corpus)",
    )
    p_ingest.add_argument(
        "--no-reset",
        action="store_true",
        help="keep existing tables (default: drop and recreate them first)",
    )
    p_ingest.add_argument(
        "--batch-size", type=positive_int, default=64, help="embedding batch size (default 64)"
    )
    p_ingest.add_argument(
        "--no-prefix",
        action="store_true",
        help='embed the bare chunk text instead of "title > section: text"',
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = commands.add_parser("query", help="search the chunks for a text")
    p_query.add_argument("text", help="the query text")
    p_query.add_argument(
        "--k", type=positive_int, default=DEFAULT_K, help=f"number of hits (default {DEFAULT_K})"
    )
    p_query.add_argument(
        "--min-words", type=non_negative_int, help="only pages with at least N words"
    )
    p_query.add_argument(
        "--max-words", type=non_negative_int, help="only pages with at most N words"
    )
    p_query.add_argument(
        "--heading", metavar="PATTERN", help="only sections whose heading matches a SQL LIKE pattern"
    )
    p_query.add_argument(
        "--path", metavar="PATTERN", help="only sections whose path matches a SQL LIKE pattern"
    )
    p_query.add_argument(
        "--linked-from", metavar="TITLE", help="only pages linked from the page with this title"
    )
    p_query.add_argument(
        "--links-to", metavar="TITLE", help="only pages that link to the page with this title"
    )
    p_query.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="inline",
        help="how filters are applied: inline (one statement), overfetch, none (default inline)",
    )
    p_query.add_argument(
        "--overfetch",
        type=positive_int,
        default=DEFAULT_OVERFETCH,
        help=f"inner limit factor of the overfetch strategy (default {DEFAULT_OVERFETCH})",
    )
    p_query.add_argument(
        "--ef-search", type=positive_int, metavar="N", help="mhnsw_ef_search for this query"
    )
    p_query.add_argument(
        "--explain", action="store_true", help="also print the SQL that ran and its EXPLAIN rows"
    )
    p_query.add_argument(
        "--json",
        action="store_true",
        help="print a JSON array of hits (with --explain: an object with hits, sql, explain)",
    )
    p_query.set_defaults(func=cmd_query)

    p_eval = commands.add_parser("eval", help="measure recall and latency over the corpus claims")
    p_eval.add_argument(
        "--k",
        type=k_list,
        default=evaluate.DEFAULT_KS,
        metavar="LIST",
        help=f"comma-separated cut-offs (default {','.join(map(str, evaluate.DEFAULT_KS))})",
    )
    p_eval.add_argument(
        "--repeats",
        type=positive_int,
        default=evaluate.DEFAULT_REPEATS,
        help=f"timed passes per claim after one warm-up (default {evaluate.DEFAULT_REPEATS})",
    )
    p_eval.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=evaluate.DEFAULT_STRATEGY,
        help=f"search strategy (default {evaluate.DEFAULT_STRATEGY}: plain k-nearest neighbours)",
    )
    p_eval.add_argument(
        "--overfetch",
        type=positive_int,
        default=evaluate.DEFAULT_OVERFETCH,
        help=f"inner limit factor of the overfetch strategy (default {evaluate.DEFAULT_OVERFETCH})",
    )
    p_eval.add_argument(
        "--ef-search", type=positive_int, metavar="N", help="mhnsw_ef_search for the whole run"
    )
    p_eval.add_argument(
        "--out",
        type=Path,
        default=evaluate.DEFAULT_RESULTS_DIR,
        help="output directory (default: results/ in the repository)",
    )
    p_eval.add_argument(
        "--name",
        type=file_stem,
        default=evaluate.DEFAULT_NAME,
        help=f"stem of the .json and .md files (default {evaluate.DEFAULT_NAME}); "
        "give every sweep run its own name",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_serve = commands.add_parser("serve", help="start the web query page")
    p_serve.add_argument("--host", default=DEFAULT_HOST, help=f"bind address (default {DEFAULT_HOST})")
    p_serve.add_argument(
        "--port", type=port_number, default=DEFAULT_PORT, help=f"TCP port (default {DEFAULT_PORT})"
    )
    p_serve.set_defaults(func=cmd_serve)
    return parser


# ---------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------


def _mysql_message(exc: pymysql.err.MySQLError) -> str:
    """Return the server's message of a PyMySQL error, with its code, without the tuple repr."""
    if len(exc.args) >= 2:
        return f"{exc.args[1]} (error {exc.args[0]})"
    return str(exc)


def _connect(settings: Settings) -> pymysql.Connection:
    """Return a connection to ``settings.db_name``; a connection failure becomes a ``CliError``."""
    try:
        return db.connect(settings)
    except pymysql.err.OperationalError as exc:
        raise CliError(
            f"cannot connect to MariaDB at {settings.db_host}:{settings.db_port} as "
            f"{settings.db_user!r}, database {settings.db_name!r}: {_mysql_message(exc)}"
        ) from exc


def _require_chunks(conn: pymysql.Connection, settings: Settings) -> int:
    """Return the number of chunk rows; ``CliError`` when there are none or the table is missing."""
    try:
        with conn.cursor() as cur:
            cur.execute(CHUNK_COUNT_SQL)
            count = int(cur.fetchone()[0])
    except pymysql.err.ProgrammingError as exc:
        if exc.args and exc.args[0] == ER_NO_SUCH_TABLE:
            raise CliError(
                f"database {settings.db_name!r} has no schema: run 'wikilense init-db' and "
                "'wikilense ingest' first"
            ) from exc
        raise
    if count == 0:
        raise CliError(f"database {settings.db_name!r} is empty: run 'wikilense ingest' first")
    return count


class _DropHubTokenAdvice(logging.Filter):
    """Drops the Hub's "set a HF_TOKEN" advisory, logged on every revision check of a cached model."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests" not in record.getMessage()


_HUB_TOKEN_FILTER = _DropHubTokenAdvice()
HUB_HTTP_LOGGER = "huggingface_hub.utils._http"


def make_embedder(settings: Settings) -> Embedder:
    """Return the ``Embedder`` for ``settings.embedding_model`` with a quiet terminal.

    The transformers progress bars are disabled and the Hub's token advisory (a WARNING log
    record from :data:`HUB_HTTP_LOGGER`) is filtered out; other Hub warnings still show. The
    model itself is loaded on the embedder's first use.
    """
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    logging.getLogger(HUB_HTTP_LOGGER).addFilter(_HUB_TOKEN_FILTER)
    return Embedder(settings.embedding_model)


def _embed_query(embedder: Embedder, text: str) -> tuple[np.ndarray, float]:
    """Return ``(query vector, milliseconds)``; a model that cannot load becomes a ``CliError``."""
    start = time.perf_counter()
    try:
        vector = embedder.embed_queries([text])[0]
    except OSError as exc:
        raise CliError(f"cannot load embedding model {embedder.model_name!r}: {exc}") from exc
    return vector, (time.perf_counter() - start) * 1000.0


def _filters_from_args(args: argparse.Namespace) -> Filters:
    """Return the ``search.Filters`` for the ``query`` options; an invalid combination is exit 2."""
    try:
        return Filters(
            min_words=args.min_words,
            max_words=args.max_words,
            heading_like=args.heading,
            path_like=args.path,
            linked_from=args.linked_from,
            links_to=args.links_to,
        )
    except (TypeError, ValueError) as exc:
        raise CliError(str(exc), exit_code=EXIT_USAGE) from exc


@contextmanager
def _stage_logging() -> Iterator[None]:
    """Send the ``wikilense`` loggers' INFO lines (the ingest stage summaries) to stderr.

    The handler and level are added for the ``with`` block only and removed afterwards, so a
    caller of ``main()`` (the tests) does not keep a handler bound to a stale stream.
    """
    logger = logging.getLogger("wikilense")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


# ---------------------------------------------------------------------------------------------
# output formatting
# ---------------------------------------------------------------------------------------------


def format_hit(rank: int, hit: Hit, width: int = TEXT_WIDTH) -> str:
    """Return one hit as text: a header line and the chunk text wrapped at ``width`` columns.

    The header is ``rank. distance  title > section path  [chunk_id N, M words]``; the lead
    section (empty path) shows the title alone.
    """
    where = f"{hit.title} > {hit.section_path}" if hit.section_path else hit.title
    head = f"{rank:>2}. {hit.distance:.4f}  {where}  [chunk_id {hit.chunk_id}, {hit.n_words} words]"
    body = textwrap.fill(hit.text, width=width, initial_indent=INDENT, subsequent_indent=INDENT)
    return f"{head}\n{body}"


def format_hits(hits: Sequence[Hit], width: int = TEXT_WIDTH) -> str:
    """Return every hit from :func:`format_hit`, separated by blank lines, or ``"no hits"``."""
    if not hits:
        return "no hits"
    return "\n\n".join(format_hit(rank, hit, width) for rank, hit in enumerate(hits, start=1))


def _describe_param(value: Any) -> str:
    """Return a short description of a bound parameter (a vector shows its dimension only)."""
    if isinstance(value, (bytes, bytearray)):
        return f"<vector[{len(value) // 4}]>"
    return repr(value)


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "") -> str:
    """Return ``headers`` and ``rows`` as left-aligned columns separated by two spaces."""
    widths = [max([len(h)] + [len(row[i]) for row in rows]) for i, h in enumerate(headers)]
    lines = [indent + "  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    for row in rows:
        lines.append(indent + "  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def format_explain(
    sql: str, params: Sequence[Any], rows: Sequence[Mapping[str, Any]], width: int = TEXT_WIDTH
) -> str:
    """Return the SQL (wrapped), a summary of its parameters and the EXPLAIN rows as a table."""
    lines = [
        "SQL:",
        textwrap.fill(
            sql,
            width=width,
            initial_indent=INDENT,
            subsequent_indent=INDENT,
            break_long_words=False,
            break_on_hyphens=False,
        ),
        "parameters: " + ", ".join(_describe_param(p) for p in params),
        "EXPLAIN:",
    ]
    if not rows:
        lines.append(INDENT + "(no rows)")
        return "\n".join(lines)
    columns = list(rows[0])
    cells = [["NULL" if row.get(c) is None else str(row.get(c)) for c in columns] for row in rows]
    lines.append(format_table(columns, cells, indent=INDENT))
    return "\n".join(lines)


EVAL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "k", "article_recall", "evidence_recall", "unit_coverage", "sql_p50_ms", "sql_p95_ms"
)  # fmt: skip


def format_eval_summary(result: evaluate.EvalResult) -> str:
    """Return the per-k metrics of an ``EvalResult`` as a table plus one latency line."""
    rows = [
        (
            str(m.k),
            f"{m.article_recall:.3f}",
            f"{m.evidence_recall:.3f}",
            f"{m.unit_coverage:.3f}",
            f"{m.sql_latency.p50_ms:.2f}",
            f"{m.sql_latency.p95_ms:.2f}",
        )
        for m in result.per_k
    ]
    emb = result.embedding_latency
    return (
        format_table(EVAL_SUMMARY_COLUMNS, rows)
        + f"\nclaims {result.n_claims} (evidence-eligible {result.n_evidence_claims}); "
        f"query embedding p50 {emb.p50_ms:.2f} ms, p95 {emb.p95_ms:.2f} ms over {emb.n} samples"
    )


def hits_to_json(hits: Sequence[Hit]) -> list[dict[str, Any]]:
    """Return the hits as JSON-ready dicts with the ``Hit`` fields as keys."""
    return [asdict(hit) for hit in hits]


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------


def cmd_init_db(args: argparse.Namespace) -> int:
    """Apply sql/schema.sql to the database (``--reset`` drops the tables first); returns 0."""
    settings = load_settings()
    conn = _connect(settings)
    try:
        n_statements = db.apply_schema(conn, reset=args.reset)
    finally:
        conn.close()
    action = "reset and created" if args.reset else "created (existing tables kept)"
    print(f"schema {action}: {n_statements} statements applied to database {settings.db_name!r}")
    return EXIT_OK


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run ``ingest.run_ingest`` with the given options and print the report; returns 0."""
    settings = load_settings()
    with _stage_logging():
        report = ingest.run_ingest(
            settings,
            corpus_dir=args.corpus_dir,
            reset=not args.no_reset,
            batch_size=args.batch_size,
            use_prefix=not args.no_prefix,
            progress=sys.stderr.isatty(),
        )
    seconds = report.seconds
    stages = ", ".join(f"{name} {seconds.get(name, 0.0):.1f}" for name in ingest.STAGES[:-1])
    print(f"ingest finished in {seconds.get('total', 0.0):.1f} s ({stages})")
    print(
        f"  pages {report.n_pages}, sections {report.n_sections}, sentences {report.n_sentences} "
        f"(empty {report.n_units_empty}), chunks {report.n_chunks}"
    )
    print(
        f"  links {report.n_links} (resolved {report.n_links_resolved}, "
        f"skipped {report.n_links_skipped})"
    )
    print(
        f"  claims {report.n_claims}, evidence ids {report.n_evidence} "
        f"(page resolved {report.n_evidence_page_resolved}, "
        f"sentence resolved {report.n_evidence_sentence_resolved})"
    )
    return EXIT_OK


def cmd_query(args: argparse.Namespace) -> int:
    """Embed the text, run ``search.search`` and print the hits (text or JSON); returns 0.

    The database is checked (reachable, schema present, chunks present) before the model is
    loaded. Timing goes to stderr so that stdout stays parseable with ``--json``.
    """
    settings = load_settings()
    filters = _filters_from_args(args)
    conn = _connect(settings)
    sql = ""
    params: tuple[Any, ...] = ()
    explain_rows: list[dict[str, Any]] = []
    try:
        _require_chunks(conn, settings)
        embedder = make_embedder(settings)
        qvec, embed_ms = _embed_query(embedder, args.text)
        try:
            start = time.perf_counter()
            hits = search.search(
                conn,
                qvec,
                k=args.k,
                filters=filters,
                strategy=args.strategy,
                overfetch=args.overfetch,
                ef_search=args.ef_search,
            )
            sql_ms = (time.perf_counter() - start) * 1000.0
            if args.explain:
                sql, params = search.search_statement(
                    qvec, args.k, filters, args.strategy, args.overfetch
                )
                explain_rows = search.explain_search(
                    conn, qvec, args.k, filters, args.strategy, args.overfetch
                )
        except (TypeError, ValueError) as exc:
            raise CliError(str(exc), exit_code=EXIT_USAGE) from exc
    finally:
        conn.close()
    if args.json:
        payload: Any = hits_to_json(hits)
        if args.explain:
            payload = {"hits": payload, "sql": sql, "explain": explain_rows}
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_hits(hits))
        if args.explain:
            print()
            print(format_explain(sql, params, explain_rows))
    sys.stdout.flush()  # keep the order stable when stdout and stderr share a pipe
    print(
        f"{len(hits)} hits; embedding {embed_ms:.1f} ms, SQL {sql_ms:.1f} ms "
        f"({settings.embedding_model} on {embedder.device}, strategy {args.strategy})",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    """Run ``evaluate.evaluate`` over the corpus claims, write and summarise the results; returns 0.

    The database is checked before the model is loaded; ``evaluate``'s own argument errors (for
    example a database without claims) are reported as one line.
    """
    settings = load_settings()
    conn = _connect(settings)
    try:
        _require_chunks(conn, settings)
        embedder = make_embedder(settings)
        try:
            result = evaluate.evaluate(
                conn,
                embedder,
                ks=args.k,
                repeats=args.repeats,
                strategy=args.strategy,
                filters=None,
                ef_search=args.ef_search,
                overfetch=args.overfetch,
            )
        except ValueError as exc:
            raise CliError(str(exc)) from exc
    finally:
        conn.close()
    json_path, md_path = evaluate.write_results(result, out_dir=args.out, name=args.name)
    print(format_eval_summary(result))
    print(f"results written: {json_path}, {md_path}")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    """Check the database, load the model, then serve the web page with uvicorn; returns 0."""
    import uvicorn

    from wikilense import web

    settings = load_settings()
    conn = _connect(settings)
    try:
        n_chunks = _require_chunks(conn, settings)
    finally:
        conn.close()
    embedder = make_embedder(settings)
    print(f"loading embedding model {settings.embedding_model} ...", file=sys.stderr)
    _embed_query(embedder, "warm-up")
    app = web.create_app(settings, embedder=embedder)
    print(
        f"serving database {settings.db_name!r} ({n_chunks} chunks) on "
        f"http://{args.host}:{args.port}/",
        file=sys.stderr,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------


def _fail(message: str) -> None:
    """Print one error line to stderr."""
    print(f"wikilense: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line with ``argv`` (default ``sys.argv[1:]``) and return the exit code.

    Expected failures are printed as a single stderr line (see the module docstring for the
    codes); anything else propagates with its traceback.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.func(args))
    except CliError as exc:
        _fail(str(exc))
        return exc.exit_code
    except SettingsError as exc:
        _fail(str(exc))
        return EXIT_FAILURE
    except (ingest.IngestError, FileNotFoundError) as exc:
        _fail(str(exc))
        return EXIT_FAILURE
    except pymysql.err.MySQLError as exc:
        _fail(f"database error: {_mysql_message(exc)}")
        return EXIT_FAILURE
    except KeyboardInterrupt:
        _fail("interrupted")
        return EXIT_INTERRUPTED
    except SystemExit as exc:  # --help and --version exit through argparse
        code = exc.code
        return code if isinstance(code, int) else (EXIT_OK if code is None else EXIT_FAILURE)


if __name__ == "__main__":  # pragma: no cover - the console script calls main()
    sys.exit(main())
