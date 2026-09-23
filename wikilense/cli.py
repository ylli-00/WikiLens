"""The ``wikilense`` command: ``init-db``, ``ingest``, ``query``, ``eval`` and ``serve``.

``main(argv=None) -> int`` is the console-script entry point (pyproject.toml). Every subcommand
reads its settings with ``config.load_settings()`` (environment and ``.env``), connects with
``db.connect`` and closes the connection in a ``finally`` clause. The embedding model is loaded
only by the commands that need it (``ingest``, ``query``, ``eval``, ``serve``) and, for the
query commands, only after the database has been checked, so an unreachable or empty database
fails before the model is touched. A model that cannot be loaded (a bad name, or offline
without a cached copy; sentence-transformers raises ``OSError``) is reported as one line by
every command that needs it.

Exit codes: 0 success; 1 an expected failure (settings missing, server unreachable, database
empty or without schema, ingest refused, model not loadable or of another dimension than
``chunk.embedding``, results not writable), reported as one line on stderr without a traceback;
2 bad arguments (including a filter given with ``--strategy none``, which would silently drop
it, and an ``eval --out`` that cannot be a results directory); 130 interrupted. ``query`` writes its result to stdout (text, or JSON
with ``--json``) and one timing line to stderr, so ``--json`` output can be piped. The timing
line separates the model load (one warm-up call that loads the model, seconds from a cold
process) from the query embedding and the SQL round trip, names the model that ran (the
embedder built from the command's settings, ``settings.embedding_model``) and its device, and
the effective ``mhnsw_ef_search``: ``settings.ef_search`` (``WIKILENSE_EF_SEARCH``) unless
``--ef-search`` is given; ``--ef-search 0`` or ``--ef-search server`` leaves the server's
session value alone. Every ``search.Filters`` field has an option (``--title`` repeats for
several titles).

The strategy choices are read from ``search.STRATEGIES`` when the parser is built, so a
strategy added to the search module appears here without a change; ``search()`` always gets
``query_text=<the query>`` (the ``rrf`` strategy's full-text half; the others ignore it).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from wikilense.search import Filters, Hit

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

EXIT_CODES_HELP = (
    "exit codes: 0 success; 1 an expected failure (settings missing, server unreachable, "
    "database empty or without schema, ingest refused, embedding model not loadable or of the "
    "wrong dimension, results not writable), reported as one line on stderr; 2 bad arguments; "
    "130 interrupted."
)

DEFAULT_K = 5
DEFAULT_OVERFETCH = 10
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

EF_SEARCH_SERVER_WORD = "server"
"""``--ef-search server``: the same as ``--ef-search 0`` (``search.EF_SEARCH_SERVER``)."""

WARM_UP_TEXT = "warm-up"
"""Text embedded once before a query is timed, so the model load is measured on its own."""

#: The strategy that runs without filters; a filter given with it is a usage error.
STRATEGY_NO_FILTERS = "none"

#: One line per known strategy for ``--help``; a strategy missing here is listed by name only.
STRATEGY_HELP: dict[str, str] = {
    "inline": "filters and joins inside the index-driven statement, always k rows",
    "overfetch": "the index returns k x OVERFETCH candidates and the filters run on those, "
    "may return fewer than k",
    "none": "plain nearest neighbours, no filter allowed",
}

#: ``query`` filter options as (namespace attribute, command-line flag), in help order.
FILTER_OPTIONS: tuple[tuple[str, str], ...] = (
    ("min_words", "--min-words"),
    ("max_words", "--max-words"),
    ("heading", "--heading"),
    ("path", "--path"),
    ("linked_from", "--linked-from"),
    ("links_to", "--links-to"),
    ("titles", "--title"),
)

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


def ef_search_arg(text: str) -> int:
    """Return ``--ef-search``: an int from 1 to ``search.MAX_EF_SEARCH``, or
    ``search.EF_SEARCH_SERVER`` (0) for ``0``/``server``.

    Raises ``argparse.ArgumentTypeError`` for anything else (the server would silently clamp a
    value above its maximum).
    """
    if text.strip().lower() == EF_SEARCH_SERVER_WORD:
        return search.EF_SEARCH_SERVER
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer >= 0 or '{EF_SEARCH_SERVER_WORD}', got {text!r}"
        ) from None
    if not search.EF_SEARCH_SERVER <= value <= search.MAX_EF_SEARCH:
        raise argparse.ArgumentTypeError(
            f"expected an integer from 0 (the server's session value) to "
            f"{search.MAX_EF_SEARCH}, got {value}"
        )
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


def strategy_help(default: str) -> str:
    """Return the ``--strategy`` help text listing every ``search.STRATEGIES`` entry."""
    parts = [f"{name}: {STRATEGY_HELP.get(name, 'see wikilense.search')}" for name in search.STRATEGIES]
    return "how the filters are applied. " + "; ".join(parts) + f" (default {default})"


EF_SEARCH_HELP = (
    "mhnsw_ef_search for this session: the number of candidates the HNSW index keeps while "
    f"searching (more = closer to the exact ranking, slower; at most {search.MAX_EF_SEARCH}). "
    f"Default WIKILENSE_EF_SEARCH from .env; 0 or '{EF_SEARCH_SERVER_WORD}' leaves the server's "
    "session value alone"
)


def build_parser() -> argparse.ArgumentParser:
    """Return the ``wikilense`` argument parser with its five subcommands.

    Each subparser sets ``func`` to the command function, which takes the parsed namespace and
    returns the exit code. The ``--strategy`` choices are ``search.STRATEGIES`` at call time.
    """
    parser = _Parser(
        prog="wikilense",
        description=(
            "Semantic search over Wikipedia inside MariaDB: VECTOR column, VECTOR INDEX and "
            "VEC_DISTANCE_COSINE combined with SQL filters."
        ),
        epilog=EXIT_CODES_HELP,
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
        help="keep the tables made by 'init-db' instead of dropping and recreating them; a "
        "database that already holds pages is refused",
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

    p_query = commands.add_parser(
        "query",
        help="search the chunks for a text",
        description=(
            "Embed the text, run the nearest-neighbour statement with the given SQL filters "
            "and print the hits (stdout) and one timing line (stderr). Every filter keeps only "
            "chunks whose page or section passes; filters combine with AND."
        ),
        epilog=EXIT_CODES_HELP,
    )
    p_query.add_argument("text", help="the query text")
    p_query.add_argument(
        "--k", type=positive_int, default=DEFAULT_K, help=f"number of hits (default {DEFAULT_K})"
    )
    p_query.add_argument(
        "--min-words",
        type=non_negative_int,
        metavar="N",
        help="keep only chunks of pages with at least N words (page.n_words >= N)",
    )
    p_query.add_argument(
        "--max-words",
        type=non_negative_int,
        metavar="N",
        help="keep only chunks of pages with at most N words (page.n_words <= N)",
    )
    p_query.add_argument(
        "--heading",
        metavar="PATTERN",
        help="keep only chunks whose section heading matches the SQL LIKE pattern, "
        "e.g. '%%History%%' (case- and accent-insensitive)",
    )
    p_query.add_argument(
        "--path",
        metavar="PATTERN",
        help="keep only chunks whose section path ('Heading > Subheading') matches the SQL "
        "LIKE pattern, e.g. 'Geography%%'",
    )
    p_query.add_argument(
        "--linked-from",
        metavar="TITLE",
        help="keep only chunks of pages that the page TITLE links to (exact title)",
    )
    p_query.add_argument(
        "--links-to",
        metavar="TITLE",
        help="keep only chunks of pages that link to the page TITLE (exact title)",
    )
    p_query.add_argument(
        "--title",
        dest="titles",
        action="append",
        metavar="TITLE",
        help="keep only chunks of the page TITLE (exact title); repeat for several pages",
    )
    p_query.add_argument(
        "--strategy",
        choices=search.STRATEGIES,
        default="inline",
        help=strategy_help("inline"),
    )
    p_query.add_argument(
        "--overfetch",
        type=positive_int,
        default=DEFAULT_OVERFETCH,
        metavar="N",
        help="overfetch strategy only: the index returns k x N candidates before the filters "
        f"run (default {DEFAULT_OVERFETCH})",
    )
    p_query.add_argument("--ef-search", type=ef_search_arg, metavar="N", help=EF_SEARCH_HELP)
    p_query.add_argument(
        "--sentences",
        action="store_true",
        help="also print, under each hit, the sentences of the chunk: the rows of "
        "chunk_sentence joined to sentence, in page order",
    )
    p_query.add_argument(
        "--explain", action="store_true", help="also print the SQL that ran and its EXPLAIN rows"
    )
    p_query.add_argument(
        "--json",
        action="store_true",
        help="print a JSON array of hits (with --explain: an object with hits, sql, explain, "
        "ef_search)",
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
        choices=search.STRATEGIES,
        default=evaluate.DEFAULT_STRATEGY,
        help=strategy_help(evaluate.DEFAULT_STRATEGY),
    )
    p_eval.add_argument(
        "--overfetch",
        type=positive_int,
        default=evaluate.DEFAULT_OVERFETCH,
        metavar="N",
        help="overfetch strategy only: the index returns k x N candidates before the filters "
        f"run (default {evaluate.DEFAULT_OVERFETCH})",
    )
    p_eval.add_argument("--ef-search", type=ef_search_arg, metavar="N", help=EF_SEARCH_HELP)
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


def _model_error(model_name: str, exc: OSError) -> CliError:
    """Return the ``CliError`` (exit 1) for a model that could not be loaded."""
    message = " ".join(str(exc).split())  # the Hub's messages span several lines
    return CliError(f"cannot load embedding model {model_name!r}: {message}")


def _load_model(embedder: Embedder) -> float:
    """Load the model by embedding :data:`WARM_UP_TEXT` and return the milliseconds it took.

    Called before a query is timed, so that the lazy model load (seconds from a cold process,
    including the first inference) is never reported as embedding time. A model that cannot be
    loaded raises ``OSError`` in sentence-transformers and becomes a ``CliError``.
    """
    start = time.perf_counter()
    try:
        embedder.embed_queries([WARM_UP_TEXT])
    except OSError as exc:
        raise _model_error(embedder.model_name, exc) from exc
    return (time.perf_counter() - start) * 1000.0


def _embed_query(embedder: Embedder, text: str) -> tuple[np.ndarray, float]:
    """Return ``(query vector, milliseconds)``.

    A model that cannot load, or whose vectors do not have the ``chunk.embedding`` dimension (a
    ``WIKILENSE_EMBEDDING_MODEL`` other than the one the corpus was ingested with), becomes a
    ``CliError`` with exit 1: it is the setup that is wrong, not the arguments.
    """
    start = time.perf_counter()
    try:
        vector = embedder.embed_queries([text])[0]
    except OSError as exc:
        raise _model_error(embedder.model_name, exc) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if vector.shape != (db.VECTOR_DIM,):
        raise CliError(
            f"embedding model {embedder.model_name!r} gives vectors of {vector.shape[-1]} "
            f"dimensions, but chunk.embedding is VECTOR({db.VECTOR_DIM}): set "
            "WIKILENSE_EMBEDDING_MODEL to the model the corpus was ingested with"
        )
    return vector, elapsed_ms


def _prepare_out_dir(out: Path) -> None:
    """Create ``--out`` (with parents) and check it is a writable directory, before the minutes of
    evaluation it would otherwise follow; ``CliError`` exit 2 naming the option otherwise."""
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CliError(
            f"--out {out}: cannot use it as the results directory: {exc.strerror or exc}",
            exit_code=EXIT_USAGE,
        ) from exc
    if not os.access(out, os.W_OK | os.X_OK):
        raise CliError(
            f"--out {out}: cannot use it as the results directory: not writable",
            exit_code=EXIT_USAGE,
        )


def _eval_ef_search(value: int | None) -> int | str | None:
    """Return ``evaluate.evaluate``'s ``ef_search`` for the parsed ``--ef-search``.

    Not given is None (``evaluate`` then takes ``settings.ef_search``, where 0 also means the
    server's value); ``search.EF_SEARCH_SERVER`` (0 / ``server``) is
    ``evaluate.SERVER_DEFAULT_EF_SEARCH``; any other value is passed on. ``evaluate`` reads a
    bare None as "the settings", so the CLI's 0 must not become None here.
    """
    if value is None:
        return None
    if value == search.EF_SEARCH_SERVER:
        return evaluate.SERVER_DEFAULT_EF_SEARCH
    return value


def _given_filter_flags(args: argparse.Namespace) -> list[str]:
    """Return the ``query`` filter flags that were given on the command line, in help order."""
    return [flag for attr, flag in FILTER_OPTIONS if getattr(args, attr) is not None]


def _filters_from_args(args: argparse.Namespace) -> Filters:
    """Return the ``search.Filters`` for the ``query`` options; an invalid combination is exit 2.

    A filter given with ``--strategy none`` is refused (the strategy would drop it silently).
    """
    try:
        filters = Filters(
            min_words=args.min_words,
            max_words=args.max_words,
            heading_like=args.heading,
            path_like=args.path,
            linked_from=args.linked_from,
            links_to=args.links_to,
            titles=args.titles,
        )
    except (TypeError, ValueError) as exc:
        raise CliError(str(exc), exit_code=EXIT_USAGE) from exc
    if args.strategy == STRATEGY_NO_FILTERS and not filters.is_empty():
        others = ", ".join(s for s in search.STRATEGIES if s != STRATEGY_NO_FILTERS)
        raise CliError(
            f"--strategy {STRATEGY_NO_FILTERS} runs without filters, but "
            f"{', '.join(_given_filter_flags(args))} given: drop the filter or choose "
            f"--strategy {others}",
            exit_code=EXIT_USAGE,
        )
    return filters


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
    section (empty path) shows the title alone, and a hit with an ``rrf`` score shows it too.
    """
    where = f"{hit.title} > {hit.section_path}" if hit.section_path else hit.title
    score = getattr(hit, "score", None)
    tail = f", rrf score {score:.4f}" if score is not None else ""
    head = (
        f"{rank:>2}. {hit.distance:.4f}  {where}  "
        f"[chunk_id {hit.chunk_id}, {hit.n_words} words{tail}]"
    )
    body = textwrap.fill(hit.text, width=width, initial_indent=INDENT, subsequent_indent=INDENT)
    return f"{head}\n{body}"


def format_sentences(rows: Sequence[tuple[str, str]], width: int = TEXT_WIDTH) -> str:
    """Return one hit's sentences as indented ``element_key  text`` lines wrapped at ``width``.

    ``rows`` are the ``(element_key, text)`` pairs of ``search.hit_sentences`` in page order;
    the keys are padded to one column so the join through ``chunk_sentence`` reads as a table.
    An empty list gives one line saying so.
    """
    if not rows:
        return f"{INDENT}sentences: none (no chunk_sentence rows)"
    key_width = max(len(key) for key, _ in rows)
    lines = [f"{INDENT}sentences (chunk_sentence -> sentence, page order):"]
    for key, text in rows:
        first = f"{INDENT}{INDENT}{key.ljust(key_width)}  "
        lines.append(
            textwrap.fill(
                text or "(empty)",
                width=width,
                initial_indent=first,
                subsequent_indent=" " * len(first),
            )
        )
    return "\n".join(lines)


def format_hits(
    hits: Sequence[Hit],
    width: int = TEXT_WIDTH,
    sentences: Mapping[int, Sequence[tuple[str, str]]] | None = None,
) -> str:
    """Return every hit from :func:`format_hit`, separated by blank lines, or ``"no hits"``.

    With ``sentences`` (``search.hit_sentences`` output keyed by chunk id) each hit is followed
    by its :func:`format_sentences` block.
    """
    if not hits:
        return "no hits"
    blocks = []
    for rank, hit in enumerate(hits, start=1):
        block = format_hit(rank, hit, width)
        if sentences is not None:
            block += "\n" + format_sentences(sentences.get(hit.chunk_id, []), width)
        blocks.append(block)
    return "\n\n".join(blocks)


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


def hits_to_json(
    hits: Sequence[Hit], sentences: Mapping[int, Sequence[tuple[str, str]]] | None = None
) -> list[dict[str, Any]]:
    """Return the hits as JSON-ready dicts with the ``Hit`` fields as keys.

    With ``sentences`` (``search.hit_sentences`` output) every dict also carries
    ``"sentences": [{"element_key": ..., "text": ...}, ...]`` in page order.
    """
    result = []
    for hit in hits:
        row = asdict(hit)
        if sentences is not None:
            row["sentences"] = [
                {"element_key": key, "text": text} for key, text in sentences.get(hit.chunk_id, [])
            ]
        result.append(row)
    return result


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
    """Run ``ingest.run_ingest`` with the given options and print the report; returns 0.

    ``run_ingest`` loads the model at its embedding stage; an ``OSError`` from that load is
    reported as one line. A missing corpus file (``FileNotFoundError``, raised before the
    model) keeps its own message.
    """
    settings = load_settings()
    with _stage_logging():
        try:
            report = ingest.run_ingest(
                settings,
                corpus_dir=args.corpus_dir,
                reset=not args.no_reset,
                batch_size=args.batch_size,
                use_prefix=not args.no_prefix,
                progress=sys.stderr.isatty(),
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise _model_error(settings.embedding_model, exc) from exc
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
    loaded; the model is loaded with a warm-up call before the query is timed. Timing goes to
    stderr so that stdout stays parseable with ``--json``.
    """
    settings = load_settings()
    filters = _filters_from_args(args)
    ef_search = search.resolve_ef_search(args.ef_search, settings.ef_search)
    conn = _connect(settings)
    sql = ""
    params: tuple[Any, ...] = ()
    explain_rows: list[dict[str, Any]] = []
    sentences: dict[int, list[tuple[str, str]]] | None = None
    try:
        _require_chunks(conn, settings)
        embedder = make_embedder(settings)
        load_ms = _load_model(embedder)
        qvec, embed_ms = _embed_query(embedder, args.text)
        start = time.perf_counter()
        try:
            hits = search.search(
                conn,
                qvec,
                k=args.k,
                filters=filters,
                strategy=args.strategy,
                overfetch=args.overfetch,
                ef_search=ef_search,
                query_text=args.text,
            )
        except (TypeError, ValueError) as exc:  # search() refused an argument
            raise CliError(str(exc), exit_code=EXIT_USAGE) from exc
        sql_ms = (time.perf_counter() - start) * 1000.0
        ef_effective = (
            db.get_session_var(conn, search.EF_SEARCH_VARIABLE) if ef_search is None else ef_search
        )
        if args.sentences:
            sentences = search.hit_sentences(conn, [hit.chunk_id for hit in hits])
        if args.explain:
            sql, params = search.search_statement(
                qvec, args.k, filters, args.strategy, args.overfetch, query_text=args.text
            )
            explain_rows = search.explain_search(
                conn, qvec, args.k, filters, args.strategy, args.overfetch, query_text=args.text
            )
    finally:
        conn.close()
    if args.json:
        payload: Any = hits_to_json(hits, sentences)
        if args.explain:
            payload = {"hits": payload, "sql": sql, "explain": explain_rows, "ef_search": ef_effective}
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_hits(hits, sentences=sentences))
        if args.explain:
            print()
            print(format_explain(sql, params, explain_rows))
    sys.stdout.flush()  # keep the order stable when stdout and stderr share a pipe
    ef_note = f"{ef_effective}" if ef_search is not None else f"{ef_effective} (server session value)"
    print(
        f"{len(hits)} hits; model load {load_ms:.0f} ms, embedding {embed_ms:.1f} ms, "
        f"SQL {sql_ms:.1f} ms ({embedder.model_name} on {embedder.device}, "
        f"strategy {args.strategy}, ef_search {ef_note})",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    """Run ``evaluate.evaluate`` over the corpus claims, write and summarise the results; returns 0.

    ``--out`` is created and checked first, then the database, before the model is loaded;
    ``evaluate``'s own argument errors (for example a database without claims), a model that
    cannot be loaded and a failed write of the results are reported as one line. ``ef_search`` defaults to ``settings.ef_search``; ``--ef-search 0`` leaves the
    server's session value.
    """
    settings = load_settings()
    _prepare_out_dir(args.out)
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
                ef_search=_eval_ef_search(args.ef_search),
                overfetch=args.overfetch,
                settings=settings,
            )
        except ValueError as exc:
            raise CliError(str(exc)) from exc
        except OSError as exc:
            raise _model_error(settings.embedding_model, exc) from exc
    finally:
        conn.close()
    try:
        json_path, md_path = evaluate.write_results(result, out_dir=args.out, name=args.name)
    except OSError as exc:
        raise CliError(f"cannot write the results to {args.out}: {exc.strerror or exc}") from exc
    print(format_eval_summary(result))
    print(f"results written: {json_path}, {md_path}")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    """Check the database, build the app (which loads the model), then run uvicorn; returns 0.

    ``web.create_app`` embeds a warm-up text so that the first request's timing excludes the
    model load; the load time it measured is printed. A model that cannot be loaded is
    reported as one line.
    """
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
    try:
        app = web.create_app(settings, embedder=embedder)
    except OSError as exc:
        raise _model_error(embedder.model_name, exc) from exc
    print(
        f"model loaded in {app.state.model_load_ms:.0f} ms; serving database "
        f"{settings.db_name!r} ({n_chunks} chunks) on http://{args.host}:{args.port}/ "
        f"(API docs at http://{args.host}:{args.port}{web.DOCS_URL})",
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
