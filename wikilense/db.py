"""MariaDB access for WikiLense: connection, schema apply/reset, VECTOR parameters, bulk inserts.

Every statement is parameterised with ``%s``. The only identifiers that are ever interpolated
are table and column names taken from the fixed allowlist ``SCHEMA_COLUMNS`` below, which
mirrors ``sql/schema.sql`` (tests/test_db.py checks the two against each other).

The schema is read from ``sql/schema.sql`` in the repository checkout (``REPO_ROOT``, the parent
of the package directory), not from package data, so the package works only as an editable
install of the checkout (``pip install -e .``); a wheel would not carry the file, and
``apply_schema`` refuses with the expected path when the file is missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pymysql
import pymysql.cursors

from wikilense.config import REPO_ROOT, Settings, load_settings

SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"

#: Vector dimension written literally in sql/schema.sql (chunk.embedding VECTOR(384)).
VECTOR_DIM = 384

#: Tables in dependency order (parents first) with their columns, exactly as in sql/schema.sql.
#: insert_rows() accepts only these identifiers; drops run in the reverse order.
SCHEMA_COLUMNS: dict[str, tuple[str, ...]] = {
    "page": (
        "page_id",
        "title",
        "n_sentences",
        "n_items",
        "n_words",
        "n_chars",
        "n_sections",
        "n_tables",
        "n_lists",
    ),
    "section": ("section_id", "page_id", "ordinal", "heading", "level", "path"),
    "sentence": ("sentence_id", "page_id", "section_id", "element_key", "ordinal", "text"),
    "chunk": ("chunk_id", "page_id", "section_id", "ordinal", "text", "n_words", "embedding"),
    "chunk_sentence": ("chunk_id", "sentence_id"),
    "link": ("link_id", "from_page_id", "to_title", "to_page_id", "source_element"),
    "claim": ("claim_id", "split", "text", "label", "challenge"),
    "claim_evidence": (
        "claim_id",
        "evidence_set",
        "position",
        "element_id",
        "page_title",
        "element_type",
        "page_id",
        "sentence_id",
    ),
    "ingest_meta": ("key", "value"),
}

#: DROP statements, children first, built once from the fixed table list above.
_DROP_STATEMENTS: tuple[str, ...] = tuple(
    f"DROP TABLE IF EXISTS `{name}`" for name in reversed(SCHEMA_COLUMNS)
)

#: Session variables that set_session_var()/get_session_var() may touch, with their statements.
_SET_SESSION_SQL: dict[str, str] = {
    "mhnsw_ef_search": "SET SESSION mhnsw_ef_search = %s",
}
_GET_SESSION_SQL: dict[str, str] = {
    "mhnsw_ef_search": "SELECT @@SESSION.mhnsw_ef_search",
}
SESSION_VARIABLES: tuple[str, ...] = tuple(_SET_SESSION_SQL)


def table_names() -> tuple[str, ...]:
    """Return the table names of sql/schema.sql in dependency order (parents first)."""
    return tuple(SCHEMA_COLUMNS)


def connect(settings: Settings | None = None, database: str | None = None) -> pymysql.Connection:
    """Return an open PyMySQL connection (utf8mb4, autocommit off, default tuple cursors).

    ``settings`` defaults to ``load_settings()``; ``database`` overrides ``settings.db_name``
    (the test suite passes ``settings.test_db_name``). The connection is opened with
    ``binary_prefix=True`` so that a ``bytes`` parameter is sent as a ``_binary'...'`` literal,
    which is what a ``VECTOR`` column and ``VEC_DISTANCE_COSINE`` accept; without the prefix
    PyMySQL sends bytes as a utf8mb4 string and MariaDB rejects them ("Incorrect vector value").
    """
    if settings is None:
        settings = load_settings()
    return pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name if database is None else database,
        charset="utf8mb4",
        autocommit=False,
        binary_prefix=True,
    )


def split_sql(script: str) -> list[str]:
    """Return the statements of an SQL script, split on top-level semicolons, comments removed.

    Semicolons inside ``'...'`` and ``"..."`` strings (with backslash escapes and doubled
    quotes), inside backtick identifiers, in ``--`` and ``#`` line comments and in ``/* */``
    block comments do not split. Comment text is dropped; empty statements are skipped;
    surrounding whitespace is stripped. DELIMITER blocks are not supported (none are used).
    """
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(script)
    while i < n:
        ch = script[i]
        pair = script[i : i + 2]
        if ch == "#" or (pair == "--" and (i + 2 >= n or script[i + 2] in " \t\r\n")):
            end = script.find("\n", i)
            i = n if end < 0 else end  # keep the newline as statement whitespace
            continue
        if pair == "/*":
            end = script.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch in ("'", '"', "`"):
            j = i + 1
            while j < n:
                if script[j] == "\\" and ch != "`":
                    j += 2
                    continue
                if script[j] == ch:
                    if j + 1 < n and script[j + 1] == ch:
                        j += 2  # doubled quote inside the literal
                        continue
                    break
                j += 1
            buf.append(script[i : j + 1])
            i = j + 1
            continue
        if ch == ";":
            statement = "".join(buf).strip()
            if statement:
                statements.append(statement)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def drop_tables(conn: pymysql.Connection) -> int:
    """Drop every schema table that exists, children first, and return the number of DROPs run.

    DDL commits implicitly in MariaDB, so this ends any open transaction on ``conn``.
    """
    with conn.cursor() as cur:
        for statement in _DROP_STATEMENTS:
            cur.execute(statement)
    conn.commit()
    return len(_DROP_STATEMENTS)


def apply_schema(conn: pymysql.Connection, reset: bool = False) -> int:
    """Run sql/schema.sql statement by statement and return the number of statements executed.

    With ``reset=True`` the tables are dropped first (``drop_tables``), so the schema is
    recreated from scratch; otherwise ``CREATE TABLE IF NOT EXISTS`` leaves existing tables
    untouched, which makes a second call a no-op. DDL commits implicitly in MariaDB.

    Raises FileNotFoundError, naming the expected path, when ``sql/schema.sql`` is missing
    (the file lives in the repository checkout, so the package must be an editable install).
    Nothing is dropped in that case: the check runs before ``drop_tables``.
    """
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"schema file not found: {SCHEMA_PATH} (wikilense reads sql/schema.sql from the "
            "repository checkout and works only as an editable install: run "
            "'pip install -e .' from the repository root)"
        )
    if reset:
        drop_tables(conn)
    statements = split_sql(SCHEMA_PATH.read_text(encoding="utf-8"))
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)
    conn.commit()
    return len(statements)


def insert_sql(table: str, columns: Sequence[str]) -> str:
    """Return the parameterised ``INSERT INTO table (columns) VALUES (%s, ...)`` statement.

    ``table`` and every column must be in ``SCHEMA_COLUMNS`` and the columns must be distinct;
    otherwise ValueError. Identifiers are backtick-quoted (``key`` is a reserved word).
    """
    allowed = SCHEMA_COLUMNS.get(table)
    if allowed is None:
        raise ValueError(f"unknown table {table!r}; known tables: {', '.join(SCHEMA_COLUMNS)}")
    cols = tuple(columns)
    if not cols:
        raise ValueError(f"no columns given for table {table!r}")
    unknown = [c for c in cols if c not in allowed]
    if unknown:
        raise ValueError(f"unknown column(s) {unknown} for table {table!r}; allowed: {allowed}")
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate column names in {cols}")
    column_list = ", ".join(f"`{c}`" for c in cols)
    placeholders = ", ".join(["%s"] * len(cols))
    return f"INSERT INTO `{table}` ({column_list}) VALUES ({placeholders})"


def insert_rows(
    conn: pymysql.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> int:
    """Insert ``rows`` into ``table`` with ``executemany`` and return the number of rows inserted.

    ``table`` and ``columns`` are validated against ``SCHEMA_COLUMNS`` (ValueError otherwise).
    Each row is a sequence of values in the order of ``columns``; bind a VECTOR value with
    ``vec_param()``. Nothing is committed here: call ``conn.commit()`` when the batch is done.
    """
    statement = insert_sql(table, columns)
    batch = [tuple(row) for row in rows]
    if not batch:
        return 0
    width = len(columns)
    bad = next((row for row in batch if len(row) != width), None)
    if bad is not None:
        raise ValueError(f"row has {len(bad)} values for {width} columns: {bad!r}")
    with conn.cursor() as cur:
        cur.executemany(statement, batch)
        return int(cur.rowcount)


def vec_param(v: np.ndarray) -> bytes:
    """Return the bytes to bind for a ``VECTOR`` parameter: float32, little-endian, C order.

    Verified on MariaDB 11.8.9 with PyMySQL 2.2.8 (tests/test_db.py): with the
    ``binary_prefix=True`` that ``connect()`` sets, these bytes bound as a plain ``%s`` work
    both in ``INSERT ... VALUES (%s)`` and in ``VEC_DISTANCE_COSINE(embedding, %s)``, and the
    vector index is used. On a connection without ``binary_prefix`` the same bytes arrive as a
    utf8mb4 string and fail ("Incorrect vector value", "Illegal parameter data type varchar");
    the forms that work on any connection are ``UNHEX(%s)`` with ``vec_param(v).hex()`` and
    ``VEC_FromText(%s)`` with a JSON list of numbers.

    Raises ValueError when ``v`` is not one-dimensional or contains NaN or infinity.
    """
    arr = np.ascontiguousarray(v, dtype="<f4")
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("vector contains NaN or infinity")
    return arr.tobytes()


def vec_from_bytes(b: bytes) -> np.ndarray:
    """Return the float32 vector stored in a ``VECTOR`` column's raw bytes (little-endian).

    Raises ValueError when the length is not a multiple of 4.
    """
    if len(b) % 4:
        raise ValueError(f"vector bytes length {len(b)} is not a multiple of 4")
    return np.frombuffer(b, dtype="<f4").astype(np.float32)


def set_session_var(conn: pymysql.Connection, name: str, value: int) -> None:
    """Set a session variable from the allowlist (only ``mhnsw_ef_search``); returns None.

    The value is bound as a parameter; the variable name is one of ``SESSION_VARIABLES``.
    Raises ValueError for any other name and TypeError for a value that is not an int.
    MariaDB clamps an out-of-range value to the variable's limits with a warning (for
    ``mhnsw_ef_search`` the minimum is 1); it does not raise.
    """
    statement = _SET_SESSION_SQL.get(name)
    if statement is None:
        raise ValueError(f"session variable {name!r} is not allowed; allowed: {SESSION_VARIABLES}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} needs an int value, got {type(value).__name__}")
    with conn.cursor() as cur:
        cur.execute(statement, (value,))


def get_session_var(conn: pymysql.Connection, name: str) -> int:
    """Return the current session value of an allowlisted variable (only ``mhnsw_ef_search``).

    Raises ValueError for any other name.
    """
    statement = _GET_SESSION_SQL.get(name)
    if statement is None:
        raise ValueError(f"session variable {name!r} is not allowed; allowed: {SESSION_VARIABLES}")
    with conn.cursor() as cur:
        cur.execute(statement)
        row = cur.fetchone()
    return int(row[0])


def explain(
    conn: pymysql.Connection, sql: str, params: Sequence[Any] | None = None
) -> list[dict[str, Any]]:
    """Return the ``EXPLAIN`` rows of a parameterised query as dicts, one per plan row.

    Keys are MariaDB's columns: id, select_type, table, type, possible_keys, key, key_len,
    ref, rows, Extra. ``params`` are bound exactly as for the query itself.
    """
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("EXPLAIN " + sql, params)
        return [dict(row) for row in cur.fetchall()]
