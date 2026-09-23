"""Tests for wikilense.config, wikilense.db and sql/schema.sql.

The pure tests run everywhere; the ``db`` tests need the MariaDB test database (see conftest).
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pymysql
import pytest

from wikilense import db as dbmod
from wikilense.config import Settings, SettingsError, load_settings

DIM = dbmod.VECTOR_DIM

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _unit(seed: int, dim: int = DIM) -> np.ndarray:
    """Return a deterministic unit-length float32 vector."""
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _angled(angle: float, axis: int, dim: int = DIM) -> np.ndarray:
    """Return the unit vector cos(angle) * e_0 + sin(angle) * e_axis (cosine distance to e_0
    is 1 - cos(angle))."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = np.cos(angle)
    v[axis] = np.sin(angle)
    return v


def _insert_page(conn: pymysql.Connection, page_id: int, title: str, n_words: int = 10) -> None:
    """Insert one page row with the given id and title and one lead section (id = page_id)."""
    dbmod.insert_rows(
        conn,
        "page",
        ("page_id", "title", "n_sentences", "n_items", "n_words", "n_chars", "n_sections",
         "n_tables", "n_lists"),
        [(page_id, title, 1, 0, n_words, 60, 1, 0, 0)],
    )
    dbmod.insert_rows(
        conn,
        "section",
        ("section_id", "page_id", "ordinal", "heading", "level", "path"),
        [(page_id, page_id, 0, "", 1, "")],
    )


def _insert_chunk(
    conn: pymysql.Connection, chunk_id: int, page_id: int, vector: np.ndarray, ordinal: int = 0
) -> None:
    """Insert one chunk row under the lead section of ``page_id`` with ``vector`` as bytes."""
    dbmod.insert_rows(
        conn,
        "chunk",
        ("chunk_id", "page_id", "section_id", "ordinal", "text", "n_words", "embedding"),
        [(chunk_id, page_id, page_id, ordinal, f"chunk {chunk_id}", 2, dbmod.vec_param(vector))],
    )


def _count(conn: pymysql.Connection, sql: str, params: tuple = ()) -> int:
    """Return the single integer of a COUNT(*) query."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.fetchone()[0])


def _clear_wikilense_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every WIKILENSE_* variable from the process environment for one test."""
    import os

    for name in [k for k in os.environ if k.startswith("WIKILENSE_")]:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------------------------
# config (no database)
# ---------------------------------------------------------------------------------------------


def test_load_settings_from_file_with_defaults(tmp_path: Path, monkeypatch) -> None:
    _clear_wikilense_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("WIKILENSE_DB_PASSWORD=pw-for-test\nWIKILENSE_DB_NAME=mydb\n", encoding="utf-8")
    s = load_settings(env)
    assert s.db_password == "pw-for-test"
    assert s.db_name == "mydb"
    assert s.test_db_name == "mydb_test"
    assert (s.db_host, s.db_port, s.db_user) == ("127.0.0.1", 3306, "wikilense")
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert (s.chunk_max_words, s.chunk_overlap_units, s.vector_dim) == (240, 1, 384)
    assert s.ef_search == 100


def test_environment_overrides_file(tmp_path: Path, monkeypatch) -> None:
    _clear_wikilense_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        "WIKILENSE_DB_PASSWORD=from-file\nWIKILENSE_DB_PORT=3306\nWIKILENSE_TEST_DB_NAME=t\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WIKILENSE_DB_PORT", "3307")
    monkeypatch.setenv("WIKILENSE_DB_PASSWORD", "from-env")
    s = load_settings(env)
    assert s.db_port == 3307
    assert s.db_password == "from-env"
    assert s.test_db_name == "t"


def test_missing_password_names_the_variable(tmp_path: Path, monkeypatch) -> None:
    _clear_wikilense_env(monkeypatch)
    with pytest.raises(SettingsError, match="WIKILENSE_DB_PASSWORD"):
        load_settings(tmp_path / "absent.env")


def test_bad_integer_names_the_variable(tmp_path: Path, monkeypatch) -> None:
    _clear_wikilense_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("WIKILENSE_DB_PASSWORD=x\nWIKILENSE_CHUNK_MAX_WORDS=many\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="WIKILENSE_CHUNK_MAX_WORDS"):
        load_settings(env)


def test_repr_hides_password_and_settings_are_frozen() -> None:
    s = Settings(db_password="top-secret-value")
    assert "top-secret-value" not in repr(s)
    assert "top-secret-value" not in str(s)
    with pytest.raises(AttributeError):
        s.db_port = 1  # type: ignore[misc]
    assert replace(s, db_port=1).db_port == 1


# ---------------------------------------------------------------------------------------------
# db helpers (no database)
# ---------------------------------------------------------------------------------------------


def test_split_sql_ignores_semicolons_in_comments_and_strings() -> None:
    script = """
    -- leading comment; with a semicolon
    # hash comment; too
    /* block; comment */
    CREATE TABLE a (x VARCHAR(10) DEFAULT 'a;b', y VARCHAR(10) DEFAULT "c;d", `we;ird` INT);
    INSERT INTO a VALUES ('it''s; fine', 'back\\'; slash', 1); /* trailing; */
    SELECT 1 -- not; here
    ;

    """
    statements = dbmod.split_sql(script)
    assert len(statements) == 3
    assert statements[0].startswith("CREATE TABLE a")
    assert "'a;b'" in statements[0] and '"c;d"' in statements[0] and "`we;ird`" in statements[0]
    assert statements[1] == "INSERT INTO a VALUES ('it''s; fine', 'back\\'; slash', 1)"
    assert statements[2] == "SELECT 1"
    assert dbmod.split_sql("") == []
    assert dbmod.split_sql(";;  ; -- x") == []


def test_schema_file_matches_the_fixed_table_list() -> None:
    script = dbmod.SCHEMA_PATH.read_text(encoding="utf-8")
    created = re.findall(r"^CREATE TABLE IF NOT EXISTS (\w+) \(", script, flags=re.MULTILINE)
    assert tuple(created) == dbmod.table_names()
    statements = dbmod.split_sql(script)
    assert len(statements) == len(created)
    assert all(s.startswith("CREATE TABLE IF NOT EXISTS") for s in statements)
    assert "DELIMITER" not in script
    assert re.search(r"embedding\s+VECTOR\(384\) NOT NULL", script)
    assert "VECTOR INDEX (embedding) M=16 DISTANCE=cosine" in script
    for column in ("title", "to_title", "page_title"):
        pattern = rf"^\s+{column}\s+VARCHAR\(\d+\) COLLATE utf8mb4_bin"
        assert re.search(pattern, script, flags=re.MULTILINE), column
    # a comment line directly above every CREATE TABLE
    for match in re.finditer(r"^CREATE TABLE IF NOT EXISTS", script, flags=re.MULTILINE):
        before = script[: match.start()].rstrip("\n").rsplit("\n", 1)[-1]
        assert before.startswith("-- "), match.group(0)


def test_vec_param_and_vec_from_bytes_round_trip_in_memory() -> None:
    v = _unit(1)
    b = dbmod.vec_param(v)
    assert isinstance(b, bytes) and len(b) == DIM * 4
    assert b == v.astype("<f4").tobytes()
    back = dbmod.vec_from_bytes(b)
    assert back.dtype == np.float32 and back.shape == (DIM,)
    assert np.array_equal(back, v)
    assert dbmod.vec_param(v.astype(np.float64)) == b  # float64 input is converted
    assert dbmod.vec_param([1.0, 2.0]) == np.array([1.0, 2.0], dtype="<f4").tobytes()
    with pytest.raises(ValueError):
        dbmod.vec_param(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        dbmod.vec_param(np.array([1.0, np.nan], dtype=np.float32))
    with pytest.raises(ValueError):
        dbmod.vec_from_bytes(b"\x00\x00\x00")


def test_vec_param_writes_float32_little_endian_from_any_input() -> None:
    values = [1.0, -2.5, 0.125, 3.0e-3]
    assert dbmod.vec_param(np.array(values, dtype=np.float32)) == struct.pack("<4f", *values)
    # known bit patterns: 1.0f is 0x3F800000, written least significant byte first
    assert dbmod.vec_param([1.0]) == b"\x00\x00\x80\x3f"
    assert dbmod.vec_param([-2.5]) == b"\x00\x00\x20\xc0"
    # a big-endian float32 input is converted, not copied byte for byte
    assert dbmod.vec_param(np.array([1.0, 2.0], dtype=">f4")) == struct.pack("<2f", 1.0, 2.0)


def test_vec_from_bytes_accepts_any_bytes_like_and_returns_a_writable_copy() -> None:
    packed = struct.pack("<3f", 1.0, 2.0, 3.0)
    for raw in (packed, bytearray(packed), memoryview(packed)):
        back = dbmod.vec_from_bytes(raw)
        assert np.array_equal(back, [1.0, 2.0, 3.0]) and back.flags.writeable
    assert dbmod.vec_from_bytes(b"").shape == (0,)


def test_insert_sql_validates_identifiers() -> None:
    sql = dbmod.insert_sql("ingest_meta", ("key", "value"))
    assert sql == "INSERT INTO `ingest_meta` (`key`, `value`) VALUES (%s, %s)"
    with pytest.raises(ValueError, match="unknown table"):
        dbmod.insert_sql("page; DROP TABLE page", ("title",))
    with pytest.raises(ValueError, match="unknown table"):
        dbmod.insert_sql("users", ("title",))
    with pytest.raises(ValueError, match="unknown column"):
        dbmod.insert_sql("page", ("title", "n_words) VALUES (1"))
    with pytest.raises(ValueError, match="duplicate"):
        dbmod.insert_sql("page", ("title", "title"))
    with pytest.raises(ValueError, match="no columns"):
        dbmod.insert_sql("page", ())


def test_session_var_allowlist_is_checked_before_any_sql() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        dbmod.set_session_var(None, "max_allowed_packet", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not allowed"):
        dbmod.get_session_var(None, "mhnsw_max_cache_size")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        dbmod.set_session_var(None, "mhnsw_ef_search", "40")  # type: ignore[arg-type]
    assert dbmod.SESSION_VARIABLES == ("mhnsw_ef_search",)


# ---------------------------------------------------------------------------------------------
# database tests
# ---------------------------------------------------------------------------------------------


@pytest.mark.db
def test_connection_uses_the_test_database(db_conn, settings) -> None:
    with db_conn.cursor() as cur:
        cur.execute("SELECT DATABASE(), @@autocommit, @@character_set_client")
        database, autocommit, charset = cur.fetchone()
    assert database == settings.test_db_name
    assert database != settings.db_name
    assert autocommit == 0
    assert charset == "utf8mb4"


@pytest.mark.db
def test_schema_applies_twice_without_error(db_conn) -> None:
    n_first = dbmod.apply_schema(db_conn)  # the fixture already applied it with reset=True
    n_second = dbmod.apply_schema(db_conn)
    assert n_first == n_second == len(dbmod.table_names())
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME"
        )
        present = sorted(row[0] for row in cur.fetchall())
    assert present == sorted(dbmod.table_names())


@pytest.mark.db
def test_reset_drops_and_recreates(db_conn) -> None:
    _insert_page(db_conn, 1, "Reset me")
    db_conn.commit()
    assert _count(db_conn, "SELECT COUNT(*) FROM page") == 1
    dbmod.apply_schema(db_conn, reset=True)
    assert _count(db_conn, "SELECT COUNT(*) FROM page") == 0
    assert _count(db_conn, "SELECT COUNT(*) FROM section") == 0
    assert dbmod.drop_tables(db_conn) == len(dbmod.table_names())
    assert (
        _count(
            db_conn,
            "SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE()",
        )
        == 0
    )
    dbmod.apply_schema(db_conn)
    assert _count(db_conn, "SELECT COUNT(*) FROM chunk") == 0


@pytest.mark.db
def test_fixed_column_list_matches_the_database(db_conn) -> None:
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME, ORDINAL_POSITION"
        )
        rows = cur.fetchall()
    live: dict[str, list[str]] = {}
    for table, column in rows:
        live.setdefault(table, []).append(column)
    assert {t: list(c) for t, c in dbmod.SCHEMA_COLUMNS.items()} == live
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COLUMN_NAME, COLUMN_TYPE, COLLATION_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
            ("chunk", "embedding"),
        )
        assert cur.fetchone() == ("embedding", f"vector({DIM})", None)
        cur.execute(
            "SELECT TABLE_NAME, COLUMN_NAME, COLLATION_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND COLLATION_NAME = %s ORDER BY 1, 2",
            ("utf8mb4_bin",),
        )
        assert cur.fetchall() == (
            ("claim_evidence", "page_title", "utf8mb4_bin"),
            ("link", "to_title", "utf8mb4_bin"),
            ("page", "title", "utf8mb4_bin"),
        )
        cur.execute(
            "SELECT INDEX_TYPE FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
            ("chunk", "embedding"),
        )
        assert cur.fetchone() == ("VECTOR",)


@pytest.mark.db
def test_vector_round_trip_bytes_unhex_and_vec_fromtext(db_conn) -> None:
    _insert_page(db_conn, 1, "Vectors")
    v = _unit(7)
    # form 1: plain bytes parameter (PyMySQL binary_prefix -> _binary'...')
    _insert_chunk(db_conn, 1, 1, v, ordinal=0)
    # form 2: VEC_FromText(%s) with the JSON text that embedding.vector_to_text produces
    text = json.dumps([float(x) for x in v])
    # form 3: UNHEX(%s) with the hex of the same bytes
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunk (chunk_id, page_id, section_id, ordinal, text, n_words, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s, VEC_FromText(%s))",
            (2, 1, 1, 1, "chunk 2", 2, text),
        )
        cur.execute(
            "INSERT INTO chunk (chunk_id, page_id, section_id, ordinal, text, n_words, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s, UNHEX(%s))",
            (3, 1, 1, 2, "chunk 3", 2, dbmod.vec_param(v).hex()),
        )
        cur.execute("SELECT chunk_id, VEC_ToText(embedding), embedding FROM chunk ORDER BY chunk_id")
        rows = cur.fetchall()
    assert [r[0] for r in rows] == [1, 2, 3]
    for _, as_text, raw in rows:
        assert isinstance(raw, bytes) and len(raw) == DIM * 4
        assert np.array_equal(dbmod.vec_from_bytes(raw), v)  # the raw column is bit-exact
        parsed = np.array(json.loads(as_text), dtype=np.float32)
        np.testing.assert_array_almost_equal(parsed, v, decimal=6)  # VEC_ToText rounds
    with db_conn.cursor() as cur:  # VEC_DISTANCE_COSINE also accepts the bytes parameter
        cur.execute(
            "SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) FROM chunk ORDER BY chunk_id",
            (dbmod.vec_param(v),),
        )
        assert all(abs(d) < 1e-6 for _, d in cur.fetchall())


def _driver_prefixes_bytes() -> bool:
    """Return True when the installed PyMySQL escapes bytes as a ``_binary`` literal by itself.

    PyMySQL 1.2.3 does (``_binary X'...'``); 1.2.0 to 1.2.2 send bytes as a utf8mb4 string unless
    the connection is opened with ``binary_prefix=True``, which ``db.connect`` sets.
    """
    try:
        return pymysql.converters.escape_bytes(b"\x00\x80").startswith("_binary")
    except UnicodeEncodeError:  # 1.2.0 to 1.2.2 cannot even escape non-UTF-8 bytes
        return False


@pytest.mark.db
def test_bytes_parameter_binding_depends_on_the_driver(settings) -> None:
    """Documents the binding rule for a connection opened WITHOUT binary_prefix.

    With PyMySQL 1.2.3 and later the bytes arrive as ``_binary X'...'`` and work; with older
    drivers they arrive as a varchar and MariaDB rejects them. ``db.connect`` sets
    ``binary_prefix=True`` so the code works with either (test_vector_round_trip_bytes).
    """
    conn = pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.test_db_name,
        charset="utf8mb4",
        autocommit=False,
    )
    try:
        dbmod.apply_schema(conn, reset=True)
        _insert_page(conn, 1, "No prefix")
        conn.commit()
        if _driver_prefixes_bytes():
            _insert_chunk(conn, 1, 1, _unit(3))
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT VEC_DISTANCE_COSINE(embedding, %s) FROM chunk",
                    (dbmod.vec_param(_unit(3)),),
                )
                assert abs(cur.fetchone()[0]) < 1e-6
            conn.rollback()
        else:
            with pytest.raises(pymysql.err.OperationalError, match="Incorrect vector value"):
                _insert_chunk(conn, 1, 1, _unit(3))
            conn.rollback()
            with (
                pytest.raises(pymysql.err.OperationalError, match="Illegal parameter data type"),
                conn.cursor() as cur,
            ):
                cur.execute(
                    "SELECT VEC_DISTANCE_COSINE(embedding, %s) FROM chunk",
                    (dbmod.vec_param(_unit(3)),),
                )
        with conn.cursor() as cur:  # the text form works on any connection
            cur.execute(
                "INSERT INTO chunk (chunk_id, page_id, section_id, ordinal, text, n_words, "
                "embedding) VALUES (%s, %s, %s, %s, %s, %s, VEC_FromText(%s))",
                (1, 1, 1, 0, "c", 1, json.dumps([float(x) for x in _unit(3)])),
            )
        conn.rollback()
        dbmod.apply_schema(conn, reset=True)  # leave the test database empty
    finally:
        conn.close()


@pytest.mark.db
def test_knn_order_and_explain_use_the_vector_index(db_conn) -> None:
    _insert_page(db_conn, 1, "Nearest")
    # angles to the query direction e_0, keyed by chunk_id: expected order 4, 2, 5, 1, 3
    angles = {1: 1.3, 2: 0.5, 3: 1.7, 4: 0.1, 5: 0.9}
    for chunk_id, angle in angles.items():
        _insert_chunk(db_conn, chunk_id, 1, _angled(angle, axis=chunk_id), ordinal=chunk_id)
    query = _angled(0.0, axis=1)  # e_0
    sql = (
        "SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance "
        "FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT 2"
    )
    params = (dbmod.vec_param(query), dbmod.vec_param(query))
    with db_conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    assert [r[0] for r in rows] == [4, 2]
    np.testing.assert_allclose(
        [r[1] for r in rows], [1 - np.cos(0.1), 1 - np.cos(0.5)], atol=1e-6
    )
    plan = dbmod.explain(db_conn, sql, params)
    assert len(plan) == 1, plan
    row = plan[0]
    assert row["table"] == "chunk", row
    assert row["key"] == "embedding", row
    assert row["type"] == "index", row
    assert set(row) >= {"id", "select_type", "table", "type", "key", "key_len", "rows", "Extra"}


@pytest.mark.db
def test_set_session_var_changes_ef_search_for_this_session_only(db_conn, settings) -> None:
    before = dbmod.get_session_var(db_conn, "mhnsw_ef_search")  # the server's default, 20 unless
    assert before >= 1  # the server was started with another value
    target = before + 30
    dbmod.set_session_var(db_conn, "mhnsw_ef_search", target)
    assert dbmod.get_session_var(db_conn, "mhnsw_ef_search") == target
    other = dbmod.connect(settings, database=settings.test_db_name)
    try:
        assert dbmod.get_session_var(other, "mhnsw_ef_search") == before
    finally:
        other.close()
    with pytest.raises(ValueError):
        dbmod.set_session_var(db_conn, "mhnsw_max_cache_size", 1)
    with pytest.raises(ValueError):
        dbmod.set_session_var(db_conn, "mhnsw_ef_search; SET GLOBAL x = 1", 1)
    dbmod.set_session_var(db_conn, "mhnsw_ef_search", 0)  # MariaDB clamps to the minimum, 1
    assert dbmod.get_session_var(db_conn, "mhnsw_ef_search") == 1


def _knn(conn: pymysql.Connection, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return ``(chunk_id, distance)`` of the ``k`` nearest chunks through the vector index."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance "
            "FROM chunk ORDER BY VEC_DISTANCE_COSINE(embedding, %s) LIMIT %s",
            (dbmod.vec_param(query), dbmod.vec_param(query), k),
        )
        return [(int(cid), float(d)) for cid, d in cur.fetchall()]


@pytest.mark.db
def test_vector_index_follows_the_transaction(db_conn, settings) -> None:
    """The HNSW index lives in InnoDB: uncommitted rows are visible to their own transaction
    only, a rollback takes them out of the index, and UPDATE / DELETE of a row change the
    k-nearest result at once."""
    _insert_page(db_conn, 1, "Transactions")
    _insert_chunk(db_conn, 1, 1, _angled(0.5, axis=1), ordinal=1)
    _insert_chunk(db_conn, 2, 1, _angled(0.9, axis=2), ordinal=2)
    db_conn.commit()
    query = _angled(0.0, axis=1)  # e_0
    assert [cid for cid, _ in _knn(db_conn, query, 5)] == [1, 2]

    # 1. an INSERT inside the open transaction: the same connection sees it in the index ...
    _insert_chunk(db_conn, 3, 1, _angled(0.1, axis=3), ordinal=3)
    assert [cid for cid, _ in _knn(db_conn, query, 5)] == [3, 1, 2]
    other = dbmod.connect(settings, database=settings.test_db_name)
    try:
        # ... another connection does not, and after the rollback nobody does
        assert [cid for cid, _ in _knn(other, query, 5)] == [1, 2]
        db_conn.rollback()
        assert [cid for cid, _ in _knn(db_conn, query, 5)] == [1, 2]
        assert _count(db_conn, "SELECT COUNT(*) FROM chunk") == 2

        # 2. an UPDATE of one embedding: the new distance shows and the order changes
        with db_conn.cursor() as cur:
            cur.execute(
                "UPDATE chunk SET embedding = %s WHERE chunk_id = %s",
                (dbmod.vec_param(_angled(0.05, axis=2)), 2),
            )
        rows = _knn(db_conn, query, 5)
        assert [cid for cid, _ in rows] == [2, 1]
        np.testing.assert_allclose(
            [d for _, d in rows], [1 - np.cos(0.05), 1 - np.cos(0.5)], atol=1e-6
        )
        assert [cid for cid, _ in _knn(other, query, 5)] == [1, 2]  # not committed yet
        db_conn.commit()
        other.commit()  # end the other connection's snapshot so that it sees the commit
        assert [cid for cid, _ in _knn(other, query, 5)] == [2, 1]

        # 3. a DELETE removes the chunk from the k-nearest result
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM chunk WHERE chunk_id = %s", (1,))
        assert _knn(db_conn, query, 5) == [(2, pytest.approx(1 - np.cos(0.05), abs=1e-6))]
        db_conn.commit()
        other.commit()
        assert [cid for cid, _ in _knn(other, query, 5)] == [2]
    finally:
        other.close()


@pytest.mark.db
def test_insert_rows_rejects_unknown_table_and_column(db_conn) -> None:
    with pytest.raises(ValueError, match="unknown table"):
        dbmod.insert_rows(db_conn, "pages", ("title",), [("x",)])
    with pytest.raises(ValueError, match="unknown column"):
        dbmod.insert_rows(db_conn, "page", ("title", "password"), [("x", "y")])
    with pytest.raises(ValueError, match="values for"):
        dbmod.insert_rows(db_conn, "ingest_meta", ("key", "value"), [("k",)])
    assert dbmod.insert_rows(db_conn, "ingest_meta", ("key", "value"), []) == 0
    n = dbmod.insert_rows(db_conn, "ingest_meta", ("key", "value"), [("a", "1"), ("b", "2")])
    assert n == 2
    with db_conn.cursor() as cur:
        cur.execute("SELECT `key`, `value` FROM ingest_meta ORDER BY `key`")
        assert cur.fetchall() == (("a", "1"), ("b", "2"))


@pytest.mark.db
def test_fk_cascade_from_page_deletes_children(db_conn) -> None:
    _insert_page(db_conn, 1, "Parent")
    _insert_page(db_conn, 2, "Other")
    dbmod.insert_rows(
        db_conn,
        "sentence",
        ("sentence_id", "page_id", "section_id", "element_key", "ordinal", "text"),
        [(1, 1, 1, "sentence_0", 0, "First."), (2, 1, 1, "item_0_0", 1, "Second.")],
    )
    _insert_chunk(db_conn, 1, 1, _unit(11))
    dbmod.insert_rows(db_conn, "chunk_sentence", ("chunk_id", "sentence_id"), [(1, 1), (1, 2)])
    dbmod.insert_rows(
        db_conn,
        "link",
        ("from_page_id", "to_title", "to_page_id", "source_element"),
        [(1, "Other", 2, "sentence_0"), (2, "Parent", 1, "sentence_0")],
    )
    dbmod.insert_rows(
        db_conn,
        "claim",
        ("claim_id", "split", "text", "label", "challenge"),
        [(100, "dev", "A claim.", "REFUTES", "Other")],
    )
    dbmod.insert_rows(
        db_conn,
        "claim_evidence",
        ("claim_id", "evidence_set", "position", "element_id", "page_title", "element_type",
         "page_id", "sentence_id"),
        [(100, 0, 0, "Parent_sentence_0", "Parent", "sentence", 1, 1)],
    )
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM page WHERE page_id = %s", (1,))
    assert _count(db_conn, "SELECT COUNT(*) FROM section WHERE page_id = %s", (1,)) == 0
    assert _count(db_conn, "SELECT COUNT(*) FROM sentence") == 0
    assert _count(db_conn, "SELECT COUNT(*) FROM chunk") == 0
    assert _count(db_conn, "SELECT COUNT(*) FROM chunk_sentence") == 0
    assert _count(db_conn, "SELECT COUNT(*) FROM link WHERE from_page_id = %s", (1,)) == 0
    # references that only resolve a title are set to NULL, the rows stay
    assert _count(db_conn, "SELECT COUNT(*) FROM link WHERE from_page_id = %s", (2,)) == 1
    assert _count(db_conn, "SELECT COUNT(*) FROM link WHERE to_page_id IS NULL") == 1
    assert _count(db_conn, "SELECT COUNT(*) FROM claim_evidence") == 1
    assert (
        _count(
            db_conn,
            "SELECT COUNT(*) FROM claim_evidence WHERE page_id IS NULL AND sentence_id IS NULL",
        )
        == 1
    )
    assert _count(db_conn, "SELECT COUNT(*) FROM page") == 1


@pytest.mark.db
def test_wrong_dimension_raises_a_database_error(db_conn) -> None:
    _insert_page(db_conn, 1, "Short vector")
    with pytest.raises(pymysql.err.DatabaseError):
        _insert_chunk(db_conn, 1, 1, _unit(5, dim=10))
    db_conn.rollback()
    with pytest.raises(pymysql.err.DatabaseError), db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunk (chunk_id, page_id, section_id, ordinal, text, n_words, "
            "embedding) VALUES (%s, %s, %s, %s, %s, %s, VEC_FromText(%s))",
            (1, 1, 1, 0, "c", 1, json.dumps([0.1] * 10)),
        )


@pytest.mark.db
def test_titles_match_exactly_with_utf8mb4_bin(db_conn) -> None:
    _insert_page(db_conn, 1, "Paris")
    _insert_page(db_conn, 2, "paris")
    _insert_page(db_conn, 3, "Pâris")
    with db_conn.cursor() as cur:
        cur.execute("SELECT page_id FROM page WHERE title = %s", ("Paris",))
        assert cur.fetchall() == ((1,),)
        cur.execute("SELECT page_id FROM page WHERE title = %s", ("pâris",))
        assert cur.fetchall() == ()
        cur.execute("SELECT COUNT(*) FROM section WHERE heading LIKE %s", ("",))
        assert cur.fetchone() == (3,)
