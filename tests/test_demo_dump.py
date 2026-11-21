"""Tests of the demo database's dump, data/corpus/wikilense_demo.sql.gz, and the Makefile's
load-demo target.

The pure tests read the file: its pinned SHA-256, its ingest record against the committed corpus
and the default settings, its tables against the code's column list, and its vectors. The target's
commands are read with `make -n`, so those tests touch no database. One db test loads the dump into
the test database and runs the first example of sql/examples.sql on it.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import subprocess

import pytest

from wikilense import db as dbmod
from wikilense.config import (
    DEFAULT_CHUNK_MAX_WORDS,
    DEFAULT_CHUNK_OVERLAP_UNITS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_VECTOR_DIM,
    REPO_ROOT,
)

CORPUS = REPO_ROOT / "data" / "corpus"
DUMP = CORPUS / "wikilense_demo.sql.gz"
#: The committed file. `make demo-dump` writes the same bytes from the same database; a new file
#: (a new ingest) needs these two values and the size in docs/SETUP.md changed with it.
DUMP_SHA256 = "ad47247e95f67a03b68723020ab9d63297556b623e5f3322e5eebea043f37b6a"
DUMP_BYTES = 12_590_416
#: The ingest's report for the demo corpus (README, "Setup and platforms").
N_PAGES, N_CHUNKS, N_CLAIMS = 100, 4598, 75
VECTOR_KEY = "VECTOR KEY `embedding` (`embedding`) `M`='16' `DISTANCE`='cosine'"
CHECK_DATABASE = "wikilense_demo_check"

needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


@pytest.fixture(scope="module")
def dump_text() -> str:
    """The dump's SQL text."""
    return gzip.decompress(DUMP.read_bytes()).decode("utf-8")


def table_block(text: str, table: str) -> str:
    """Return the dump's section for ``table``: from its DROP TABLE to the next table's."""
    start = text.index(f"DROP TABLE IF EXISTS `{table}`;")
    end = text.find("DROP TABLE IF EXISTS `", start + 1)
    return text[start:] if end == -1 else text[start:end]


def ingest_record(text: str) -> dict[str, str]:
    """Return the dump's ingest_meta rows as a dict of key to value, as the dump writes them."""
    rows = re.findall(r"^\('(\w+)','((?:[^'\\]|\\.)*)'\)[,;]$", table_block(text, "ingest_meta"),
                      re.MULTILINE)
    return dict(rows)


def vector_hex(text: str) -> dict[int, str]:
    """Return each chunk row's embedding, as the dump writes it (hex digits after 0x)."""
    return {int(chunk_id): hexed for chunk_id, hexed in re.findall(
        r"^\((\d+),.*,0x([0-9A-F]+)\)[,;]$", table_block(text, "chunk"), re.MULTILINE)}


def dump_statements(text: str) -> list[str]:
    """Return the dump's statements in order, as the mariadb client would send them.

    mariadb-dump ends every statement with ';' at the end of a line and escapes newlines inside
    strings, so a statement is a run of lines up to one ending in ';'. Comment lines and the
    client-only sandbox line are skipped; executable comments (``/*!40014 SET ... */;``) are
    statements and stay, since they turn off the foreign key checks for the load.
    """
    statements: list[str] = []
    lines: list[str] = []
    for line in text.splitlines():
        if not lines and (not line.strip() or line.startswith(("--", "/*M!999999"))):
            continue
        lines.append(line)
        if line.endswith(";"):
            statements.append("\n".join(lines))
            lines = []
    assert not lines, "the dump ends inside a statement"
    return statements


def test_the_dump_is_the_pinned_file_and_setup_states_its_size() -> None:
    data = DUMP.read_bytes()
    assert len(data) == DUMP_BYTES
    assert hashlib.sha256(data).hexdigest() == DUMP_SHA256
    assert f"{DUMP_BYTES:,} bytes" in (REPO_ROOT / "docs" / "SETUP.md").read_text(encoding="utf-8")


def test_the_dump_was_made_from_the_committed_corpus_with_the_default_settings(
        dump_text: str) -> None:
    record = ingest_record(dump_text)
    for name in ("pages", "claims", "page_meta"):
        digest = hashlib.sha256((CORPUS / f"{name}.jsonl").read_bytes()).hexdigest()
        assert record[f"corpus_{name}_sha256"] == digest, name
    assert record["chunk_max_words"] == str(DEFAULT_CHUNK_MAX_WORDS)
    assert record["chunk_overlap_units"] == str(DEFAULT_CHUNK_OVERLAP_UNITS)
    assert record["embedding_model"] == DEFAULT_EMBEDDING_MODEL
    assert record["embedding_revision"] == DEFAULT_EMBEDDING_REVISION
    assert record["embedding_dim"] == str(DEFAULT_VECTOR_DIM)
    assert (record["index_m"], record["index_distance"]) == ("16", "cosine")


def test_the_dump_creates_the_tables_and_columns_of_the_schema(dump_text: str) -> None:
    tables = dict(re.findall(r"^CREATE TABLE `(\w+)` \((.*?)\n\) ENGINE=InnoDB", dump_text,
                             re.MULTILINE | re.DOTALL))
    assert set(tables) == set(dbmod.SCHEMA_COLUMNS)
    for table, body in tables.items():
        columns = tuple(re.findall(r"^  `(\w+)` ", body, re.MULTILINE))
        assert columns == dbmod.SCHEMA_COLUMNS[table], table
    assert f"`embedding` vector({DEFAULT_VECTOR_DIM}) NOT NULL" in tables["chunk"]
    assert VECTOR_KEY in tables["chunk"]


def test_every_chunk_has_its_whole_vector_and_the_dump_is_complete(dump_text: str) -> None:
    vectors = vector_hex(dump_text)
    assert sorted(vectors) == list(range(1, N_CHUNKS + 1))
    assert {len(hexed) for hexed in vectors.values()} == {2 * 4 * DEFAULT_VECTOR_DIM}
    assert dump_text.rstrip("\n").endswith("-- Dump completed")


def make_n(*args: str) -> list[str]:
    """Return the commands `make -n load-demo` prints for the given make arguments."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("MAKEFLAGS", "MFLAGS", "COMPOSE", "REPLACE", "DEMO_CHECK")}
    out = subprocess.run(["make", "-n", "-C", str(REPO_ROOT), "load-demo", *args],
                         env=env, check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "docker compose" in line]


@needs_make
def test_load_demo_checks_the_file_then_refuses_a_database_with_tables() -> None:
    default = make_n()
    assert default[0] == "docker compose exec -T mariadb gzip -t < data/corpus/wikilense_demo.sql.gz"
    assert "information_schema.tables" in default[1] and "REPLACE=1" in default[1]
    assert default[2].startswith("docker compose exec -T mariadb sh -c 'gzip -dc | ")
    assert all('"$MARIADB_DATABASE"' in line for line in default[1:])
    assert not any("CREATE DATABASE" in line for line in default)
    replace = make_n("REPLACE=1")
    assert replace == [default[0], *default[2:]]


@needs_make
def test_load_demo_check_mode_loads_only_into_the_scratch_database() -> None:
    lines = make_n("DEMO_CHECK=1")
    assert f"CREATE DATABASE IF NOT EXISTS {CHECK_DATABASE};" in lines[1]
    assert "MARIADB_ROOT_PASSWORD" in lines[1]
    assert all(f'"{CHECK_DATABASE}"' in line for line in lines[2:])
    assert not any("MARIADB_DATABASE" in line for line in lines)


@pytest.mark.db
def test_the_dump_loads_into_the_test_database_and_answers_the_first_example(
        db_conn: dbmod.pymysql.Connection, dump_text: str) -> None:
    with db_conn.cursor() as cur:
        for statement in dump_statements(dump_text):
            cur.execute(statement)
        cur.execute("SELECT (SELECT COUNT(*) FROM page), (SELECT COUNT(*) FROM chunk), "
                    "(SELECT COUNT(*) FROM claim)")
        assert cur.fetchone() == (N_PAGES, N_CHUNKS, N_CLAIMS)
        cur.execute("SHOW CREATE TABLE chunk")
        assert VECTOR_KEY in cur.fetchone()[1]
        cur.execute("SELECT HEX(embedding) FROM chunk WHERE chunk_id = 1")
        assert cur.fetchone()[0] == vector_hex(dump_text)[1]
        set_ef, first = dbmod.split_sql(
            (REPO_ROOT / "sql" / "examples.sql").read_text(encoding="utf-8"))[:2]
        assert set_ef.startswith("SET SESSION mhnsw_ef_search")
        cur.execute(set_ef)
        cur.execute(first)
        rows = cur.fetchall()
    assert len(rows) == 5
    assert rows[0][0] == 1 and rows[0][1] == pytest.approx(0.0, abs=1e-6)
    distances = [row[1] for row in rows]
    assert distances == sorted(distances)
