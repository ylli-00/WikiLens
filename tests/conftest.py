"""Shared pytest fixtures: the Settings and a connection to the *test* database.

Tests marked ``db`` need a running MariaDB. They are skipped, with the reason, when no settings
can be loaded or the server does not accept a TCP connection within one second. The suite only
ever connects to ``settings.test_db_name`` and refuses to run when that equals the main database.
"""

from __future__ import annotations

import functools
import socket
from collections.abc import Iterator

import pytest

from wikilense import db as dbmod
from wikilense.config import Settings, SettingsError, load_settings

CONNECT_TIMEOUT_S = 1.0


@functools.lru_cache(maxsize=1)
def _db_state() -> tuple[Settings | None, str | None]:
    """Return (settings, None) when the test database is usable, else (settings or None, reason).

    Cached so that the socket check runs once per session.
    """
    try:
        settings = load_settings()
    except SettingsError as exc:
        return None, f"settings not available: {exc}"
    if settings.test_db_name == settings.db_name:
        return settings, (
            "refusing to run db tests: WIKILENSE_TEST_DB_NAME equals WIKILENSE_DB_NAME"
        )
    try:
        with socket.create_connection((settings.db_host, settings.db_port), CONNECT_TIMEOUT_S):
            pass
    except OSError:
        return settings, (
            f"MariaDB not reachable at {settings.db_host}:{settings.db_port} "
            f"within {CONNECT_TIMEOUT_S:g} s"
        )
    return settings, None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Add a skip marker with the reason to every ``db`` test when the server is not usable."""
    if not any(item.get_closest_marker("db") for item in items):
        return
    _, reason = _db_state()
    if reason is None:
        return
    for item in items:
        if item.get_closest_marker("db"):
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def settings() -> Settings:
    """The Settings from the environment and .env; skips the test when they cannot be loaded."""
    loaded, reason = _db_state()
    if loaded is None:
        pytest.skip(reason or "settings not available")
    return loaded


@pytest.fixture
def db_conn(settings: Settings) -> Iterator[dbmod.pymysql.Connection]:
    """A connection to the test database with a freshly reset schema; rolled back and closed.

    DDL commits implicitly, so the schema stays; only uncommitted test rows are rolled back.
    """
    _, reason = _db_state()
    if reason is not None:
        pytest.skip(reason)
    conn = dbmod.connect(settings, database=settings.test_db_name)
    try:
        dbmod.apply_schema(conn, reset=True)
        yield conn
    finally:
        try:
            conn.rollback()
        finally:
            conn.close()
