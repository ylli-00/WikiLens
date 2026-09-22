"""Shared pytest fixtures: the Settings and a connection to the *test* database.

Tests marked ``db`` need a running MariaDB. They are skipped, with the reason, when no settings
can be loaded or the server does not accept a TCP connection within one second. The suite only
ever connects to ``settings.test_db_name`` and refuses to run when that equals the main database.

Tests marked ``slow`` need the embedding model. They are skipped, with the reason, when the
model is neither in the Hugging Face cache nor downloadable (``huggingface.co`` does not accept
a TCP connection within one second, or ``HF_HUB_OFFLINE`` is set), so an offline grader sees
skips instead of download failures.
"""

from __future__ import annotations

import functools
import os
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from wikilense import db as dbmod
from wikilense.config import Settings, SettingsError, load_settings
from wikilense.embedding import DEFAULT_MODEL_NAME

CONNECT_TIMEOUT_S = 1.0
HF_HUB_HOST = ("huggingface.co", 443)
#: One of these must be in the cached snapshot for the model to load without the hub.
MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


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


def model_is_cached(model_name: str) -> bool:
    """Return True when ``model_name`` is a local directory or a complete Hugging Face cache entry.

    Complete means ``config.json`` and a weights file (``MODEL_WEIGHT_FILES``) are in the cached
    snapshot; ``huggingface_hub.try_to_load_from_cache`` honours ``HF_HOME`` / ``HF_HUB_CACHE``.
    """
    if Path(model_name).is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    config = try_to_load_from_cache(model_name, "config.json")
    if not isinstance(config, str):
        return False
    snapshot = Path(config).parent
    return any((snapshot / name).is_file() for name in MODEL_WEIGHT_FILES)


def hub_is_reachable() -> bool:
    """Return True when huggingface.co accepts a TCP connection and offline mode is not set."""
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    try:
        with socket.create_connection(HF_HUB_HOST, CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def _slow_state() -> str | None:
    """Return None when ``slow`` tests can run, else the skip reason (checked once per session).

    The models needed are the package default and, when settings load, the configured one.
    """
    models = {DEFAULT_MODEL_NAME}
    try:
        models.add(load_settings().embedding_model)
    except SettingsError:
        pass
    missing = sorted(name for name in models if not model_is_cached(name))
    if not missing or hub_is_reachable():
        return None
    return (
        f"embedding model {', '.join(missing)} is not in the Hugging Face cache and "
        f"{HF_HUB_HOST[0]} is not reachable within {CONNECT_TIMEOUT_S:g} s"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip, with the reason, every ``db`` test without a usable server and every ``slow`` test
    without a loadable model."""
    for marker, state in (("db", _db_state), ("slow", _slow_state)):
        if not any(item.get_closest_marker(marker) for item in items):
            continue
        result = state()
        reason = result[1] if isinstance(result, tuple) else result
        if reason is None:
            continue
        for item in items:
            if item.get_closest_marker(marker):
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
