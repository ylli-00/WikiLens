"""Settings for WikiLense, read from the environment and the repository's ``.env`` file.

Every setting is a ``WIKILENSE_*`` variable. Real environment variables win over the ``.env``
file; the file is read with python-dotenv but never written into ``os.environ``, so loading is
side-effect free and repeatable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # chosen in phase 1, see docs/DESIGN.md
DEFAULT_CHUNK_MAX_WORDS = 240  # chosen with results/SUMMARY.md: equal recall at equal retrieved text, half the vectors
DEFAULT_CHUNK_OVERLAP_UNITS = 1  # one unit of overlap, kept through the experiments
DEFAULT_VECTOR_DIM = 384  # the model's dimension; sql/schema.sql writes it literally
DEFAULT_INDEX_M = 16  # chosen with results/SUMMARY.md: exact-ranking recall at the default ef_search
DEFAULT_EF_SEARCH = 100  # mhnsw_ef_search applied per query by the CLI, web page and harness


class SettingsError(ValueError):
    """A required setting is missing or a value has the wrong type; the message names it."""


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Connection, model, chunking and index parameters. The password is kept out of repr()."""

    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "wikilense"
    db_password: str = field(repr=False)
    db_name: str = "wikilense"
    test_db_name: str = "wikilense_test"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_max_words: int = DEFAULT_CHUNK_MAX_WORDS
    chunk_overlap_units: int = DEFAULT_CHUNK_OVERLAP_UNITS
    vector_dim: int = DEFAULT_VECTOR_DIM
    index_m: int = DEFAULT_INDEX_M
    ef_search: int = DEFAULT_EF_SEARCH


def _get_str(env: Mapping[str, str], name: str, default: str | None) -> str:
    """Return env[name], or default when the variable is unset or empty.

    Raises SettingsError naming the variable when it is unset and there is no default.
    """
    value = env.get(name, "")
    if value != "":
        return value
    if default is None:
        raise SettingsError(
            f"{name} is not set: add it to {DEFAULT_ENV_FILE.name} (see .env.example) "
            "or export it in the environment"
        )
    return default


def _get_int(env: Mapping[str, str], name: str, default: int) -> int:
    """Return env[name] as an int, or default when unset or empty.

    Raises SettingsError naming the variable when the value is not an integer.
    """
    value = env.get(name, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer, got {value!r}") from exc


def load_settings(env_file: str | os.PathLike[str] | None = None) -> Settings:
    """Return the Settings from the environment, with ``.env`` values as the fallback.

    ``env_file`` is the dotenv file to read; ``None`` means the repository-root ``.env``.
    A file that does not exist is simply skipped. Real environment variables take precedence
    over the file. Only ``WIKILENSE_DB_PASSWORD`` has no default; the other variables fall back
    to the values in ``.env.example`` and docs/DESIGN.md. ``WIKILENSE_TEST_DB_NAME`` defaults to
    the database name plus ``_test``.

    Raises SettingsError, naming the variable, when a required one is missing or an integer
    variable is malformed.
    """
    path = DEFAULT_ENV_FILE if env_file is None else Path(env_file)
    file_values: dict[str, str] = {}
    if path.is_file():
        file_values = {k: v for k, v in dotenv_values(path).items() if v is not None}
    env: dict[str, str] = {**file_values, **os.environ}

    db_name = _get_str(env, "WIKILENSE_DB_NAME", "wikilense")
    return Settings(
        db_host=_get_str(env, "WIKILENSE_DB_HOST", "127.0.0.1"),
        db_port=_get_int(env, "WIKILENSE_DB_PORT", 3306),
        db_user=_get_str(env, "WIKILENSE_DB_USER", "wikilense"),
        db_password=_get_str(env, "WIKILENSE_DB_PASSWORD", None),
        db_name=db_name,
        test_db_name=_get_str(env, "WIKILENSE_TEST_DB_NAME", f"{db_name}_test"),
        embedding_model=_get_str(env, "WIKILENSE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        chunk_max_words=_get_int(env, "WIKILENSE_CHUNK_MAX_WORDS", DEFAULT_CHUNK_MAX_WORDS),
        chunk_overlap_units=_get_int(
            env, "WIKILENSE_CHUNK_OVERLAP_UNITS", DEFAULT_CHUNK_OVERLAP_UNITS
        ),
        vector_dim=_get_int(env, "WIKILENSE_VECTOR_DIM", DEFAULT_VECTOR_DIM),
        index_m=_get_int(env, "WIKILENSE_INDEX_M", DEFAULT_INDEX_M),
        ef_search=_get_int(env, "WIKILENSE_EF_SEARCH", DEFAULT_EF_SEARCH),
    )
