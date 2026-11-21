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
#: The Hugging Face commit of DEFAULT_EMBEDDING_MODEL that results/ were measured with; the
#: default model is always loaded at this revision (embedding.Embedder), not at the Hub's main.
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_CHUNK_MAX_WORDS = 240  # chosen with results/SUMMARY.md section 4: half the vectors of 120 words
DEFAULT_CHUNK_OVERLAP_UNITS = 1  # one unit of overlap, kept through the experiments
DEFAULT_VECTOR_DIM = 384  # the model's dimension; sql/schema.sql writes it literally
DEFAULT_EF_SEARCH = 100  # mhnsw_ef_search applied per query by the CLI, web page and harness
#: The range MariaDB 11.8 accepts for mhnsw_ef_search (information_schema.SYSTEM_VARIABLES); the
#: server clamps anything else with only a warning, so a value outside it is refused here.
MIN_EF_SEARCH = 1
MAX_EF_SEARCH = 10000
#: The ef_search value that means "leave the server's session value alone": WIKILENSE_EF_SEARCH=0,
#: --ef-search 0 on the command line, ef_search=0 in the web API.
EF_SEARCH_SERVER = 0
#: The values of WIKILENSE_FACTCHECK, the web page's live fact-check (wikilense.factcheck): off,
#: or on through the Claude Code CLI's own login, or through the Anthropic API with a key. An API
#: key in .env turns nothing on by itself; only this setting does.
FACTCHECK_MODES = ("off", "subscription", "api")
DEFAULT_FACTCHECK = "off"
#: Seconds a search of the web page or of `wikilense query` may run before MariaDB stops it
#: (``max_statement_time``, error 1969); 0 sets no limit. On 11.8.9 a vector search that fewer
#: rows pass than its LIMIT can keep searching until the server runs out of memory
#: (docs/VECTOR_INDEX.md). The evaluation and the measurement scripts do not use it.
DEFAULT_STATEMENT_SECONDS = 5


class SettingsError(ValueError):
    """A required setting is missing or a value has the wrong type; the message names it."""


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Connection, model, chunking and search parameters. The password is kept out of repr().

    The vector index's ``M`` is not a setting: sql/schema.sql writes it (``db.VECTOR_INDEX_M``).
    """

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
    ef_search: int = DEFAULT_EF_SEARCH
    factcheck: str = DEFAULT_FACTCHECK
    claude_bin: str = ""
    statement_seconds: int = DEFAULT_STATEMENT_SECONDS


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

    ``WIKILENSE_FACTCHECK`` (default ``off``) turns the web page's live fact-check on;
    ``WIKILENSE_CLAUDE_BIN`` names the Claude Code CLI it uses with ``subscription`` (empty: the
    ``CLAUDE_BIN`` variable, then ``claude`` on the PATH). Neither reads the API key.
    ``WIKILENSE_STATEMENT_SECONDS`` (default ``DEFAULT_STATEMENT_SECONDS``, 0 for no limit) is the
    time limit of the web page's and ``wikilense query``'s searches.

    Raises SettingsError, naming the variable, when a required one is missing, an integer
    variable is malformed, ``WIKILENSE_EF_SEARCH`` is neither 0 (the server's value) nor in
    ``MIN_EF_SEARCH`` .. ``MAX_EF_SEARCH``, ``WIKILENSE_FACTCHECK`` is not one of
    ``FACTCHECK_MODES``, or ``WIKILENSE_STATEMENT_SECONDS`` is negative.
    """
    path = DEFAULT_ENV_FILE if env_file is None else Path(env_file)
    file_values: dict[str, str] = {}
    if path.is_file():
        file_values = {k: v for k, v in dotenv_values(path).items() if v is not None}
    env: dict[str, str] = {**file_values, **os.environ}

    db_name = _get_str(env, "WIKILENSE_DB_NAME", "wikilense")
    ef_search = _get_int(env, "WIKILENSE_EF_SEARCH", DEFAULT_EF_SEARCH)
    if ef_search != EF_SEARCH_SERVER and not MIN_EF_SEARCH <= ef_search <= MAX_EF_SEARCH:
        raise SettingsError(
            f"WIKILENSE_EF_SEARCH must be {EF_SEARCH_SERVER} (keep the server's value) or from "
            f"{MIN_EF_SEARCH} to {MAX_EF_SEARCH}, got {ef_search}"
        )
    factcheck = _get_str(env, "WIKILENSE_FACTCHECK", DEFAULT_FACTCHECK).strip().lower()
    if factcheck not in FACTCHECK_MODES:
        raise SettingsError(
            f"WIKILENSE_FACTCHECK must be one of {', '.join(FACTCHECK_MODES)}, got {factcheck!r}"
        )
    statement_seconds = _get_int(env, "WIKILENSE_STATEMENT_SECONDS", DEFAULT_STATEMENT_SECONDS)
    if statement_seconds < 0:
        raise SettingsError(
            "WIKILENSE_STATEMENT_SECONDS must be 0 (no limit) or a positive whole number of seconds, "
            f"got {statement_seconds}"
        )
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
        ef_search=ef_search,
        factcheck=factcheck,
        claude_bin=_get_str(env, "WIKILENSE_CLAUDE_BIN", ""),
        statement_seconds=statement_seconds,
    )
