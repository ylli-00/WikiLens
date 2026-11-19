"""Tests for scripts/scaled_stage2.py without a database, a GPU or the model: the database-name
guard, the cached embedder, the saved parts and the root statements."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wikilense.config import REPO_ROOT, Settings

SCRIPT = REPO_ROOT / "scripts" / "scaled_stage2.py"
MODEL = {"model_name": "BAAI/bge-small-en-v1.5", "revision_label": "abc123"}


@pytest.fixture(scope="module")
def s2() -> Iterator[Any]:
    spec = importlib.util.spec_from_file_location("scaled_stage2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def settings(name: str) -> Settings:
    return Settings(db_password="x", db_name=name)


@pytest.mark.parametrize("name", ["wikilense", "wikilense_test", "wikilense_s0_test", "other", ""])
def test_every_database_but_s0_and_s1_is_refused(s2: Any, name: str) -> None:
    with pytest.raises(s2.Stage2Error, match="refusing database"):
        s2.check_database(settings(name))


@pytest.mark.parametrize("name", ["wikilense_s0", "wikilense_s1"])
def test_the_two_new_databases_are_accepted(s2: Any, name: str) -> None:
    assert s2.check_database(settings(name)) == name


def test_the_root_statements_are_exactly_these_four(s2: Any) -> None:
    assert s2.CREATE_STATEMENTS == (
        "CREATE DATABASE wikilense_s0",
        "CREATE DATABASE wikilense_s1",
        "GRANT ALL PRIVILEGES ON wikilense_s0.* TO 'wikilense'@'%'",
        "GRANT ALL PRIVILEGES ON wikilense_s1.* TO 'wikilense'@'%'",
    )


def test_root_sql_refuses_other_databases(s2: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(s2.ROOT_PASSWORD_VARIABLE, "dummy")
    with pytest.raises(s2.Stage2Error, match="not allowed"):
        s2.root_sql("SELECT 1", database="wikilense")


def test_cached_embedder_returns_vectors_in_the_order_asked(s2: Any) -> None:
    cache = {"a": np.array([1, 0], dtype=np.float32), "b": np.array([0, 1], dtype=np.float32)}
    emb = s2.CachedEmbedder(cache, **MODEL)
    assert emb.dim == 2 and emb.model_name == MODEL["model_name"]
    out = emb.embed_passages(["b", "a", "b"])
    assert out.dtype == np.float32 and out.tolist() == [[0, 1], [1, 0], [0, 1]]
    assert emb.embed_passages([]).shape == (0, 2)
    with pytest.raises(s2.Stage2Error, match="no saved vector"):
        emb.embed_passages(["a", "c"])


def test_parts_round_trip_and_resume(s2: Any, tmp_path: Path) -> None:
    texts = ["one", "two", "one more"]
    vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert not s2.part_is_done(tmp_path, 0, texts, 2)
    s2.save_part(tmp_path, 0, texts, vectors, MODEL)
    assert s2.part_is_done(tmp_path, 0, texts, 2)
    assert not s2.part_is_done(tmp_path, 0, ["one", "two", "changed"], 2)
    assert not s2.part_is_done(tmp_path, 0, texts, 3)
    s2.save_part(tmp_path, 1, ["four"], np.ones((1, 2), dtype=np.float32), MODEL)
    cache, model = s2.load_vectors(tmp_path)
    assert model == MODEL and sorted(cache) == ["four", "one", "one more", "two"]
    assert cache["two"].tolist() == [2.0, 3.0]


def test_parts_with_two_models_are_refused(s2: Any, tmp_path: Path) -> None:
    s2.save_part(tmp_path, 0, ["a"], np.ones((1, 2), dtype=np.float32), MODEL)
    s2.save_part(tmp_path, 1, ["b"], np.ones((1, 2), dtype=np.float32),
                 {**MODEL, "revision_label": "other"})
    with pytest.raises(s2.Stage2Error, match="made with"):
        s2.load_vectors(tmp_path)


def test_distinct_keeps_first_order_and_digest_matches_scaled_build(s2: Any) -> None:
    assert s2.distinct(["b", "a", "b", "c"]) == ["b", "a", "c"]
    import hashlib

    want = hashlib.sha256(b"x\ny\n").hexdigest()
    assert s2.texts_digest(["x", "y"]) == want
