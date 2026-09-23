"""Tests for scripts/run_experiments.py that need neither the server nor docker nor the model.

The script is not a package module, so it is loaded from its file. These tests cover what the
README's comparison numbers and the root-level operations rest on: the hit-list comparison, the
configuration check, the fixed SQL of the index rebuild and of the tablespace query, and how the
root password reaches ``docker exec``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wikilense import db as dbmod
from wikilense.config import REPO_ROOT, Settings

SCRIPT = REPO_ROOT / "scripts" / "run_experiments.py"


@pytest.fixture(scope="module")
def rexp() -> Any:
    """The script, imported as a module (``main()`` is not run)."""
    spec = importlib.util.spec_from_file_location("run_experiments", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiments"] = module  # dataclasses look the module up while defining
    spec.loader.exec_module(module)
    return module


def _write_run(directory: Path, name: str, hits: dict[int, list[int]], ranks: dict[int, int | None]) -> None:
    """Write the ``claims`` part of a result JSON, the part compare_hits reads."""
    claims = [
        {"claim_id": cid, "hit_chunk_ids": ids, "gold_page_rank": ranks.get(cid)}
        for cid, ids in hits.items()
    ]
    (directory / f"{name}.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")


def test_compare_hits_counts_identical_lists_and_where_they_part(rexp: Any, tmp_path: Path) -> None:
    _write_run(tmp_path, "ref", {1: [5, 6, 7], 2: [8, 9], 3: [1, 2]}, {1: 1, 2: 2, 3: None})
    _write_run(tmp_path, "same", {1: [5, 6, 7], 2: [8, 9], 3: [1, 2]}, {1: 1, 2: 2, 3: None})
    _write_run(tmp_path, "other", {1: [5, 7, 6], 2: [8, 9], 3: [4, 2]}, {1: 1, 2: None, 3: None})
    result = rexp.compare_hits(tmp_path, "ref", ["same", "other"])
    assert result["n_claims"] == 3
    assert result["runs"]["same"]["identical_sequence"] == 3
    other = result["runs"]["other"]
    assert other["identical_sequence"] == 1  # claim 2 only
    assert other["identical_set"] == 2  # claim 1 has the same chunks in another order
    assert other["differing_claims"] == [1, 3]
    details = {d["claim_id"]: d for d in other["differing_details"]}
    assert details[1]["first_differing_position"] == 2 and details[1]["common_chunks"] == 3
    assert details[3]["first_differing_position"] == 1 and details[3]["common_chunks"] == 1
    assert result["identical_in_all"] == 1 and result["differing_claims_any"] == [1, 3]
    assert rexp.lost_gold_pages(tmp_path, "ref", "other") == [2]  # rank 2 -> not retrieved


def test_configuration_differences_include_the_embedding_model(rexp: Any) -> None:
    config = rexp.FINAL
    meta = {**config.ingest_meta_expected(), "embedding_model": "BAAI/bge-small-en-v1.5"}
    index = {"name": "embedding", "m": config.index_m, "distance": "cosine"}
    assert config.differences(meta, index, "BAAI/bge-small-en-v1.5") == {}
    diffs = config.differences(meta, index, "some/other-384-model")
    assert set(diffs) == {"ingest_meta.embedding_model"}
    assert set(config.differences(meta, {**index, "m": 6}, "BAAI/bge-small-en-v1.5")) == {
        "vector_index_m"
    }
    assert config.index_m == dbmod.VECTOR_INDEX_M
    # a configuration changes only the chunking settings; the index M is the schema's
    settings = rexp.OLD.settings(Settings(db_password="x"))
    assert (settings.chunk_max_words, settings.chunk_overlap_units) == (120, 1)


def test_rebuild_and_tablespace_sql_are_fixed_literals(rexp: Any) -> None:
    assert set(rexp.INDEX_M_VALUES) == {6, 16, 32}
    for m, statement in rexp.ADD_INDEX_SQL.items():
        assert statement == (
            f"ALTER TABLE chunk ADD VECTOR INDEX `embedding` (embedding) M={m} DISTANCE=cosine"
        )
    assert rexp.DROP_INDEX_SQL == "ALTER TABLE chunk DROP INDEX `embedding`"
    assert "{" not in rexp.HIDDEN_TABLESPACE_SQL and "%" in rexp.HIDDEN_TABLESPACE_SQL
    assert "DATABASE()" in rexp.HIDDEN_TABLESPACE_SQL
    for statement in rexp.SET_CACHE_SQL.values():
        assert statement.startswith("SET GLOBAL mhnsw_max_cache_size = ")


def test_root_sql_passes_the_password_through_the_environment_only(
    rexp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, "env": kwargs.get("env") or {}})
        return subprocess.CompletedProcess(argv, 0, stdout="wikilense/chunk#i#04\t13631488\n", stderr="")

    monkeypatch.setattr(rexp.subprocess, "run", fake_run)
    monkeypatch.setenv(rexp.ROOT_PASSWORD_VARIABLE, "sentinel-root-password")
    server = rexp.Server(Settings(db_password="x", db_name="wikilense"))
    assert server.vector_index_tablespace_bytes() == 13631488
    argv = calls[0]["argv"]
    assert "sentinel-root-password" not in " ".join(argv)
    assert argv[:5] == ["docker", "exec", "-e", "MYSQL_PWD", rexp.CONTAINER]
    assert calls[0]["env"]["MYSQL_PWD"] == "sentinel-root-password"
    assert argv[-3:] == ["wikilense", "-e", rexp.HIDDEN_TABLESPACE_SQL]  # the default database
    with pytest.raises(rexp.ExperimentError, match="plain identifier"):
        server.root_sql("SELECT 1", database="wikilense; DROP DATABASE x")


def test_device_phrase_and_repeats_option(rexp: Any) -> None:
    assert rexp.device_phrase({"machine": {"embedding_device": "cuda"}}) == "on the GPU"
    assert rexp.device_phrase({"machine": {"embedding_device": "cpu"}}) == "on the CPU"
    assert rexp.device_phrase(None) == "on ?"
    with pytest.raises(SystemExit):  # argparse refuses --repeats 0 before anything runs
        rexp.main(["summary", "--repeats", "0"])
