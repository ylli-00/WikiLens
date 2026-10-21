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


MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def test_configuration_differences_include_the_embedding_model(rexp: Any) -> None:
    config = rexp.FINAL
    meta = {**config.ingest_meta_expected(), "embedding_model": MODEL, "embedding_revision": REVISION}
    index = {"name": "embedding", "m": config.index_m, "distance": "cosine"}
    assert config.differences(meta, index, MODEL, REVISION) == {}
    diffs = config.differences(meta, index, "some/other-384-model", REVISION)
    assert set(diffs) == {"ingest_meta.embedding_model"}
    unrecorded = {k: v for k, v in meta.items() if k != "embedding_revision"}  # a pre-pin ingest
    assert set(config.differences(unrecorded, index, MODEL, REVISION)) == {
        "ingest_meta.embedding_revision"
    }
    assert set(config.differences(meta, {**index, "m": 6}, "BAAI/bge-small-en-v1.5")) == {
        "vector_index_m"
    }
    assert config.index_m == dbmod.VECTOR_INDEX_M
    # a configuration changes only the chunking settings; the index M is the schema's
    settings = rexp.OLD.settings(Settings(db_password="x"))
    assert (settings.chunk_max_words, settings.chunk_overlap_units) == (120, 1)


class _Embedder:
    model_name = MODEL
    revision_label = REVISION


def _context(rexp: Any, tmp_path: Path, meta: dict[str, str], index: dict[str, Any]) -> Any:
    ctx = rexp.Context(settings=Settings(db_password="x"), server=None, out_dir=tmp_path, repeats=1,
                       _embedder=_Embedder())
    ctx.database_state = lambda: (meta, index)  # instance attribute: no database
    return ctx


def test_check_configuration_refuses_an_ingest_of_another_model_or_revision(
    rexp: Any, tmp_path: Path
) -> None:
    good = {**rexp.FINAL.ingest_meta_expected(), "embedding_model": MODEL,
            "embedding_revision": REVISION}
    index = {"name": "embedding", "m": rexp.FINAL.index_m, "distance": "cosine"}
    assert rexp.check_configuration(_context(rexp, tmp_path, good, index), rexp.FINAL, "t") == good
    for key, value in (("embedding_model", "other/model"), ("embedding_revision", "unpinned")):
        ctx = _context(rexp, tmp_path, {**good, key: value}, index)
        with pytest.raises(rexp.ExperimentError, match=f"ingest_meta.{key}"):
            rexp.check_configuration(ctx, rexp.FINAL, "t")


def test_rebuild_index_refuses_an_index_that_is_not_the_schemas(rexp: Any, tmp_path: Path) -> None:
    executed: list[str] = []

    class Conn:
        def cursor(self) -> Any:
            import contextlib

            return contextlib.nullcontext(self)

        def execute(self, sql: str, params: Any = None) -> None:
            executed.append(sql)

        def fetchone(self) -> tuple[str, str]:
            ddl = "CREATE TABLE `chunk` (\n  VECTOR KEY `other` (`embedding`) `M`='16' `DISTANCE`='cosine'\n)"
            return ("chunk", ddl)

        def close(self) -> None:
            pass

    ctx = _context(rexp, tmp_path, {}, {})
    ctx.connect = Conn
    with pytest.raises(rexp.ExperimentError, match="not 'embedding'"):
        rexp.rebuild_index(ctx, 6, label="t")
    assert executed == ["SHOW CREATE TABLE chunk"]  # nothing was dropped


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


def test_an_interrupted_group_names_the_way_back_to_the_defaults(
    rexp: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def interrupted(ctx: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(rexp, "load_settings", lambda: Settings(db_password="x"))
    monkeypatch.setitem(rexp.GROUPS, "filters", interrupted)
    monkeypatch.setitem(rexp.GROUPS, "summary", interrupted)
    with caplog.at_level("ERROR", logger="experiments"):
        assert rexp.main(["filters", "--out", str(tmp_path)]) == 130
    assert "interrupted" in caplog.text and "restore" in caplog.text
    caplog.clear()
    with caplog.at_level("ERROR", logger="experiments"):
        assert rexp.main(["summary", "--out", str(tmp_path)]) == 130
    assert "restore" not in caplog.text  # the summary group never touches the database


def test_device_phrase_and_repeats_option(rexp: Any, tmp_path: Path) -> None:
    assert rexp.device_phrase({"machine": {"embedding_device": "cuda"}}) == "on the GPU"
    assert rexp.device_phrase({"machine": {"embedding_device": "cpu"}}) == "on the CPU"
    assert rexp.device_phrase(None) == "on ?"
    # argparse refuses --repeats 0 before anything runs; --out points at a scratch directory so
    # that a regression could never rewrite the committed results/SUMMARY.md
    with pytest.raises(SystemExit):
        rexp.main(["summary", "--repeats", "0", "--out", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_uncommitted_probe_sql_mirrors_the_schema_and_is_fixed(rexp: Any) -> None:
    """The scratch table copies chunk.embedding's type and index from sql/schema.sql, and every
    statement of the group is a fixed literal (values are parameters)."""
    schema = (rexp.REPO_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    assert f"VECTOR({rexp.db.VECTOR_DIM})" in rexp.UNCOMMITTED_CREATE_SQL
    assert f"embedding  VECTOR({rexp.db.VECTOR_DIM}) NOT NULL" in schema
    index = f"VECTOR INDEX (embedding) M={rexp.db.VECTOR_INDEX_M} DISTANCE=cosine"
    assert index in rexp.UNCOMMITTED_CREATE_SQL and index in schema
    for statement in (rexp.UNCOMMITTED_CREATE_SQL, rexp.UNCOMMITTED_DROP_SQL,
                      rexp.UNCOMMITTED_INSERT_SQL, rexp.UNCOMMITTED_KNN_SQL):
        assert "{" not in statement and "uncommitted_probe" in statement
    assert rexp.UNCOMMITTED_KNN_SQL.count("%s") == 2
    assert "uncommitted" in rexp.GROUPS and rexp.PROTOCOL_ORDER[-1] == "summary"


def test_uncommitted_rows_and_markdown_average_the_builds(rexp: Any) -> None:
    def cell(overlap: float, worst: float, p50: float) -> dict[str, float]:
        return {"mean_overlap": overlap, "min_overlap": worst, "p50_ms": p50, "p95_ms": p50}

    builds = [
        {"rows": 3, "queries": 2, "rows_left_after_rollback": 0,
         "insert_seconds": {"uncommitted": 0.5, "committed": 0.4},
         "uncommitted": {"20": cell(0.6, 0.3, 0.2)}, "committed": {"20": cell(1.0, 1.0, 0.5)}},
        {"rows": 3, "queries": 2, "rows_left_after_rollback": 0,
         "insert_seconds": {"uncommitted": 0.6, "committed": 0.4},
         "uncommitted": {"20": cell(0.8, 0.5, 0.4)}, "committed": {"20": cell(0.9, 0.8, 0.7)}},
    ]  # fmt: skip
    data = {"name": rexp.UNCOMMITTED, "created_at": "now", "mariadb_version": "11.8.9", "k": 10,
            "builds": 2, "efs": [20], "datasets": {"chunks": {"label": "x", "builds": builds}}}
    assert rexp.uncommitted_rows(data) == [
        ["chunks", "uncommitted", 20, "0.700 (0.600..0.800)", "0.3", "0.30"],
        ["chunks", "committed", 20, "0.950 (0.900..1.000)", "0.8", "0.60"],
    ]
    text = rexp.uncommitted_markdown(data)
    assert "| chunks | uncommitted | 20 | 0.700 (0.600..0.800) | 0.3 | 0.30 |" in text
    assert "rows left after ROLLBACK [0]" in text and "0.5 / 0.4, 0.6 / 0.4" in text
