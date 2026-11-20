"""Tests of scripts/scaled_release.py and the Makefile's fetch-s1 and load-s1 targets.

A tiny archive built in tmp_path stands in for the release file and is fetched through a file URL,
so no test uses the network. The load order runs `make load-s1` with a stub Python that only
records its arguments, so no test touches a database.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from wikilense.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "scaled_release.py"
FILES = {"NOTICE.md": b"notice\n", "corpora.json": b"{}\n", "page_meta_S1.jsonl": b"",
         "S1/pages.jsonl": b'{"title": "A"}\n', "S1/claims.jsonl": b'{"id": 1}\n',
         "vectors/part_000.json": b"{}", "vectors/part_000.npy": b"\x93NUMPY"}


@pytest.fixture(scope="module")
def rel() -> Iterator[Any]:
    spec = importlib.util.spec_from_file_location("scaled_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def make_archive(path: Path, files: dict[str, bytes], top: str = "wikilense_s1",
                 extra: list[tarfile.TarInfo] | None = None) -> tuple[str, int]:
    """Write a .tar.gz of ``files`` under ``top/``; return its SHA-256 and size."""
    with tarfile.open(path, "w:gz") as tar:
        top_dir = tarfile.TarInfo(top)
        top_dir.type = tarfile.DIRTYPE
        tar.addfile(top_dir)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info in extra or []:
            tar.addfile(info, io.BytesIO(b"x" * info.size) if info.isreg() else None)
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def test_matching_archive_is_unpacked_into_dest_and_then_deleted(rel: Any, tmp_path: Path) -> None:
    sha, size = make_archive(tmp_path / "release.tar.gz", FILES)
    dest = tmp_path / "data" / "scaled" / "wikilense_s1"
    assert rel.run_fetch((tmp_path / "release.tar.gz").as_uri(), sha, size, dest) == 0
    for name, data in FILES.items():
        assert (dest / name).read_bytes() == data
    assert not (dest / "wikilense_s1").exists()  # the top folder is the destination itself
    assert (dest / ".sha256").read_text(encoding="utf-8").strip() == sha
    # Nothing else is left next to the destination: no download, no partial folder.
    assert sorted(p.name for p in dest.parent.iterdir()) == ["wikilense_s1"]


def test_a_second_fetch_with_the_same_checksum_does_nothing(rel: Any, tmp_path: Path) -> None:
    sha, size = make_archive(tmp_path / "release.tar.gz", FILES)
    dest = tmp_path / "wikilense_s1"
    rel.run_fetch((tmp_path / "release.tar.gz").as_uri(), sha, size, dest)

    def no_download(url: str, target: Path) -> None:
        raise AssertionError("downloaded again")

    assert rel.run_fetch("", sha, size, dest, fetch=no_download) == 0


@pytest.mark.parametrize("wrong", ["sha256", "bytes"])
def test_a_mismatch_stops_before_anything_is_unpacked(rel: Any, tmp_path: Path, wrong: str) -> None:
    sha, size = make_archive(tmp_path / "release.tar.gz", FILES)
    if wrong == "sha256":
        sha = "0" * 64
    else:
        size += 1
    dest = tmp_path / "out" / "wikilense_s1"
    with pytest.raises(rel.ReleaseError, match="SHA-256" if wrong == "sha256" else "bytes"):
        rel.run_fetch((tmp_path / "release.tar.gz").as_uri(), sha, size, dest)
    assert not dest.exists()
    assert list((tmp_path / "out").iterdir()) == []  # the .part download was removed


def test_the_checksum_is_compared_without_regard_to_case(rel: Any, tmp_path: Path) -> None:
    sha, size = make_archive(tmp_path / "release.tar.gz", FILES)
    dest = tmp_path / "wikilense_s1"
    assert rel.run_fetch((tmp_path / "release.tar.gz").as_uri(), sha.upper(), size, dest) == 0


def test_an_existing_folder_of_another_release_is_not_overwritten(rel: Any, tmp_path: Path) -> None:
    sha, size = make_archive(tmp_path / "release.tar.gz", FILES)
    dest = tmp_path / "wikilense_s1"
    dest.mkdir()
    (dest / "mine.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(rel.ReleaseError, match="move it away"):
        rel.run_fetch((tmp_path / "release.tar.gz").as_uri(), sha, size, dest)
    assert (dest / "mine.txt").read_text(encoding="utf-8") == "keep"


def test_no_url_is_refused_with_the_reason(rel: Any, tmp_path: Path) -> None:
    with pytest.raises(rel.ReleaseError, match="no download URL"):
        rel.run_fetch("", "0" * 64, 1, tmp_path / "wikilense_s1")


def test_keep_archive_keeps_the_checked_download(rel: Any, tmp_path: Path) -> None:
    sha, size = make_archive(tmp_path / "release.tar.gz", FILES)
    dest = tmp_path / "out" / "wikilense_s1"
    rel.run_fetch((tmp_path / "release.tar.gz").as_uri(), sha, size, dest, keep_archive=True)
    assert (tmp_path / "out" / "wikilense_s1.tar.gz").read_bytes() == (
        tmp_path / "release.tar.gz").read_bytes()


def unsafe_members() -> list[tuple[str, list[tarfile.TarInfo], dict[str, bytes], str]]:
    link = tarfile.TarInfo("wikilense_s1/link")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    up = tarfile.TarInfo("wikilense_s1/../escaped.txt")
    up.size = 1
    absolute = tarfile.TarInfo("/tmp/absolute.txt")
    absolute.size = 1
    return [("symlink", [link], FILES, "wikilense_s1"),
            ("parent path", [up], FILES, "wikilense_s1"),
            ("absolute path", [absolute], FILES, "wikilense_s1"),
            ("other top folder", [], FILES, "something_else")]


@pytest.mark.parametrize("case", unsafe_members(), ids=lambda case: case[0])
def test_members_outside_the_top_folder_or_not_regular_are_refused(rel: Any, tmp_path: Path,
                                                                   case: Any) -> None:
    _, extra, files, top = case
    sha, size = make_archive(tmp_path / "release.tar.gz", files, top=top, extra=extra)
    dest = tmp_path / "out" / "wikilense_s1"
    with pytest.raises(rel.ReleaseError, match="archive member"):
        rel.run_fetch((tmp_path / "release.tar.gz").as_uri(), sha, size, dest)
    assert not dest.exists()
    assert not (tmp_path / "out" / "wikilense_s1.partial").exists()
    assert not (tmp_path / "out" / "escaped.txt").exists()
    assert not (tmp_path / "escaped.txt").exists()


class FakeCursor:
    def __init__(self, tables: int, seen: list[Any]) -> None:
        self.tables, self.seen = tables, seen

    def __enter__(self):  # returns self, like a real cursor
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[str, ...]) -> None:
        self.seen.append((sql, params))

    def fetchone(self) -> tuple[int]:
        return (self.tables,)


class FakeConnection:
    def __init__(self, tables: int, seen: list[Any]) -> None:
        self.tables, self.seen, self.closed = tables, seen, False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.tables, self.seen)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(("tables", "status"), [(0, 0), (9, 1)])
def test_check_empty_stops_only_when_the_database_holds_tables(rel: Any, tables: int,
                                                               status: int) -> None:
    seen: list[Any] = []
    connections: list[FakeConnection] = []

    def connect(database: str) -> FakeConnection:
        assert database == "information_schema"
        connections.append(FakeConnection(tables, seen))
        return connections[-1]

    assert rel.run_check_empty("wikilense_s1", connect=connect) == status
    assert seen == [(rel.TABLES_SQL, ("wikilense_s1",))]
    assert connections[0].closed


# ---- the Makefile -------------------------------------------------------------------------------

needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def run_make(tmp_path: Path, *args: str) -> list[str]:
    """Run make in the repository with a stub $(VENV)/bin/python; return the calls it recorded."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    stub = venv / "bin" / "python"
    stub.write_text('#!/bin/sh\necho "DB=${WIKILENSE_DB_NAME:-} $*" >> "$STUB_LOG"\n',
                    encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {k: v for k, v in os.environ.items()
           if k not in ("WIKILENSE_DB_NAME", "REPLACE", "MAKEFLAGS") and not k.startswith("S1_")}
    env["STUB_LOG"] = str(log)
    subprocess.run(["make", "-s", "-C", str(REPO_ROOT), f"VENV={venv}", *args], env=env,
                   check=True, capture_output=True, text=True)
    return log.read_text(encoding="utf-8").splitlines()


def steps(calls: list[str]) -> list[str]:
    """Return each call as 'script subcommand'."""
    return [" ".join(call.split()[1:3]) for call in calls]


LOAD_ORDER = ["scripts/scaled_stage2.py create-dbs", "scripts/scaled_stage2.py ingest",
              "scripts/scaled_stage2.py verify", "scripts/scaled_stage3.py load",
              "scripts/scaled_stage3.py verify"]


@needs_make
def test_load_s1_checks_then_loads_in_order_into_wikilense_s1(tmp_path: Path) -> None:
    calls = run_make(tmp_path, "load-s1", "S1_DIR=some/dir")
    assert steps(calls) == ["scripts/scaled_release.py check-empty", *LOAD_ORDER]
    assert all("--allow-check-database" not in call for call in calls)
    assert "--database wikilense_s1" in calls[0]
    assert calls[2].startswith("DB=wikilense_s1 ") and calls[3].startswith("DB=wikilense_s1 ")
    assert "--corpus-dir some/dir/S1 --vectors-dir some/dir/vectors" in calls[2]
    assert "--corpora-json some/dir/corpora.json" in calls[3]
    assert "--database wikilense_s1 --meta some/dir/page_meta_S1.jsonl" in calls[4]


@needs_make
def test_replace_skips_only_the_empty_check(tmp_path: Path) -> None:
    assert steps(run_make(tmp_path, "load-s1", "REPLACE=1")) == LOAD_ORDER


@needs_make
def test_the_check_load_names_the_scratch_database_in_every_step(tmp_path: Path) -> None:
    calls = run_make(tmp_path, "load-s1", "S1_CHECK=1")
    assert steps(calls) == ["scripts/scaled_release.py check-empty", *LOAD_ORDER]
    assert "--database wikilense_s1_check" in calls[0]
    assert all(call.endswith("--allow-check-database") for call in calls[1:])
    assert calls[2].startswith("DB=wikilense_s1_check ")
    assert "--database wikilense_s1_check " in calls[4]


@needs_make
def test_fetch_s1_passes_the_committed_size_and_checksum(tmp_path: Path) -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    sha = re.search(r"^S1_SHA256 := ([0-9a-f]{64})$", makefile, re.MULTILINE).group(1)
    size = re.search(r"^S1_BYTES  := (\d+)$", makefile, re.MULTILINE).group(1)
    (call,) = run_make(tmp_path, "fetch-s1", "S1_URL=file:///nowhere/x.tar.gz")
    assert call.split()[1:] == ["scripts/scaled_release.py", "fetch", "--url",
                                "file:///nowhere/x.tar.gz", "--sha256", sha, "--bytes", size,
                                "--dest", "data/scaled/wikilense_s1"]
