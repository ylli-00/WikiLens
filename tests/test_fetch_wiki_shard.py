"""Tests for scripts/fetch_wiki_shard.py without the network: a failed or corrupt download is one
message and writes nothing."""

from __future__ import annotations

import http.client
import importlib.util
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest

from wikilense.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "fetch_wiki_shard.py"
NAME = "FeverousWikiv1/wiki_000.jsonl"


@pytest.fixture
def fetch(monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location("fetch_wiki_shard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(fetch: Any, monkeypatch: pytest.MonkeyPatch, out_dir: Path) -> str:
    """Run main() for wiki_000.jsonl into ``out_dir`` and return its exit message."""
    monkeypatch.setattr(sys, "argv", ["fetch_wiki_shard.py", "wiki_000.jsonl", "--out-dir", str(out_dir)])
    with pytest.raises(SystemExit) as raised:
        fetch.main()
    return str(raised.value.code)


def _archive(fetch: Any, monkeypatch: pytest.MonkeyPatch, body: bytes, crc: int, size: int) -> None:
    """Serve one deflated member ``body`` from a fake archive (the network functions replaced)."""
    local = b"PK\x03\x04" + b"\x00" * 22 + struct.pack("<HH", 0, 0)
    entry = (NAME, 8, crc, len(body), size, 0)
    monkeypatch.setattr(fetch, "central_directory", lambda url: iter([entry]))

    def fetch_range(url: str, start: int, end: int) -> bytes:
        return local if start == 0 else body

    monkeypatch.setattr(fetch, "fetch_range", fetch_range)


def test_a_connection_dropped_mid_body_is_one_message(
    fetch: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def dropped(url: str, start: int, end: int) -> bytes:
        raise http.client.IncompleteRead(b"partial", 990)

    monkeypatch.setattr(fetch, "central_directory", lambda url: iter([(NAME, 8, 0, 10, 10, 0)]))
    monkeypatch.setattr(fetch, "fetch_range", dropped)
    message = _run(fetch, monkeypatch, tmp_path)
    assert message.startswith("download failed:") and "IncompleteRead" in message
    assert list(tmp_path.iterdir()) == []


def test_a_truncated_deflate_stream_is_one_message(
    fetch: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = b'{"title": "Aare"}\n' * 50
    compressor = zlib.compressobj(wbits=-15)
    deflated = compressor.compress(data) + compressor.flush()
    _archive(fetch, monkeypatch, deflated[: len(deflated) // 2], zlib.crc32(data), len(data))
    message = _run(fetch, monkeypatch, tmp_path)
    assert message.startswith("download is corrupt") and "nothing written" in message
    assert list(tmp_path.iterdir()) == []


def test_a_complete_member_is_written_after_the_crc_check(
    fetch: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = b'{"title": "Aare"}\n' * 50
    compressor = zlib.compressobj(wbits=-15)
    deflated = compressor.compress(data) + compressor.flush()
    _archive(fetch, monkeypatch, deflated, zlib.crc32(data), len(data))
    monkeypatch.setattr(sys, "argv", ["fetch_wiki_shard.py", "wiki_000.jsonl", "--out-dir", str(tmp_path)])
    fetch.main()
    assert (tmp_path / "wiki_000.jsonl").read_bytes() == data
    assert "(CRC ok)" in capsys.readouterr().out
    _archive(fetch, monkeypatch, deflated, zlib.crc32(data) ^ 1, len(data))  # a wrong CRC
    assert "CRC or size mismatch" in _run(fetch, monkeypatch, tmp_path / "other")
