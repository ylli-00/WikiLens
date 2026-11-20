"""The S1 release file: download and unpack it (``make fetch-s1``), and check that the target
database is empty before the S1 load (``make load-s1``). Subcommands:

    fetch        download the archive, check its size and SHA-256 against the values given, and
                 unpack its one top folder (``wikilense_s1/``) into --dest. The download goes to
                 ``<dest>.tar.gz.part`` and is renamed to ``<dest>.tar.gz`` only when both checks
                 pass; the files go into ``<dest>.partial/``, which is renamed to --dest at the
                 end, so a stopped run never leaves a half folder under the final name.
                 ``<dest>/.sha256`` records the archive the folder came from: when it matches,
                 fetch does nothing. Only folders and regular files under the top folder are
                 unpacked, and nothing outside --dest is written. The archive is deleted after a
                 successful unpack unless --keep-archive is given.
    check-empty  exit 1 when the database already holds tables (the load drops and replaces
                 them), 0 when it holds none or does not exist.

Usage (from the repository root; ``make -n fetch-s1`` shows the values the Makefile passes):
    .venv/bin/python scripts/scaled_release.py fetch --url <url> --sha256 <hex> --bytes <n> \
        --dest data/scaled/wikilense_s1
    .venv/bin/python scripts/scaled_release.py check-empty --database wikilense_s1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

TOP_FOLDER = "wikilense_s1"
MARKER = ".sha256"
BLOCK = 1 << 20
TIMEOUT_S = 60
TABLES_SQL = "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s"


class ReleaseError(RuntimeError):
    """The download, the archive or the target folder is not as expected; the message says how."""


def log(message: str) -> None:
    """Print one progress line, flushed."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Stream ``url`` (https, or file for a local copy) into ``target``, flushed to disk."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out, BLOCK)
        out.flush()
        os.fsync(out.fileno())


def check_archive(path: Path, sha256: str, size: int) -> None:
    """Raise unless the file has exactly ``size`` bytes and the SHA-256 ``sha256``."""
    got_size = path.stat().st_size
    if got_size != size:
        raise ReleaseError(f"{path.name} has {got_size:,} bytes, expected {size:,}")
    got = file_sha256(path)
    if got != sha256.lower():
        raise ReleaseError(f"{path.name} has SHA-256 {got}, expected {sha256.lower()}")


def member_path(name: str) -> PurePosixPath | None:
    """Return a member's path below TOP_FOLDER, or None for TOP_FOLDER itself.

    Raises for an absolute path, a ``..`` part or a member outside TOP_FOLDER.
    """
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or ".." in parts or not parts or parts[0] != TOP_FOLDER:
        raise ReleaseError(f"archive member {name!r} is not inside {TOP_FOLDER}/")
    return PurePosixPath(*parts[1:]) if len(parts) > 1 else None


def unpack(archive: Path, target: Path) -> tuple[int, int]:
    """Unpack the archive's TOP_FOLDER into the existing folder ``target``; return (files, bytes).

    Only folders and regular files are written; a link or device member stops the unpack.
    """
    files = size = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            rel = member_path(member.name)
            if member.isdir():
                if rel is not None:
                    target.joinpath(*rel.parts).mkdir(parents=True, exist_ok=True)
                continue
            if rel is None or not member.isreg():
                raise ReleaseError(f"archive member {member.name!r} is not a regular file")
            out_path = target.joinpath(*rel.parts)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            assert source is not None
            with open(out_path, "wb") as out:
                shutil.copyfileobj(source, out, BLOCK)
            files += 1
            size += member.size
    return files, size


def run_fetch(url: str, sha256: str, size: int, dest: Path, keep_archive: bool = False,
              fetch: Callable[[str, Path], None] = download) -> int:
    """Download, check and unpack the release into ``dest``; return 0, or raise ReleaseError."""
    sha256 = sha256.lower()
    marker = dest / MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha256:
        log(f"{dest} already holds this release ({sha256[:12]}...); nothing to do")
        return 0
    if dest.exists():
        raise ReleaseError(f"{dest} exists but holds another or an unfinished release: "
                           "move it away, then run this again")
    archive = dest.with_name(dest.name + ".tar.gz")
    if archive.is_file():
        try:
            check_archive(archive, sha256, size)
            log(f"{archive} is already downloaded and checked")
        except ReleaseError as exc:
            log(f"{exc}; downloading it again")
            archive.unlink()
    if not archive.is_file():
        if not url:
            raise ReleaseError("no download URL (the Makefile's S1_URL is empty)")
        part = archive.with_name(archive.name + ".part")
        archive.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading {url} ({size:,} bytes) to {part}")
        try:
            fetch(url, part)
            check_archive(part, sha256, size)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, archive)
        log(f"size and SHA-256 match: {sha256}")
    staging = dest.with_name(dest.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        files, unpacked = unpack(archive, staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    (staging / MARKER).write_text(sha256 + "\n", encoding="utf-8")
    os.replace(staging, dest)
    log(f"unpacked {files} files ({unpacked:,} bytes) into {dest}")
    if not keep_archive:
        archive.unlink()
        log(f"deleted {archive} (pass --keep-archive to keep it)")
    return 0


def run_check_empty(database: str, connect: Callable[..., Any] | None = None) -> int:
    """Return 1 when ``database`` holds tables, else 0; the message says what to do."""
    if connect is None:
        from wikilense import db

        connect = db.connect
    conn = connect(database="information_schema")
    try:
        with conn.cursor() as cur:
            cur.execute(TABLES_SQL, (database,))
            tables = int(cur.fetchone()[0])
    finally:
        conn.close()
    if tables:
        log(f"{database} already holds {tables} tables, and the S1 load would drop and replace "
            "them. To replace them, run the load again with REPLACE=1.")
        return 1
    log(f"{database} holds no tables")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="step", required=True)
    f = sub.add_parser("fetch", help="download, check and unpack the release file")
    f.add_argument("--url", required=True, help="the release file's URL (https or file)")
    f.add_argument("--sha256", required=True, help="its SHA-256, hex")
    f.add_argument("--bytes", required=True, type=int, help="its size in bytes")
    f.add_argument("--dest", required=True, type=Path, help="the folder to unpack into")
    f.add_argument("--keep-archive", action="store_true", help="keep the .tar.gz after unpacking")
    c = sub.add_parser("check-empty", help="exit 1 when the database already holds tables")
    c.add_argument("--database", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one step; return its exit status (ReleaseError -> 2)."""
    args = parse_args(argv)
    try:
        if args.step == "fetch":
            return run_fetch(args.url, args.sha256, args.bytes, args.dest, args.keep_archive)
        return run_check_empty(args.database)
    except ReleaseError as exc:
        log(f"STOP: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
