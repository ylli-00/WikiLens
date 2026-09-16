#!/usr/bin/env python3
"""Fetch one FEVEROUS Wikipedia shard without downloading the whole archive.

feverous-wiki-pages.zip (9,906,569,155 bytes, https://fever.ai/dataset/feverous.html)
holds the pages as 544 JSONL shards named FeverousWikiv1/wiki_NNN.jsonl. A zip file
keeps its file list at the end, so HTTP range requests can read that list and then
download only the bytes of one shard. The shard is checked against the CRC-32 stored
in the archive.

Each line of a shard is one Wikipedia page: {"title", "order", "sentence_N", "table_N",
"section_N", "list_N"}.

Standard library only. Usage (from the project root):
    python3 scripts/fetch_wiki_shard.py --list                       # print shard names and sizes
    python3 scripts/fetch_wiki_shard.py wiki_000.jsonl               # -> data/feverous/wiki_pages/wiki_000.jsonl
    python3 scripts/fetch_wiki_shard.py wiki_000.jsonl --out-dir other/dir
Shard numbers have gaps (000-610, 544 shards), so take names from --list. An existing
output file is left alone unless --force is given.
"""

import argparse
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path

URL = "https://fever.ai/download/feverous/feverous-wiki-pages.zip"
MEMBER_PREFIX = "FeverousWikiv1/"
TIMEOUT = 60  # seconds per request


def fetch_range(url, start, end):
    """Return bytes start..end (inclusive) of url."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        if resp.status != 206:
            raise RuntimeError(f"server did not honour the range request (HTTP {resp.status})")
        return resp.read()


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return int(resp.headers["Content-Length"])


def zip64_values(extra, sizes):
    """Replace 0xFFFFFFFF placeholders (uncompressed, compressed, offset) from the ZIP64 extra field."""
    pos = 0
    while pos + 4 <= len(extra):
        header_id, length = struct.unpack("<HH", extra[pos:pos + 4])
        if header_id == 1:
            data, k, out = extra[pos + 4:pos + 4 + length], 0, list(sizes)
            for i, value in enumerate(sizes):
                if value == 0xFFFFFFFF:
                    out[i] = struct.unpack("<Q", data[k:k + 8])[0]
                    k += 8
            return out
        pos += 4 + length
    return list(sizes)


def central_directory(url):
    """Yield (name, method, crc, compressed, uncompressed, local_header_offset) for every entry."""
    size = remote_size(url)
    tail = fetch_range(url, max(0, size - 65536), size - 1)
    locator = tail.rfind(b"PK\x06\x07")
    if locator < 0:
        raise RuntimeError("ZIP64 end-of-central-directory locator not found")
    _, eocd64_offset, _ = struct.unpack("<IQI", tail[locator + 4:locator + 20])
    eocd64 = fetch_range(url, eocd64_offset, eocd64_offset + 55)
    cd_size, cd_offset = struct.unpack("<QQ", eocd64[40:56])
    cd = fetch_range(url, cd_offset, cd_offset + cd_size - 1)
    pos = 0
    while cd[pos:pos + 4] == b"PK\x01\x02":
        (method, crc, compressed, uncompressed, name_len, extra_len, comment_len, offset) = (
            struct.unpack("<H", cd[pos + 10:pos + 12])[0], *struct.unpack("<III", cd[pos + 16:pos + 28]),
            *struct.unpack("<HHH", cd[pos + 28:pos + 34]), struct.unpack("<I", cd[pos + 42:pos + 46])[0])
        name = cd[pos + 46:pos + 46 + name_len].decode("utf-8")
        extra = cd[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        uncompressed, compressed, offset = zip64_values(extra, (uncompressed, compressed, offset))
        yield name, method, crc, compressed, uncompressed, offset
        pos += 46 + name_len + extra_len + comment_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shard", nargs="?", help="shard file name, e.g. wiki_000.jsonl")
    ap.add_argument("--list", action="store_true", help="list shards and sizes, then exit")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "feverous" / "wiki_pages")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()
    if not args.list and not args.shard:
        ap.error("give a shard name or --list")

    try:
        entries = [e for e in central_directory(args.url) if not e[0].endswith("/")]
        if args.list:
            for name, _, _, compressed, uncompressed, _ in sorted(entries):
                print(f"{name}\tcompressed={compressed}\tuncompressed={uncompressed}")
            return

        wanted = args.shard if args.shard.startswith(MEMBER_PREFIX) else MEMBER_PREFIX + args.shard
        match = [e for e in entries if e[0] == wanted]
        if not match:
            sys.exit(f"{wanted} not found in the archive (use --list)")
        name, method, crc, compressed, uncompressed, offset = match[0]
        out = args.out_dir / Path(name).name
        if out.exists() and not args.force:
            sys.exit(f"{out} already exists ({out.stat().st_size:,} bytes); use --force to download it again")

        local = fetch_range(args.url, offset, offset + 29)
        if local[:4] != b"PK\x03\x04":
            sys.exit("local file header not found at the expected offset")
        name_len, extra_len = struct.unpack("<HH", local[26:30])
        start = offset + 30 + name_len + extra_len
        raw = fetch_range(args.url, start, start + compressed - 1)
    except (urllib.error.URLError, RuntimeError, OSError) as e:
        sys.exit(f"download failed: {e}")

    if method == 8:
        data = zlib.decompress(raw, -15)
    elif method == 0:
        data = raw
    else:
        sys.exit(f"unsupported compression method {method}")
    if zlib.crc32(data) != crc or len(data) != uncompressed:
        sys.exit("CRC or size mismatch - download is corrupt, nothing written")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    part.write_bytes(data)
    os.replace(part, out)  # the final name only ever holds a complete, CRC-checked file
    print(f"{name}: downloaded {compressed:,} bytes, wrote {uncompressed:,} bytes to {out} (CRC ok)")


if __name__ == "__main__":
    main()
