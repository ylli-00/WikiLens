#!/usr/bin/env python3
"""Select the phase-1 corpus from FEVEROUS shards and claim files -> data/corpus/.

Rule (docs/DESIGN.md, "Corpus selection"; implemented in wikilense.corpus.select_corpus):
1. A claim is in-shard when every page cited in every `content` id of every evidence set is a
   shard title (NFC).
2. Evidence pages = the union of those pages.
3. Filler pages = link targets of the evidence pages that are shard titles and not evidence pages,
   ranked by the number of distinct evidence pages linking to them (descending), then title
   (ascending); up to --n-fill. If fewer exist, continue in shard order with pages of at least
   --min-sentences sentences.
4. Write pages.jsonl (raw shard lines, shard order), claims.jsonl (raw records plus "split") and
   MANIFEST.md (rule, counts, sizes, SHA-256 of both files, the page titles).

Two passes over the shards: one to collect titles, sentence counts and links, one to copy the
selected raw lines unchanged. Every count in MANIFEST.md is recomputed from the written files.
Standard library only. Usage (from the project root):
    python3 scripts/build_corpus.py
    python3 scripts/build_corpus.py --shard data/feverous/wiki_pages/wiki_000.jsonl --n-fill 46
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wikilense.corpus import (
    CorpusSelection,
    _link_targets,
    claim_pages,
    iter_claims,
    iter_pages,
    nfc,
    parse_element_id,
    select_corpus,
)

FEVEROUS_DIR = REPO_ROOT / "data" / "feverous"
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def copy_pages(shard_paths: Iterable[Path], titles: set[str], out_path: Path) -> dict[str, int]:
    """Copy the raw lines of the pages in ``titles`` to ``out_path``, in shard order.

    Lines are copied as bytes, unchanged. Returns ``{shard name: pages in that shard}``.
    """
    shard_sizes: dict[str, int] = {}
    with open(out_path, "wb") as out:
        for shard in shard_paths:
            count = 0
            with open(shard, "rb") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    count += 1
                    if nfc(json.loads(line)["title"]) in titles:
                        out.write(line if line.endswith(b"\n") else line + b"\n")
            shard_sizes[shard.name] = count
    return shard_sizes


def write_claims(claims: list[dict], out_path: Path) -> None:
    """Write the claim records (each with ``split``) as JSON lines to ``out_path``."""
    with open(out_path, "w", encoding="utf-8") as out:
        out.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in claims)


def count_records(path: Path) -> int:
    """Return the number of non-header records in a FEVEROUS claim file."""
    return sum(1 for _ in iter_claims(path, "count"))


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def has_sentence_only_set(record: dict) -> bool:
    """Return True when at least one evidence set of the claim consists of sentences only."""
    return any(
        evidence_set["content"]
        and all(parse_element_id(el)[1] == "sentence" for el in evidence_set["content"])
        for evidence_set in record["evidence"]
    )


def stats_from_files(out_dir: Path) -> dict:
    """Recompute every manifest count from ``pages.jsonl`` and ``claims.jsonl`` in ``out_dir``.

    Returns a dict with the page titles (shard order), evidence titles with the number of citing
    claims, filler titles with the number of distinct evidence pages linking to them, the claim
    counts per split and label, file sizes and SHA-256 digests.
    """
    pages_path, claims_path = out_dir / "pages.jsonl", out_dir / "claims.jsonl"
    pages = list(iter_pages(pages_path))
    claims = list(iter_claims(claims_path))
    page_titles = [nfc(page["title"]) for page in pages]
    title_set = set(page_titles)

    cited: Counter[str] = Counter()
    n_element_ids = 0
    for record in claims:
        cited.update(claim_pages(record))
        n_element_ids += sum(len(evidence_set["content"]) for evidence_set in record["evidence"])
    missing = sorted(set(cited) - title_set)
    if missing:
        raise SystemExit(f"claims.jsonl cites pages missing from pages.jsonl: {missing[:5]}")

    linking: dict[str, set[str]] = {}
    for page in pages:
        title = nfc(page["title"])
        if title in cited:
            for target in _link_targets(page):
                if target in title_set and target not in cited:
                    linking.setdefault(target, set()).add(title)
    fillers = [title for title in page_titles if title not in cited]
    return {
        "page_titles": page_titles,
        "evidence": {title: cited[title] for title in sorted(cited)},
        "filler_link_counts": {title: len(linking.get(title, ())) for title in fillers},
        "n_pages": len(pages),
        "n_claims": len(claims),
        "per_split": Counter(record["split"] for record in claims),
        "per_label": Counter(record["label"] for record in claims),
        "n_sentence_only": sum(1 for record in claims if has_sentence_only_set(record)),
        "n_element_ids": n_element_ids,
        "sizes": {path.name: path.stat().st_size for path in (pages_path, claims_path)},
        "sha256": {path.name: sha256_of(path) for path in (pages_path, claims_path)},
    }


def check_against_selection(stats: dict, selection: CorpusSelection) -> None:
    """Exit with a message when the written files disagree with the in-memory selection."""
    problems = []
    if list(stats["evidence"]) != selection.evidence_titles:
        problems.append("evidence titles")
    if set(stats["filler_link_counts"]) != set(selection.filler_titles):
        problems.append("filler titles")
    n_linked = sum(1 for count in stats["filler_link_counts"].values() if count > 0)
    if n_linked != selection.filler_from_links:
        problems.append("filler pages from links")
    if stats["n_claims"] != len(selection.claims):
        problems.append("claim count")
    if problems:
        raise SystemExit("written files disagree with the selection: " + ", ".join(problems))


def render_manifest(
    stats: dict,
    selection: CorpusSelection,
    shard_sizes: dict[str, int],
    claim_totals: dict[str, int],
    n_fill: int,
    min_sentences: int,
) -> str:
    """Return the text of MANIFEST.md."""
    per_split, per_label = stats["per_split"], stats["per_label"]
    n_evidence = len(stats["evidence"])
    n_filler = stats["n_pages"] - n_evidence
    counts = [
        ("Pages (lines in pages.jsonl)", stats["n_pages"]),
        ("Evidence pages", n_evidence),
        ("Filler pages", n_filler),
        ("Filler pages from links", selection.filler_from_links),
        ("Filler pages from shard order", selection.filler_from_order),
        ("Claims (lines in claims.jsonl)", stats["n_claims"]),
        ("Claims, train", per_split.get("train", 0)),
        ("Claims, dev", per_split.get("dev", 0)),
        *[(f"Claims, {label}", per_label.get(label, 0)) for label in LABELS],
        ("Claims with a sentence-only evidence set", stats["n_sentence_only"]),
        ("Evidence element ids", stats["n_element_ids"]),
    ]
    inputs = ", ".join(f"`{name}` ({n:,} pages)" for name, n in shard_sizes.items())
    inputs += ", " + ", ".join(f"`{name}` ({n:,} claims)" for name, n in claim_totals.items())

    lines = [
        "# Corpus manifest",
        "",
        (
            "Built by `scripts/build_corpus.py` from the FEVEROUS Wikipedia shard(s) and claim "
            "files (https://fever.ai/dataset/feverous.html). Every count below is recomputed from "
            "the two files next to this manifest; re-running the script reproduces them byte for "
            "byte."
        ),
        "",
        "## Rule",
        "",
        (
            "1. A claim is *in-shard* when every page cited in every `content` id of every "
            "evidence set is a shard title (Unicode NFC). Claims without any content id are not "
            "in-shard."
        ),
        "2. Evidence pages are the union of those pages.",
        (
            "3. Filler pages are link targets of the evidence pages that are shard titles and not "
            "evidence pages, ranked by the number of distinct evidence pages linking to them "
            f"(descending), then title (ascending); up to {n_fill}. If fewer exist, the rest is "
            f"taken in shard order from pages with at least {min_sentences} sentences."
        ),
        (
            "4. `pages.jsonl` holds the raw shard lines in shard order; `claims.jsonl` holds the "
            'raw FEVEROUS records plus `"split"` (train records first, then dev, in file order).'
        ),
        "",
        f"Parameters: `--n-fill {n_fill} --min-sentences {min_sentences}`. Inputs: {inputs}.",
        "",
        "## Counts",
        "",
        "| Count | Value |",
        "|---|---|",
        *[f"| {name} | {value} |" for name, value in counts],
        "",
        "## Files",
        "",
        "| File | Bytes | SHA-256 |",
        "|---|---|---|",
        *[
            f"| {name} | {stats['sizes'][name]} | {stats['sha256'][name]} |"
            for name in ("pages.jsonl", "claims.jsonl")
        ],
        "",
        f"## Evidence pages ({n_evidence})",
        "",
        "Sorted by title; in brackets the number of claims citing the page.",
        "",
        *[f"- {title} ({n})" for title, n in stats["evidence"].items()],
        "",
        f"## Filler pages ({n_filler})",
        "",
        (
            "Ranking reason: pages linked from the evidence pages are realistic distractors and "
            'make the "linked from" SQL filter meaningful. In rank order; in brackets the number '
            "of distinct evidence pages linking to the page (ties broken by title). Pages taken "
            'in shard order because too few linked pages existed are marked "shard order".'
        ),
        "",
    ]
    link_counts = stats["filler_link_counts"]
    for rank, title in enumerate(selection.filler_titles, 1):
        reason = f"linked from {link_counts[title]}" if link_counts[title] else "shard order"
        lines.append(f"{rank}. {title} ({reason})")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the selection, write the three files and print the counts; return the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shard",
        action="append",
        type=Path,
        help="shard JSONL file; repeatable (default: data/feverous/wiki_pages/wiki_000.jsonl)",
    )
    parser.add_argument(
        "--train", type=Path, default=FEVEROUS_DIR / "feverous_train_challenges.jsonl"
    )
    parser.add_argument("--dev", type=Path, default=FEVEROUS_DIR / "feverous_dev_challenges.jsonl")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "corpus")
    parser.add_argument("--n-fill", type=int, default=46, help="filler pages to add (default 46)")
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=20,
        help="minimum sentences for a shard-order filler page (default 20)",
    )
    args = parser.parse_args(argv)
    shards: list[Path] = args.shard or [FEVEROUS_DIR / "wiki_pages" / "wiki_000.jsonl"]
    for path in (*shards, args.train, args.dev):
        if not path.is_file():
            parser.error(f"{path} not found")
    if args.n_fill < 0 or args.min_sentences < 0:
        parser.error("--n-fill and --min-sentences must be non-negative")

    selection = select_corpus(shards, args.train, args.dev, args.n_fill, args.min_sentences)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_sizes = copy_pages(shards, selection.titles, args.out_dir / "pages.jsonl")
    write_claims(selection.claims, args.out_dir / "claims.jsonl")
    claim_totals = {args.train.name: count_records(args.train), args.dev.name: count_records(args.dev)}

    stats = stats_from_files(args.out_dir)
    check_against_selection(stats, selection)
    manifest = render_manifest(
        stats, selection, shard_sizes, claim_totals, args.n_fill, args.min_sentences
    )
    (args.out_dir / "MANIFEST.md").write_text(manifest, encoding="utf-8")

    per_split, per_label = stats["per_split"], stats["per_label"]
    print(
        f"claims: {stats['n_claims']} (train {per_split.get('train', 0)}, dev "
        f"{per_split.get('dev', 0)}; " + ", ".join(f"{k} {per_label.get(k, 0)}" for k in LABELS)
        + f"; {stats['n_sentence_only']} with a sentence-only evidence set)"
    )
    print(
        f"pages: {stats['n_pages']} ({len(stats['evidence'])} evidence, "
        f"{selection.filler_from_links} filler from links, "
        f"{selection.filler_from_order} filler from shard order)"
    )
    for name in ("pages.jsonl", "claims.jsonl"):
        print(f"{name}: {stats['sizes'][name]:,} bytes, sha256 {stats['sha256'][name]}")
    print(f"wrote {args.out_dir / 'MANIFEST.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
