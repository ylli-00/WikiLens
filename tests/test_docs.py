"""Tests that keep the README and the docs honest: the first line, the pasted output, the links,
the anchors and the number of tests they state."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from wikilense.config import REPO_ROOT

README = REPO_ROOT / "README.md"
FRESH_CLONE_RUN = REPO_ROOT / "results" / "fresh_clone_run.md"
FIRST_LINE = (
    'A project for area 4, "Vector search and RAG, natively" for the '
    "[MariaDB student database projects, 2026-09](https://mariadb.org/bachelor_hackathon_2026-09/)."
)
#: The evaluators of 27 October 2026 asked for a short README with the feature on the first
#: screen; the long reference material lives in docs/. This bound keeps it from growing back.
README_MAX_LINES = 330
#: How far down the first runnable commands may start and the first output may end.
FIRST_SCREEN_LINES = 30

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def _markdown_files() -> list[Path]:
    """Every Markdown file git tracks or would track (a directory walk outside a checkout)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        files = [REPO_ROOT / line for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:
        files = [p for p in REPO_ROOT.rglob("*.md") if ".venv" not in p.parts]
    return sorted(set(files) | {README})


def _prose_lines(path: Path) -> list[str]:
    """The lines of a Markdown file outside fenced code blocks."""
    lines: list[str] = []
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return lines


def _code_blocks(path: Path) -> list[str]:
    """The contents of the fenced code blocks of a Markdown file, each ending in a newline."""
    blocks: list[str] = []
    current: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append("".join(part + "\n" for part in current))
                current = None
        elif current is not None:
            current.append(line)
    return blocks


def _slug(heading: str) -> str:
    """The anchor GitHub gives a heading: formatting dropped, lower case, spaces to hyphens."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    text = re.sub(r"[`*]", "", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    """Every heading anchor of a Markdown file, with GitHub's -1, -2 suffixes for repeats."""
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for line in _prose_lines(path):
        match = HEADING.match(line)
        if not match:
            continue
        slug = _slug(match.group(2))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def test_the_first_line_names_the_area_and_links_the_brief() -> None:
    assert README.read_text(encoding="utf-8").split("\n", 1)[0] == FIRST_LINE


def test_the_readme_stays_short_and_shows_one_question_on_the_first_screen() -> None:
    lines = README.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= README_MAX_LINES, (
        f"README.md has {len(lines)} lines; move reference material to docs/ instead"
    )
    first_screen = "\n".join(lines[:FIRST_SCREEN_LINES])
    assert "make setup" in first_screen
    assert '.venv/bin/wikilense query "' in first_screen
    assert "1 hit; model load" in first_screen  # the output of that query, not only the command


def test_the_first_screen_output_is_copied_from_the_recorded_run() -> None:
    recorded = FRESH_CLONE_RUN.read_text(encoding="utf-8")
    blocks = _code_blocks(README)
    output = next(block for block in blocks if block.startswith(" 1. "))
    assert output.rstrip("\n").endswith("ef_search 100)")
    assert output in recorded, "the README's first output block must be the recorded run, verbatim"
    command = next(block for block in blocks if "wikilense query" in block)
    query_line = next(line for line in command.splitlines() if "wikilense query" in line)
    assert f"$ {query_line}\n{output}" in recorded, "the README's command must be the recorded one"
    # The ingest report the README prints is the one that run ended with.
    report = next(block for block in blocks if block.startswith("  pages 100,"))
    assert report in recorded


def test_every_relative_link_and_anchor_resolves() -> None:
    problems: list[str] = []
    for path in _markdown_files():
        for line in _prose_lines(path):
            for target in LINK.findall(line):
                if re.match(r"[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:, broken: ...
                    continue
                file_part, _, anchor = target.partition("#")
                linked = path if not file_part else (path.parent / file_part).resolve()
                where = f"{path.relative_to(REPO_ROOT)}: ({target})"
                if not linked.exists():
                    problems.append(f"{where} -> no such file")
                elif anchor and linked.suffix == ".md" and anchor not in _anchors(linked):
                    problems.append(f"{where} -> no heading with that anchor")
    assert not problems, "\n".join(problems)


def test_the_stated_number_of_tests_is_the_number_collected(request: pytest.FixtureRequest) -> None:
    config = request.config
    selected = request.session.testscollected
    deselected = bool(config.getoption("keyword") or config.getoption("markexpr"))
    whole_suite = [Path(arg).name for arg in config.args] in (["tests"], [str(REPO_ROOT.name)], [])
    if deselected or not whole_suite or config.getoption("deselect", None):
        pytest.skip("checked only when the whole suite is collected")
    stated = {
        "README.md": re.findall(r"(\d+) tests\b", README.read_text(encoding="utf-8")),
        "docs/ARCHITECTURE.md": re.findall(
            r"(\d+) tests\b", (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        ),
    }
    for name, numbers in stated.items():
        assert numbers, f"{name} no longer states the number of tests"
        assert set(numbers) == {str(selected)}, (
            f"{name} says {sorted(set(numbers))} tests, the suite collects {selected}"
        )
    # docs/ARCHITECTURE.md also splits them by kind: "Pure tests (N)", "`db` tests (N)", ...
    items = request.session.items
    by_kind = {
        "`db` tests": sum(1 for item in items if item.get_closest_marker("db")),
        "`slow` tests": sum(1 for item in items if item.get_closest_marker("slow")),
    }
    by_kind["Pure tests"] = sum(
        1 for item in items if not item.get_closest_marker("db") and not item.get_closest_marker("slow")
    )
    architecture = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for label, count in by_kind.items():
        assert f"{label} ({count})" in architecture, f"docs/ARCHITECTURE.md: {label} should be {count}"
