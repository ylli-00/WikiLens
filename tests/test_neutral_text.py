"""Comments, docstrings and messages in the code stay neutral and hold no personal details.

The rule is in ``CLAUDE.md``, section "Comments, docstrings and personal details". These tests
scan the code, SQL and configuration files for what that rule excludes: notes on who decided
something, references to chat sessions, review findings and decision lists, paths into a home or
user folder, email addresses, and IP addresses other than the allowed ones. Data, results, docs
and the prompt sets (whose bytes are hashed into run records) are not scanned.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterator
from pathlib import Path

from wikilense.config import REPO_ROOT

#: Folders scanned: every file below them with a suffix in :data:`SUFFIXES`.
ROOTS = ("analysis", "scripts", "sql", "tests", "wikilense", ".github")
#: Files at the top of the repository that are scanned too.
TOP_FILES = ("Makefile", "pyproject.toml", "docker-compose.yml", ".env.example")
SUFFIXES = frozenset({".py", ".sql", ".sh", ".yml", ".yaml", ".toml"})
#: Relative paths not scanned: the prompt sets (hashed into run records) and this file, which
#: holds the patterns themselves.
SKIPPED = ("wikilense/llm/prompts", "tests/test_neutral_text.py")
SKIPPED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})

#: (rule, pattern). A match is a finding unless :func:`allowed` accepts it.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("who decided", re.compile(r"\bowner's\b|\bthe owner\b|\(owner\b", re.IGNORECASE)),
    ("chat session", re.compile(r"\b[Cc]hat [AB]\b")),
    ("review or decision id", re.compile(r"\bkit (?:review|REPORT)\b|\bD-\d{2,3}\b"
                                         r"|\[[A-E]\d{1,2}(?:, ?[A-E]\d{1,2})*\]")),
    ("review or decision id", re.compile(r"\bdecisions? #?\d", re.IGNORECASE)),
    ("notes folder", re.compile(r"\breviews/")),
    ("home or user path", re.compile(r"/home/[A-Za-z_]|\b[A-Za-z]:[\\/]+Users\b"
                                     r"|/mnt/[\w.-]+/Users/|(?<![\w.])/Users/[A-Za-z]")),
    ("email address", re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
    # not after "==": a four-part version pin such as nvidia-cufft==12.0.0.61
    ("IP address", re.compile(r"(?<![\w.])(?<!==)(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)")),
)
#: Email domains reserved for examples (RFC 2606).
EXAMPLE_DOMAINS = ("example.org", "example.com", "example.net")
#: IP addresses allowed: loopback and "every interface".
ALLOWED_IPS = frozenset({"127.0.0.1", "0.0.0.0"})


def allowed(rule: str, text: str) -> bool:
    """Return True when a match of ``rule`` is one of the allowed forms."""
    if rule == "email address":
        domain = text.rsplit("@", 1)[1].lower()
        return any(domain == d or domain.endswith("." + d) for d in EXAMPLE_DOMAINS)
    if rule == "IP address":
        try:
            ipaddress.ip_address(text)
        except ValueError:  # an octet over 255: a version number, not an address
            return True
        return text in ALLOWED_IPS
    return False


def findings_in_text(text: str) -> list[tuple[int, str, str]]:
    """Return ``(line number, rule, matched text)`` for every finding in ``text``."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in PATTERNS:
            for match in pattern.finditer(line):
                if not allowed(rule, match.group()):
                    found.append((number, rule, match.group()))
    return found


def scanned_files(root: Path = REPO_ROOT) -> Iterator[Path]:
    """Yield the files the rule covers, in sorted order."""
    candidates = [root / name for name in TOP_FILES]
    for folder in ROOTS:
        candidates += [p for p in (root / folder).rglob("*") if p.suffix in SUFFIXES]
    for path in sorted(set(candidates)):
        rel = path.relative_to(root).as_posix()
        if (path.is_file() and not SKIPPED_DIRS.intersection(path.parts)
                and not any(rel == s or rel.startswith(s + "/") for s in SKIPPED)):
            yield path


def findings(root: Path = REPO_ROOT) -> list[str]:
    """Return ``path:line: rule: text`` for every finding in the scanned files of ``root``."""
    out = []
    for path in scanned_files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        out += [f"{rel}:{n}: {rule}: {hit}" for n, rule, hit in findings_in_text(text)]
    return out


def test_the_patterns_catch_the_excluded_forms_and_let_the_allowed_ones_pass() -> None:
    caught = [
        "as the owner decided", "(owner, 2026-11-15)", "the owner's choice", "chat B's file",
        "kit review C4", "kit D-42", "D-117", "[A19, C22]", "decision #4c", "decisions 1, 2",
        "see reviews/NOTES.md", "/home/someone/project", "C:\\Users\\someone", "/mnt/c/Users/x/",
        "/Users/someone/", "someone@university.edu", "192.168.1.20",
    ]
    for text in caught:
        assert findings_in_text(text), text
    passing = [
        "readable by its owner only", "127.0.0.1:3306", "0.0.0.0", "'wikilense'@'localhost'",
        "someone@example.org", "~/.cache/huggingface", "nvidia-cufft==12.0.0.61",
        "@pytest.mark.db", "'wikilense'@'%'", "CLI 2.1.322", "the decision rule",
    ]
    for text in passing:
        assert not findings_in_text(text), text


def test_the_code_text_is_neutral_and_holds_no_personal_details() -> None:
    found = findings()
    assert not found, (
        'excluded by CLAUDE.md, section "Comments, docstrings and personal details":\n'
        + "\n".join(found)
    )
