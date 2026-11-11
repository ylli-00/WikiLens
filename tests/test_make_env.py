"""Tests for scripts/make_env.py: the ``.env`` that `make setup` writes on its first run."""

from __future__ import annotations

import importlib.util
import io
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

from wikilense.config import REPO_ROOT, load_settings

SCRIPT = REPO_ROOT / "scripts" / "make_env.py"
EXAMPLE = REPO_ROOT / ".env.example"
PLACEHOLDERS = {"change-me", "change-me-too"}
#: The pattern of the Makefile's check-env recipe (its `grep -E`), written for Python's `re`.
MAKEFILE_PLACEHOLDER_CHECK = re.compile(
    r"^\s*(export\s+)?WIKILENSE_DB_(ROOT_)?PASSWORD\s*=\s*[\"']?change-me(-too)?[\"']?\s*$",
    re.MULTILINE,
)


@pytest.fixture
def make_env() -> Any:
    spec = importlib.util.spec_from_file_location("make_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_file_still_holds_the_placeholders_the_makefile_refuses() -> None:
    values = dotenv_values(EXAMPLE)
    assert {values["WIKILENSE_DB_PASSWORD"], values["WIKILENSE_DB_ROOT_PASSWORD"]} == PLACEHOLDERS
    assert MAKEFILE_PLACEHOLDER_CHECK.search(EXAMPLE.read_text(encoding="utf-8"))


def test_render_replaces_only_the_two_password_values(make_env: Any) -> None:
    example = EXAMPLE.read_text(encoding="utf-8")
    passwords = iter(["aaaa", "bbbb"])

    rendered = make_env.render(example, new_password=lambda: next(passwords))

    before, after = example.splitlines(), rendered.splitlines()
    assert len(before) == len(after)
    changed = [(old, new) for old, new in zip(before, after, strict=True) if old != new]
    assert changed == [
        ("WIKILENSE_DB_PASSWORD=change-me", "WIKILENSE_DB_PASSWORD=aaaa"),
        ("WIKILENSE_DB_ROOT_PASSWORD=change-me-too", "WIKILENSE_DB_ROOT_PASSWORD=bbbb"),
    ]
    assert rendered.endswith("\n")


def test_render_draws_two_different_strong_passwords(make_env: Any) -> None:
    rendered = make_env.render(EXAMPLE.read_text(encoding="utf-8"))
    values = dotenv_values(stream=io.StringIO(rendered))
    user, root = values["WIKILENSE_DB_PASSWORD"], values["WIKILENSE_DB_ROOT_PASSWORD"]

    assert user != root
    for password in (user, root):
        assert re.fullmatch(r"[0-9a-f]{32}", password)
    # Every other setting keeps the example's value.
    example_values = dotenv_values(EXAMPLE)
    for name, value in example_values.items():
        if name not in make_env.PASSWORD_VARIABLES:
            assert values[name] == value


def test_render_keeps_an_export_prefix_and_refuses_an_incomplete_example(make_env: Any) -> None:
    rendered = make_env.render(
        "export WIKILENSE_DB_PASSWORD = x\n  WIKILENSE_DB_ROOT_PASSWORD=y\n",
        new_password=lambda: "p",
    )
    assert rendered == "export WIKILENSE_DB_PASSWORD =p\n  WIKILENSE_DB_ROOT_PASSWORD=p\n"

    with pytest.raises(make_env.EnvError, match="WIKILENSE_DB_ROOT_PASSWORD"):
        make_env.render("WIKILENSE_DB_PASSWORD=x\n# WIKILENSE_DB_ROOT_PASSWORD=y\n")


def test_create_writes_a_private_file_that_passes_the_makefile_check(
    make_env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real environment variable would win over the file (CI exports its own password).
    monkeypatch.delenv("WIKILENSE_DB_PASSWORD", raising=False)
    out = tmp_path / ".env"

    assert make_env.create(str(EXAMPLE), str(out)) is True

    text = out.read_text(encoding="utf-8")
    assert not MAKEFILE_PLACEHOLDER_CHECK.search(text)
    if os.name == "posix":
        assert stat.S_IMODE(out.stat().st_mode) == 0o600
    # The package reads the generated file like a hand-written one.
    settings = load_settings(out)
    assert settings.db_password == dotenv_values(out)["WIKILENSE_DB_PASSWORD"]
    assert settings.db_password not in PLACEHOLDERS


def test_create_never_touches_an_existing_file(make_env: Any, tmp_path: Path) -> None:
    out = tmp_path / ".env"
    out.write_text("WIKILENSE_DB_PASSWORD=mine\n", encoding="utf-8")

    assert make_env.create(str(EXAMPLE), str(out)) is False
    assert out.read_text(encoding="utf-8") == "WIKILENSE_DB_PASSWORD=mine\n"


def test_create_reports_a_missing_example(make_env: Any, tmp_path: Path) -> None:
    with pytest.raises(make_env.EnvError, match="cannot read"):
        make_env.create(str(tmp_path / "absent.example"), str(tmp_path / ".env"))
    assert not (tmp_path / ".env").exists()


def test_the_command_line_prints_no_password(tmp_path: Path) -> None:
    out = tmp_path / ".env"
    command = [sys.executable, str(SCRIPT), "--example", str(EXAMPLE), "--out", str(out)]

    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)

    assert first.returncode == 0 and second.returncode == 0
    assert "Created" in first.stdout and "kept as it is" in second.stdout
    values = dotenv_values(out)
    for password in (values["WIKILENSE_DB_PASSWORD"], values["WIKILENSE_DB_ROOT_PASSWORD"]):
        assert password not in first.stdout + first.stderr + second.stdout + second.stderr

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--example", str(tmp_path / "absent"), "--out", str(tmp_path / "x")],
        capture_output=True, text=True, check=False,
    )
    assert missing.returncode == 1 and missing.stderr.startswith("make_env: cannot read")
