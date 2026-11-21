"""Tests of the Makefile's setup recipe: which requirements file and which package index pip gets."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from wikilense.config import REPO_ROOT

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")

#: The `--extra-index-url` line of requirements-cpu.lock: PyTorch's CPU-only index.
CPU_INDEX = re.search(r"^--extra-index-url (\S+)$",
                      (REPO_ROOT / "requirements-cpu.lock").read_text(encoding="utf-8"),
                      re.MULTILINE).group(1)


def pip_line(*args: str) -> str:
    """Return the pip command that `make -n setup` prints for the given make arguments."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("MAKEFLAGS", "MFLAGS", "CPU", "GPU", "REQS", "UNAME", "VENV")}
    out = subprocess.run(["make", "-n", "-C", str(REPO_ROOT), "VENV=/venv", "setup", *args],
                         env=env, check=True, capture_output=True, text=True).stdout
    (line,) = [line for line in out.splitlines() if line.startswith("/venv/bin/pip install")]
    return line.strip()


def test_the_default_on_linux_takes_torch_from_the_cpu_index() -> None:
    assert pip_line("UNAME=Linux") == (
        f"/venv/bin/pip install -r requirements-dev.txt -e . --extra-index-url {CPU_INDEX}")


@pytest.mark.parametrize(("args", "reqs"), [
    (("UNAME=Linux", "GPU=1"), "requirements-dev.txt"),
    (("UNAME=Linux", "CPU=1"), "requirements-cpu.lock"),
    (("UNAME=Linux", "REQS=requirements.lock"), "requirements.lock"),
    (("UNAME=Darwin",), "requirements-dev.txt"),
])
def test_gpu_a_lock_file_and_macos_get_no_extra_index(args: tuple[str, ...], reqs: str) -> None:
    assert pip_line(*args) == f"/venv/bin/pip install -r {reqs} -e ."
