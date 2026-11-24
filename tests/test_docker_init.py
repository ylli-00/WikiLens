"""Tests of `sql/docker-init/01-test-database.sh`, which makes the test database when the MariaDB
container's data volume is created. The image's entrypoint sources the script when it is not
executable and runs it as a program when it is (every file is, on a Windows drive under WSL), so
both ways must send the same two statements. No server is needed: the client is a stand-in."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from wikilense.config import REPO_ROOT

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None, reason="needs bash on a POSIX system"
)

SCRIPT = REPO_ROOT / "sql" / "docker-init" / "01-test-database.sh"
EXPECTED = (
    "CREATE DATABASE IF NOT EXISTS `wikilense_test`;\n"
    "GRANT ALL PRIVILEGES ON `wikilense_test`.* TO 'wikilense'@'%';\n"
)


def _env(tmp_path: Path) -> dict[str, str]:
    """The container's variables, with the stand-in client first on the PATH."""
    return {
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
        "OUT": str(tmp_path / "out"),
        "MARIADB_DATABASE": "wikilense",
        "MARIADB_USER": "wikilense",
        "MARIADB_ROOT_PASSWORD": "root-secret",
    }


def _as_program(tmp_path: Path, client_exit: int = 0) -> subprocess.CompletedProcess:
    """Run an executable copy of the script with a stand-in `mariadb` that records its input."""
    client = tmp_path / "mariadb"
    client.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$MYSQL_PWD" "$@" > "$OUT.args"\ncat > "$OUT"\n'
        f"exit {client_exit}\n",
        encoding="utf-8",
    )
    client.chmod(0o755)
    program = tmp_path / SCRIPT.name
    shutil.copyfile(SCRIPT, program)
    program.chmod(0o755)
    return subprocess.run(
        [str(program)], check=False, env=_env(tmp_path), capture_output=True, text=True
    )


def test_sourced_the_script_hands_both_statements_to_docker_process_sql(tmp_path: Path) -> None:
    entrypoint = 'set -eo pipefail; docker_process_sql() { cat > "$OUT"; }; . "$1"'
    subprocess.run(["bash", "-c", entrypoint, "bash", str(SCRIPT)], env=_env(tmp_path), check=True)
    assert (tmp_path / "out").read_text(encoding="utf-8") == EXPECTED


def test_run_as_a_program_it_sends_them_as_root_through_the_socket(tmp_path: Path) -> None:
    result = _as_program(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out").read_text(encoding="utf-8") == EXPECTED
    assert (tmp_path / "out.args").read_text(encoding="utf-8").splitlines() == [
        "root-secret", "--protocol=socket", "-uroot", "-hlocalhost",
    ]


def test_run_as_a_program_it_fails_when_the_client_fails(tmp_path: Path) -> None:
    assert _as_program(tmp_path, client_exit=1).returncode != 0
