from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "vendor"}


def repository_files(pattern: str) -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob(pattern)
        if not any(part in EXCLUDED_PARTS for part in path.parts)
    )


def test_all_repository_python_sources_compile() -> None:
    failures: list[str] = []
    python_files = repository_files("*.py")
    assert python_files, "No Python source files found"

    for path in python_files:
        try:
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            relative = path.relative_to(REPO_ROOT)
            failures.append(f"{relative}:{exc.lineno}:{exc.offset}: {exc.msg}")

    assert not failures, "Python syntax failures:\n" + "\n".join(failures)


def test_all_repository_bash_scripts_parse() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available on this platform")

    failures: list[str] = []
    bash_files = repository_files("*.bash")
    assert bash_files, "No Bash scripts found"

    for path in bash_files:
        completed = subprocess.run(
            [bash, "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            relative = path.relative_to(REPO_ROOT)
            message = completed.stderr.strip() or completed.stdout.strip()
            failures.append(f"{relative}: {message}")

    assert not failures, "Bash syntax failures:\n" + "\n".join(failures)
