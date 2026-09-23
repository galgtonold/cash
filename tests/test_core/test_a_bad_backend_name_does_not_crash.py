"""A misspelled backend or tier type is reported and skipped, not raised.

Found while attacking the decorator before round 26: ``[[tool.cash.tiers]]
type = "memry"`` raised ``ValueError: Unknown tier type`` out of ``import
cash`` -- and out of ``python -m cash info``, the command the docs send you to
when the configuration is wrong. Configuration's own contract
(getting-started/configuration.md): "A bad value in a TOML file or an
environment variable is reported once (CONFIG-INVALID) and skipped, so the
rest of the configuration still applies."
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent("""
    import cash
    @cash.cache
    def f(n):
        return n * 2
    print("RESULT", f(21))
""")


def _project(tmp_path, pyproject: str):
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "run.py").write_text(PROGRAM, encoding="utf-8")
    return tmp_path


def _run(project, *args, env_extra=None):
    import os

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env_extra or {})}
    return subprocess.run(
        [sys.executable, *args], cwd=str(project), env=env, capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize(
    "pyproject, bad",
    [
        ('[tool.cash]\n[[tool.cash.tiers]]\ntype = "memry"\n', "memry"),
        ('[tool.cash]\nbackend = "postgres"\n', "postgres"),
    ],
)
def test_a_program_runs_and_says_what_was_wrong(tmp_path, pyproject, bad):
    project = _project(tmp_path, pyproject)
    done = _run(project, "run.py")
    assert "RESULT 42" in done.stdout, (done.stdout, done.stderr)
    assert "Traceback" not in done.stderr, done.stderr
    assert bad in done.stderr and "CONFIG-INVALID" in done.stderr, done.stderr


@pytest.mark.parametrize(
    "pyproject, bad",
    [
        ('[tool.cash]\n[[tool.cash.tiers]]\ntype = "memry"\n', "memry"),
        ('[tool.cash]\nbackend = "postgres"\n', "postgres"),
    ],
)
def test_cash_info_still_reports_the_configuration(tmp_path, pyproject, bad):
    project = _project(tmp_path, pyproject)
    done = _run(project, "-m", "cash", "info")
    assert "Traceback" not in done.stderr, done.stderr
    assert "Backend:" in done.stdout, done.stdout
    assert bad not in done.stdout.split("Settings")[0], (
        "the resolved configuration must not present the bad value as in effect"
    )


def test_a_bad_backend_env_var_is_skipped(tmp_path):
    project = _project(tmp_path, "[tool.cash]\n")
    done = _run(project, "run.py", env_extra={"CASH_BACKEND": "postgres"})
    assert "RESULT 42" in done.stdout, (done.stdout, done.stderr)
    assert "Traceback" not in done.stderr, done.stderr
