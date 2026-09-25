"""The CLI and ``show_stats()`` print names their console cannot encode.

On Windows a pipe or a file is written in cp1252. A function named ``数据``,
or a cache directory with CJK or emoji in its path, made ``cash inspect``,
``cash clear --function`` and ``cash.show_stats()`` raise UnicodeEncodeError
-- ``clear`` after it had deleted, so it reported failure for work it had
done. ``PYTHONIOENCODING=cp1252`` gives a Linux run the same stream.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.timeout(120)

_SCRIPT = """
import cash, time

@cash.cache
def 数据(x):
    time.sleep(0.2)  # past the persistence floor: the entry reaches disk
    return x

数据(1)
"""


def _run(cwd, *argv, script=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["PYTHONIOENCODING"] = "cp1252"
    env["CASH_CACHE_DIR"] = str(cwd / ".cash-数据")
    cmd = [sys.executable, *argv] if script else [sys.executable, "-m", "cash", *argv]
    return subprocess.run(cmd, capture_output=True, cwd=str(cwd), env=env, timeout=100)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "m.py").write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    ran = _run(tmp_path, "m.py", script=True)
    assert ran.returncode == 0, ran.stderr.decode("cp1252", "replace")
    return tmp_path


@pytest.mark.parametrize("argv", [("info",), ("inspect",), ("inspect", "--function", "数据")])
def test_the_listing_commands_escape_the_name(project, argv):
    out = _run(project, *argv)
    assert out.returncode == 0, out.stderr.decode("cp1252", "replace")
    assert b"UnicodeEncodeError" not in out.stderr
    assert b"\\u6570\\u636e" in out.stdout, out.stdout


def test_clear_function_reports_what_it_did(project):
    out = _run(project, "clear", "--function", "数据")
    assert out.returncode == 0, out.stderr.decode("cp1252", "replace")
    assert b"Cleared 1 entry" in out.stdout


def test_show_stats_escapes_the_name(cash_instance, monkeypatch):
    @cash_instance.cache
    def 数据(x):
        return x

    数据(1)
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252"))
    cash_instance.show_stats()  # raised UnicodeEncodeError
    sys.stdout.flush()
    assert b"\\u6570\\u636e" in buffer.getvalue()
