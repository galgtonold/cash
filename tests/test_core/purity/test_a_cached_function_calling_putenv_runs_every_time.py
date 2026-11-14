"""A cached function that calls ``os.putenv`` or ``os.unsetenv`` runs
every time.

They change the environment a child process and C code read, and leave
``os.environ`` as it was, so the before-and-after look at ``os.environ``
(`cash.tracking.process_registries`) did not see them: the call was
stored, and a hit, or a new process, never set the variable a child then
looks for. Their audit events are counted instead, also for a ``putenv``
imported by name and one whose return is kept.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

_MAIN = """
import os
import subprocess
import sys
from os import putenv

import cash


@cash.cache
def setup(how, y):
    print("ran")
    if how == "putenv":
        os.putenv("CASH_TEST_NATIVE", str(y))
    elif how == "imported":
        _r = putenv("CASH_TEST_NATIVE", str(y))
    else:
        os.unsetenv("CASH_TEST_NATIVE")
    return y * 2


setup(sys.argv[1], 4)
setup(sys.argv[1], 4)
child = "import os; print('child', os.environ.get('CASH_TEST_NATIVE', '-'))"
print(subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, check=True).stdout.strip())
print(" ".join(w["code"] for w in setup.cache_info()["warnings"]))
"""


@pytest.mark.parametrize(("how", "child"), [("putenv", "child 4"), ("imported", "child 4"), ("unsetenv", "child -")])
def test_the_environment_is_set_on_every_call(tmp_path, how, child):
    (tmp_path / "main.py").write_text(textwrap.dedent(_MAIN), encoding="utf-8")
    env = {"CASH_TEST_NATIVE": "outer"}
    for _ in range(2):
        lines = run_python("main.py", how, cwd=tmp_path, env=env).stdout.splitlines()
        assert lines.count("ran") == 2, lines
        assert child in lines
        assert "IMPURE-OBSERVED-EFFECTS" in lines[-1]
