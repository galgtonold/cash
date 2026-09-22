"""Calling a method through its class does not warn KEY-UNHASHABLE-GLOBAL.

Found while checking annotations (round 25): ``@cash.cache def parse(v):
return A.make(v)`` with ``make`` a classmethod warned that it "reads 'A.make'
whose value could not be hashed, so changes to it will NOT invalidate the
cache". Editing ``make`` did invalidate -- the method is followed as code. The
class-attribute channel read ``A.make`` statically, got the ``classmethod``
object, which is not callable, and tried to hash it as a data constant.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

MODELS = textwrap.dedent('''
    import functools

    class A:
        RATE = {RATE}

        @classmethod
        def make(cls, v):
            return v * {FACTOR}

        @staticmethod
        def twice(v):
            return v * 2

        @property
        def prop(self):
            return 1

        @functools.cached_property
        def cached(self):
            return 2
''')

MAIN = textwrap.dedent('''
    import time
    import cash
    from models import A

    @cash.cache
    def parse(v):
        time.sleep(0.3)  # @cash:assume-safe
        _ = (A.prop, A.cached)
        return A.twice(A.make(v)) + A.RATE

    print("RESULT", parse(2))
''')


def _run(tmp_path, factor=10, rate=0):
    (tmp_path / "models.py").write_text(MODELS.replace("{FACTOR}", str(factor)).replace("{RATE}", str(rate)))
    (tmp_path / "main.py").write_text(MAIN)
    # No .pyc: Python validates one by whole-second mtime and size, so the
    # RATE 0 -> 1 edit (same size) landing in the second run's second loaded
    # the old bytecode, and printed 400 with or without cash.
    env = dict(os.environ, CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONWARNINGS="always",
               PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "main.py"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip(), proc.stderr


def test_no_unhashable_warning_and_edits_still_invalidate(tmp_path):
    out, err = _run(tmp_path)
    assert out == "RESULT 40"
    assert "KEY-UNHASHABLE-GLOBAL" not in err, err

    out, _ = _run(tmp_path, factor=100)
    assert out == "RESULT 400", "editing the classmethod served the old result"

    out, _ = _run(tmp_path, factor=100, rate=1)
    assert out == "RESULT 401", "editing the class constant served the old result"
