"""A cached function must see the module attributes it reads.

``import conf; conf.RATE`` went permanently stale: the only global the body
references is ``conf``, a module, and modules were filtered out before their
attributes were ever considered. The equivalent ``from conf import RATE``
invalidated correctly, so the same dependency was tracked or not depending on
which import spelling you happened to use -- and the failure was silent, with
``explain()`` reporting a confident ``[HIT]``.

Found by an adversarial tester sweep against the 0.1.0 wheel and reproduced
independently 3/3.

The first two tests run real scripts in fresh processes on purpose. The bug is
that a *second run* reuses the persisted entry after the constant changed,
which an in-process test cannot express; and writing the function inside a test
body would make ``conf`` a closure variable (``LOAD_DEREF``), which compiles
differently from the module-level ``import`` that real code uses.
"""
from __future__ import annotations

import subprocess
import sys
import types

import pytest

import cash

# The sleep is load-bearing, not padding. Cross-process persistence has a
# ~0.1s compute floor: a cheaper function is never written to disk, so every
# fresh process recomputes and the test passes whether or not the bug exists.
# Without it these two tests pass against the UNFIXED source -- a vacuous green.
MAIN = """\
import warnings; warnings.simplefilter('ignore')
import time
import cash
import conf
RAN = [0]

@cash.cache
def compute(x):
    RAN[0] += 1
    time.sleep(0.2)
    {body}

print(compute(10), RAN[0])
"""


def _write_main(tmp_path, body):
    (tmp_path / "main.py").write_text(MAIN.format(body=body), encoding="utf-8")


def _run(tmp_path):
    # ``-B`` (no bytecode) is load-bearing for determinism, not a speed choice.
    # These tests rewrite ``conf.py`` with SAME-SIZE edits (``RATE = 2.0`` ->
    # ``3.0`` -> ``7.0``) and immediately re-run. CPython's .pyc invalidation is
    # (mtime, size)-based, so a same-size rewrite landing within the filesystem's
    # mtime resolution leaves a STALE ``conf`` .pyc — the next process imports the
    # OLD constant, cash faithfully caches what Python loaded, and the assertion
    # sees a stale hit. Observed as an intermittent CI failure across OS/versions.
    # ``-B`` writes no .pyc, so ``conf.py`` is always compiled fresh from source.
    cp = subprocess.run(
        [sys.executable, "-B", "main.py"], cwd=str(tmp_path),
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, "script failed:\n" + cp.stdout + "\n" + cp.stderr
    return cp.stdout.strip()


def test_module_attribute_read_invalidates_when_it_changes(tmp_path):
    _write_main(tmp_path, "return x * conf.RATE")
    conf = tmp_path / "conf.py"

    conf.write_text("RATE = 2.0\n", encoding="utf-8")
    assert _run(tmp_path) == "20.0 1"

    conf.write_text("RATE = 3.0\n", encoding="utf-8")
    assert _run(tmp_path) == "30.0 1", "stale value served after the constant changed"

    conf.write_text("RATE = 7.0\n", encoding="utf-8")
    assert _run(tmp_path) == "70.0 1"


def test_helper_reached_through_the_module_sees_its_own_constants(tmp_path):
    """``conf.get_rate()`` whose SOURCE never changes but whose constant does.

    The helper-source channel hashes the callee's source, which is unchanged
    here, so nothing invalidated. One level of recursion into the helper's own
    module globals closes it.
    """
    _write_main(tmp_path, "return x * conf.get_rate()")
    conf = tmp_path / "conf.py"

    conf.write_text("RATE = 2.0\ndef get_rate():\n    return RATE\n", encoding="utf-8")
    assert _run(tmp_path) == "20.0 1"

    conf.write_text("RATE = 5.0\ndef get_rate():\n    return RATE\n", encoding="utf-8")
    assert _run(tmp_path) == "50.0 1", "helper's constant changed but nothing invalidated"


def test_installed_module_attrs_do_not_churn_the_key():
    """Reading a stdlib / site-packages attribute must not add key churn.

    This is the over-invalidation guard. Folding every module attribute would
    rope in things like ``os.environ``, making every call miss. Third-party and
    stdlib contents are fixed for a given environment, so ``_is_user_module``
    excludes them.
    """
    import math
    calls = []

    @cash.cache
    def compute(x):
        calls.append(x)
        return x * math.pi

    assert compute(2) == compute(2)
    assert len(calls) == 1, "a stdlib attribute read caused a spurious cache miss"


def test_function_reading_no_module_attrs_is_unaffected():
    """Regression guard: ordinary functions must keep hitting."""
    calls = []

    @cash.cache
    def plain(x):
        calls.append(x)
        return x + 1

    assert plain(1) == 2
    assert plain(1) == 2
    assert len(calls) == 1


def test_is_user_module_classification():
    """The stdlib / site-packages boundary itself."""
    import math
    import os as os_mod

    assert cash.Cash._is_user_module(math) is False, "stdlib must be excluded"
    assert cash.Cash._is_user_module(os_mod) is False, "stdlib must be excluded"
    assert cash.Cash._is_user_module(cash) is False, "cash's own modules must be excluded"

    fake = types.ModuleType("looks_like_user_code")
    fake.__file__ = "/home/someone/project/conf.py"
    assert cash.Cash._is_user_module(fake) is True

    builtin = types.ModuleType("no_file_at_all")
    assert cash.Cash._is_user_module(builtin) is False
