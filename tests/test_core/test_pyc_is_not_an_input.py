"""A module's own ``__pycache__/*.pyc`` is not a file input of a cached function.

Round 20 (r20s4): ``f`` calls the cached ``load``; ``load``'s key is built
inside ``f``'s body, and building it checks that the code this process loaded
still matches the file -- which reads the module's ``.pyc`` header through
``open``, where ``f``'s tracker recorded it. Editing ANOTHER function in the
module rewrites the ``.pyc``, so ``f`` recomputed: a 25 s step on every deploy
of the tester's tool, and reinstalling identical code did the same.

Fresh processes, with bytecode written, and the cache filled in a run where the
``.pyc`` already existed -- the only arrangement in which it is read.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

MOD = '''\
import sys, time
import cash


@cash.cache
def load(n):
    print("[RUN] load", file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe
    return list(range(n))


@cash.cache
def f(n):
    print("[RUN] f", file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe
    return sum(load(n))


@cash.cache
def g(n):
    print("[RUN] g", file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe
    return len(load(n)) + 1  # G_BODY
'''

MAIN = "import mod\nprint(mod.f(200_000), mod.g(200_000))\n"


def _run(proj):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CASH_") and k != "PYTHONDONTWRITEBYTECODE"}
    env.update(CASH_CACHE_DIR=str(proj / ".cash"), CASH_DEBUG="1")
    p = subprocess.run([sys.executable, "main.py"], cwd=str(proj), env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    ran = {line.split()[1] for line in p.stderr.splitlines() if line.startswith("[RUN] ")}
    return p, ran


def test_editing_a_sibling_function_does_not_rerun_a_caller_via_the_pyc(tmp_path):
    (tmp_path / "mod.py").write_text(MOD, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    _run(tmp_path)                                     # writes mod's .pyc
    shutil.rmtree(tmp_path / ".cash")
    first, ran = _run(tmp_path)                        # the .pyc predates this process
    assert ran == {"load", "f", "g"}
    assert "__pycache__" not in first.stderr, "a .pyc was recorded as an input"
    _, ran = _run(tmp_path)
    assert ran == set(), "the entries never reached disk: this test proves nothing"

    (tmp_path / "mod.py").write_text(MOD.replace("+ 1  # G_BODY", "+ 2  # G_BODY"),
                                     encoding="utf-8")
    after, ran = _run(tmp_path)
    assert "f" not in ran, "editing g re-ran f: " + "\n".join(
        line for line in after.stderr.splitlines() if "FILE_DEP" in line or "MISS" in line)
    assert "g" in ran
