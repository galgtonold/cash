"""Code whose reads lead back to itself is keyed, not folded forever.

A helper reads a table holding a class whose method calls the helper again:
folding what the helper reads folds the class's code, whose helpers' reads
fold the helper, and so on. networkx does this throughout (a graph method
reads the module, whose classes' methods read it again), and the key build
died of RecursionError: nothing reaching such code cached. The fold already
running folds the helper in full, so meeting it again is a back-reference.
"""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest

from tests._scripts import CASH_SRC, run_python

pytestmark = [pytest.mark.core]

REG = """
    RATE = {rate}


    class Shape:
        def area(self):
            return lookup() * RATE


    KINDS = {{"shape": Shape}}


    def lookup():
        return len(KINDS)
"""

JOB = """
    import cash
    import reg


    @cash.cache
    def run(n):
        return reg.KINDS["shape"]().area() * n


    print(run(2), run.cache_info()["misses"], [w["code"] for w in run.cache_info()["warnings"]])
"""


def _write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    past = time.time() - 30  # saved a while before the run, like an ordinary edit
    os.utime(path, (past, past))


def _run(project: Path) -> str:
    env = {"PYTHONPATH": os.pathsep.join([CASH_SRC, str(project)])}
    return run_python("job.py", cwd=project, env=env).stdout.strip().splitlines()[-1]


def test_a_helper_reached_again_through_its_own_reads_caches_and_keys_edits(tmp_path):
    _write(tmp_path / "reg.py", REG.format(rate=3))
    _write(tmp_path / "job.py", JOB)
    assert _run(tmp_path) == "6 1 []"
    assert _run(tmp_path) == "6 0 []"
    _write(tmp_path / "reg.py", REG.format(rate=5))
    assert _run(tmp_path) == "10 1 []"
