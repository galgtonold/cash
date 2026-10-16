"""A lock read through a name is not unhashable data.

``with argmap._compile_lock:`` in networkx (``with Registry._lock:``
here) reads a class's lock through the class's name. No result is computed
from a lock, and one held by a closure, a default or a class read through
``self`` was already left out quietly, but read this way it warned
KEY-UNHASHABLE-GLOBAL on every function reaching it. It is keyed by its
kind now, and the data beside it still is by value.
"""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest

from tests._scripts import CASH_SRC, run_python

pytestmark = [pytest.mark.core]

HELPERS = """
    import threading


    class Registry:
        _lock = threading.Lock()
        TOTAL = {total}


    def get():
        with Registry._lock:
            return Registry.TOTAL
"""

JOB = """
    import cash
    import lockreg


    @cash.cache
    def total(n):
        return lockreg.get() * n


    info = (total(2), total.cache_info())
    print(info[0], info[1]["misses"], [w["code"] for w in info[1]["warnings"]])
"""


def _write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    past = time.time() - 30  # saved a while before the run, like an ordinary edit
    os.utime(path, (past, past))


def _run(project: Path) -> str:
    env = {"PYTHONPATH": os.pathsep.join([CASH_SRC, str(project)])}
    return run_python("job.py", cwd=project, env=env).stdout.strip().splitlines()[-1]


def test_a_class_lock_read_by_name_neither_warns_nor_hides_the_data_beside_it(tmp_path):
    _write(tmp_path / "lockreg.py", HELPERS.format(total=1))
    _write(tmp_path / "job.py", JOB)
    assert _run(tmp_path) == "2 1 []"
    assert _run(tmp_path) == "2 0 []"
    _write(tmp_path / "lockreg.py", HELPERS.format(total=4))
    assert _run(tmp_path) == "8 1 []"
