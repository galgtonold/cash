"""A method rebound on the class of an argument keys the globals the new
method reads, not those the old one read.

What a class's functions read was remembered per class: after a call met
``Vec`` with ``Vec.__add__`` reading nothing, ``Vec.__add__ = _add`` with an
``_add`` reading ``K`` kept the old list, so ``K = 6`` was served the result
made with ``K = 5``. Without a trace (Python below 3.12, or ``CASH_TRACE=0``)
nothing else caught it. The oracle is the same program without the decorator.
"""

from __future__ import annotations

import json

import pytest

from tests._scripts import run_python

PROGRAM = """
import json, sys
import cash
cache = cash.cache if sys.argv[1] == "cached" else (lambda fn: fn)
K = 5
class Vec:
    def __init__(self, n):
        self.n = n
@cache
def f(v):
    return v + 1
def _add(self, x):
    return -x
Vec.__add__ = _add
first = f(Vec(1))
def _add(self, x):
    return self.n + x + K
Vec.__add__ = _add
second = f(Vec(1))
K = 6
third = f(Vec(1))
print(json.dumps([first, second, third]))
"""


@pytest.mark.parametrize("trace", ["0", "1"])
def test_a_global_only_the_rebound_method_reads_is_keyed(tmp_path, trace):
    script = tmp_path / "main.py"
    script.write_text(PROGRAM, encoding="utf-8")

    def run(mode):
        done = run_python(script, mode, cwd=tmp_path, env={"CASH_TRACE": trace})
        return json.loads(done.stdout.strip().splitlines()[-1])

    assert run("cached") == run("plain") == [-1, 7, 8]
