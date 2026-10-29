"""Rebuilding an iterator for a cell re-run on its own does not repeat what
its draws write.

``rest = list(it)`` re-run alone after a Run All rebuilds ``it`` as the cells
above leave it: ``it = g()`` and ``first = [next(it) for _ in range(2)]`` run
again. The generator's body appends to ``log``, and ``log = []`` was not
re-run with them, as the draws were not known to change ``log``: it held 12
items, then 18, where a run from the top leaves 6. A draw now counts what
the code it runs changes (a generator's function, what a generator
expression or ``map`` calls) as a call would, so ``log`` is rebuilt first.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.core

GEN = "log = []\ndef g():\n    for i in range(6):\n        log.append(i)\n        yield i\n"
FN = "log = []\ndef f(x):\n    log.append(x)\n    return x\n"
FIRST = "first = [next(it) for _ in range(2)]"
TAIL = "rest = list(it)\nn = len(log)"
SHAPES = {
    "a generator": [GEN + "it = g()", FIRST, TAIL],
    "a generator expression": [FN + "it = (f(x) for x in range(6))", FIRST, TAIL],
    "a map": [FN + "it = map(f, range(6))", FIRST, TAIL],
    "a map over a generator": [GEN + "it = map(str, g())", FIRST, TAIL],
    "a zip over a generator": [GEN + "it = zip(g(), range(6))", FIRST, TAIL],
    "three maps over a generator": [GEN + "it = map(str, map(int, map(abs, g())))", FIRST, TAIL],
    "a tee of a generator": [GEN + "import itertools\nit = itertools.tee(g())[0]", FIRST, TAIL],
    "maps nested past the look": [GEN + "it = g()\nfor _ in range(12):\n    it = map(abs, it)", FIRST, TAIL],
    "the list in its own cell": ["log = []", GEN.replace("log = []\n", "") + "it = g()", FIRST, TAIL],
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_tail_re_run_alone_leaves_what_a_run_from_the_top_does(cash_magics, shape):
    cells = SHAPES[shape]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["n"] == 6
    for _ in range(2):
        run_cash_cell(cash_magics, cells[-1], cells=cells)
        assert ns["log"] == list(range(6))
        assert ns["n"] == 6
