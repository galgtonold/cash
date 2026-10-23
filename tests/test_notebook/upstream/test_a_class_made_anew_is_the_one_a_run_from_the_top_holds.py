"""A class or function made anew by a run from the top is the one it holds.

``class Rec`` runs again on every Run All and binds a new class under the
same lineage, as its code is the same. ``B['c'] = Rec`` below it looked
current to the cell's run plan and was skipped, so ``B`` kept the class the
run before made: ``B['c'] is Rec`` was False where a run from the top gives
True, and an iterator set on the new class was not where ``B['c']`` led.

A call that draws from an iterator the notebook holds -- a global, an
attribute of a class or object, an item of a default -- was served from its
stored result the next time, which left the iterator where it was: the call
after it drew the value the first one had.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.core

BIG = "B = {i: {'id': i} for i in range(20000)}\n"
HOLDERS = {
    "class-in-one-cell": ["class Rec:\n    pass\n" + BIG + "B['c'] = Rec\n", "same = B['c'] is Rec"],
    "class-cell-above": ["class Rec:\n    pass\n", BIG + "B['c'] = Rec\n", "same = B['c'] is Rec"],
    "function": ["def Rec():\n    return 1\n" + BIG + "B['c'] = Rec\n", "same = B['c'] is Rec"],
}


def _run_all(cash_magics, cells, times):
    for _ in range(times):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)


@pytest.mark.parametrize("shape", sorted(HOLDERS))
def test_a_container_holds_the_class_a_run_from_the_top_made(cash_magics, shape):
    cells = HOLDERS[shape]
    _run_all(cash_magics, cells, 3)
    assert cash_magics.shell.user_ns["same"] is True, "the container kept the class an earlier run made"


S = "import time\n"
DRAWS = {
    "global": [S + "def g(k):\n    time.sleep(0.05)\n    return next(it)\n", "it = iter(range(100, 200))"],
    "class-attribute": [
        S + "class Rec:\n    pass\ndef g(k):\n    time.sleep(0.05)\n    return next(Rec.it)\n",
        "Rec.it = iter(range(100, 200))",
    ],
    "object-attribute": [
        S + "class Rec:\n    pass\nR = Rec()\ndef g(k):\n    time.sleep(0.05)\n    return next(R.it)\n",
        "R.it = iter(range(100, 200))",
    ],
    "item-of-a-default": [
        S
        + "class Rec:\n    pass\n"
        + BIG
        + "B['c'] = Rec\ndef g(k, b=B):\n    time.sleep(0.05)\n    return next(b['c'].it)\n",
        "Rec.it = iter(range(100, 200))",
    ],
}


@pytest.mark.parametrize("shape", sorted(DRAWS))
def test_a_call_drawing_from_a_held_iterator_draws_each_time(cash_magics, shape):
    cells = [*DRAWS[shape], "y1 = g(1)", "y2 = g(1)"]
    for _ in range(3):
        _run_all(cash_magics, cells, 1)
        ns = cash_magics.shell.user_ns
        assert (ns["y1"], ns["y2"]) == (100, 101), "a stored call result left the iterator where it was"
