"""Two loops with the same header keep their own iteration measurements.

The heavy-loop store remembers a loop by its header with the numbers
blanked, so ``for k in range(130)`` around a slow call and a cheap
``for k in range(3000)`` were one loop to it: the cheap one ran pass by pass
on the slow one's measurement (seconds where it took milliseconds), and its
own measurement then sent the slow one back to one unit.
"""

import ast

from cash.notebook import heavy_loops
from tests._cell_driver import run_cash_cell

SLOW = "for k in range(130):\n    time.sleep(0.035)"
CHEAP = "total = 0\nfor k in range(3000):\n    total += k * 2"


def _loop(src):
    return next(n for n in ast.parse(src).body if isinstance(n, ast.For))


def _store():
    return heavy_loops.HeavyLoopStore(None)


def test_a_loop_with_another_body_does_not_read_the_measurement():
    store = _store()
    store.record(_loop(SLOW), 0.035, "unit")
    assert store.get(_loop(CHEAP)) is None
    assert store.get(_loop(SLOW)) == (0.035, "unit")


def test_each_loop_keeps_its_own_measurement():
    store = _store()
    store.record(_loop(SLOW), 0.035, "unit")
    store.record(_loop(CHEAP), 0.000001)
    assert store.get(_loop(SLOW)) == (0.035, "unit")
    assert store.get(_loop(CHEAP)) == (0.000001, "")


def test_an_edit_to_part_of_the_body_keeps_the_measurement():
    """The store's point: the verdict carries over the edits that follow. A
    changed number is no edit to it, and nor is half the body."""
    before = "for k in range(50):\n    s = slow(k)\n    res.append(s + 1)"
    store = _store()
    store.record(_loop(before), 0.2, "unit")
    assert store.get(_loop(before.replace("50", "80").replace("+ 1", "+ 2"))) == (0.2, "unit")
    assert store.get(_loop(before.replace("res.append(s + 1)", "out.add(s)"))) == (0.2, "unit")
    assert store.get(_loop("for k in range(50):\n    t = fast(k)\n    out.add(t)")) is None


def test_a_cheap_loop_under_a_slow_ones_header_runs_as_one_unit(cash_magics, mock_shell):
    cells = ["import time", SLOW, CHEAP]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    store = heavy_loops.store_for_backend(cash_magics._statement_processor.cash_instance.backend)
    store.record(_loop(SLOW), 0.035, "unit")
    run_cash_cell(cash_magics, CHEAP, cells=cells)
    assert len(cash_magics._last_cell_metrics["statements"]) < 10
    assert mock_shell.user_ns["total"] == sum(k * 2 for k in range(3000))
    assert store.get(_loop(SLOW)) == (0.035, "unit")
