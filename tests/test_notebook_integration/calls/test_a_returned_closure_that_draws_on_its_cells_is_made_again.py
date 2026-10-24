"""A call returning a function that draws on what its closure holds runs
again inside every Run All, as plain Python runs it.

``def mk(): c = itertools.count(); return (lambda: next(c),)``, then ``d =
mk()`` and ``a = (d[0](), d[0]())``: ``next(c)`` changes nothing the shared
reader names, so the result was stored, and the second Run All was handed
the very closure the first had drawn on: ``(2, 3)`` where plain Python gives
``(0, 1)``. A cell holding an iterator, generator or stream, or one the
function changes through ``next`` or ``heapq``, now keeps the result out of
the cache, as a counter's ``nonlocal`` does.
"""

from __future__ import annotations

import pytest

from cash.notebook.call_key import holds_a_closure_with_state
from tests._cell_driver import run_cash_cell

MAKERS = {
    "count_in_a_tuple": ("c = itertools.count()", "(lambda: next(c),)", "d[0]", (0, 1)),
    "count_bare": ("c = itertools.count()", "lambda: next(c)", "d", (0, 1)),
    "generator": ("c = (i for i in range(10))", "lambda: next(c)", "d", (0, 1)),
    "heappush": ("h = []", "lambda: (heapq.heappush(h, 1), len(h))[1]", "d", (1, 2)),
}


@pytest.mark.parametrize("shape", list(MAKERS))
def test_each_run_all_gets_a_fresh_closure(cash_magics, shape):
    state, returned, draw, expected = MAKERS[shape]
    cells = [
        f"import heapq, itertools, time\ndef mk():\n    time.sleep(0.05)\n    {state}\n    return {returned}",
        "d = mk()",
        f"a = ({draw}(), {draw}())",
    ]
    ns = cash_magics.shell.user_ns
    for run in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["a"] == expected, f"Run All {run + 1}"


def test_a_closure_that_only_reads_its_cells_keeps_no_state():
    """Control: ``lambda x: x * scale['k']`` over a dict it only reads."""
    scale = {"k": 2}
    assert not holds_a_closure_with_state((lambda x: x * scale["k"],))
