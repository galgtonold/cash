"""A call of a lambda that changes what its closure holds is not served
the first call's answer.

``textwrap.t = (mk(),)`` with ``mk`` returning ``lambda: xs.pop()``: ``a =
t[0]()`` popped 99, and ``b = t[0]()`` was served that same 99 from the call
cache where a run pops 98. A lambda has no ``def`` of its own to read, so
the pop was not found in its text and ``xs`` was not captured with the
entry; it is now read from the lambda's line.
"""

import pytest

from cash.decorator import code_trace
from cash.notebook import call_unit, shadow
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

IMP = "import time, textwrap"
SHAPES = {
    "list_pop": "xs = list(range(100))\n    return lambda: xs.pop()",
    "dict_popitem": "d = {i: i for i in range(100)}\n    return lambda: d.popitem()[0]",
}


@pytest.mark.parametrize("trace", [True, False], ids=["trace", "no_trace"])
@pytest.mark.parametrize("shape", SHAPES)
def test_the_second_call_pops_again(cash_magics, monkeypatch, shape, trace):
    if trace and not code_trace.on():
        pytest.skip("no trace before 3.12")
    if not trace:
        monkeypatch.setenv("CASH_TRACE", "0")
        monkeypatch.setattr(shadow, "SHADOW", None)
    # The pop takes no time: stored only with no floor for a call's own run.
    monkeypatch.setattr(call_unit.CallUnit, "_cost_floor_s", lambda self: 0.0)
    cells = [
        IMP,
        f"def mk():\n    {SHAPES[shape]}\ntextwrap.t = (mk(),)",
        f"a = (time.sleep({F}), textwrap.t[0]())[1]",
        f"b = (time.sleep({F}), textwrap.t[0]())[1]",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert (ns["a"], ns["b"]) == (99, 98)
