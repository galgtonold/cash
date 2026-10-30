"""A function whose closure holds a builtin method that changes the object
it is bound to (``note = calls.append`` made inside a factory, or
``functools.partial(list.append, calls)``) changes that object when called.

``note(x)`` reads as a plain call, and the trace records no Python code for
``append``: after a restart, ``y = f(1)`` was served and ``calls`` stayed
empty, where a run appends. The same method held by a global, a default or
as a list in the closure was already seen.
"""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

SLEEP = f"(time.sleep({F}), 0)[1]"
FACTORY = "def mk():\n    note = {held}\n    def g(x):\n        note(x)\n        return x\n    return g\nf = mk()"

SHAPES = {
    "bound_method": FACTORY.format(held="calls.append"),
    "partial": FACTORY.format(held="functools.partial(list.append, calls)"),
    "list_in_the_closure": "def mk():\n    L = calls\n    def g(x):\n        L.append(x)\n        return x\n    return g\nf = mk()",
    "global_alias": "note = calls.append\ndef f(x):\n    note(x)\n    return x",
}


def _restart(magics):
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("calls", "note", "f", "mk", "y", "time", "functools"):
        magics.shell.user_ns.pop(name, None)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_append_runs_after_a_restart(cash_magics, shape):
    cells = ["import time, functools\ncalls = []\n" + SHAPES[shape], f"y = (f(1), {SLEEP})"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["calls"] == [1]
        _restart(cash_magics)


def test_a_closure_over_a_method_that_only_reads_is_still_served(cash_magics):
    """Positive control: ``count = calls.count`` changes nothing."""
    from cash.notebook.call_key import changes_its_closure

    calls = []

    def mk(held):
        def g(x):
            return held(x)

        return g

    assert changes_its_closure(mk(calls.append))
    assert not changes_its_closure(mk(calls.count))
    assert not changes_its_closure(mk(len))
