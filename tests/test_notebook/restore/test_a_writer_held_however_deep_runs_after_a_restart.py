"""A function that holds a builtin which changes what it is bound to or
handed runs after a restart, wherever it holds it.

``f = mk()``, then ``y = f(1)``: ``f`` calls a writer it holds, and the
trace records no Python code for it. After a restart plain Python makes the
write; a hit skipped it. A bound ``list.append`` held directly in ``f``'s
own closure was seen; these shapes were served, and the write was lost:

- a method named for no list of writers (``d.__setitem__``, ``arr.fill``);
- a ``partial`` over a builtin function that writes its argument
  (``setattr``, ``operator.setitem``, ``heapq.heappush``);
- a writer held one step further: in a helper's closure, in a dict of hooks,
  on an object's attribute.

Every call is kept, so a hit is not left to timing. The controls hold only
readers and count their runs through a note a hit may skip
(`tests._run_log`): they are still served.
"""

import functools

import numpy as np
import pandas as pd
import pytest

from cash.notebook.call_key import changes_its_closure, holds_a_stream
from tests._cell_driver import run_cash_cell
from tests._run_log import RunLog
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

SLEEP = f"(time.sleep({F}), 0)[1]"
FACTORY = "def mk():\n    note = {held}\n    def g(x):\n        note({args})\n        return x\n    return g\nf = mk()"
INIT = (
    "import time, functools, operator, heapq\nimport numpy as np\n"
    "class R:\n    pass\n"
    "calls = []\nd = {}\nr = R()\narr = np.zeros(2)\n"
)

#: shape -> (the cells defining ``f``, an expression for what its call wrote)
SHAPES = {
    "dict_setitem": (FACTORY.format(held="d.__setitem__", args="x, x"), "d"),
    "ndarray_fill": (FACTORY.format(held="arr.fill", args="x"), "arr.tolist()"),
    "partial_setattr": (FACTORY.format(held="functools.partial(setattr, r, 'v')", args="x"), "getattr(r, 'v', None)"),
    "partial_setitem": (FACTORY.format(held="functools.partial(operator.setitem, d)", args="x, x"), "d"),
    "partial_heappush": (FACTORY.format(held="functools.partial(heapq.heappush, calls)", args="x"), "calls"),
    "in_a_helpers_closure": (
        "def mk():\n    note = calls.append\n    def h(x):\n        note(x)\n"
        "    def g(x):\n        h(x)\n        return x\n    return g\nf = mk()",
        "calls",
    ),
    "in_a_dict_of_hooks": (
        "def mk():\n    hooks = {'n': calls.append}\n    def g(x):\n        hooks['n'](x)\n        return x\n    return g\nf = mk()",
        "calls",
    ),
    "on_an_objects_attribute": (
        "w = R()\nw.note = calls.append\n"
        "def mk():\n    o = w\n    def g(x):\n        o.note(x)\n        return x\n    return g\nf = mk()",
        "calls",
    ),
}


def _restart(magics):
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in list(magics.shell.user_ns):
        if not name.startswith("_") and name != "EVENT":
            magics.shell.user_ns.pop(name, None)


def _run_twice(magics, cells):
    """What ``seen`` is after Run All, then after a restart and Run All."""
    got = []
    for _ in range(2):
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
        got.append(magics.shell.user_ns["seen"])
        _restart(magics)
    return got


@pytest.mark.parametrize("shape", SHAPES)
def test_the_write_is_made_after_a_restart(cash_magics, cash_instance, shape):
    cash_instance.config.call_cost_floor_seconds = 0.0
    body, probe = SHAPES[shape]
    cells = [INIT + body, f"y = (f(1), {SLEEP})", f"seen = repr({probe})"]
    plain: dict = {}
    for cell in cells:
        exec(cell, plain)
    assert plain["seen"] not in ("{}", "[]", "None", "[0.0, 0.0]")  # the call writes
    assert _run_twice(cash_magics, cells) == [plain["seen"]] * 2


READERS = {
    "nothing_else": "note2 = abs",
    "a_read_only_method": "note2 = calls.count",
    "a_dict_of_readers": "note2 = {'n': d.get}['n']",
    "a_library_method_in_python": "note2 = pd.Series([1, 2]).add",
}


@pytest.mark.parametrize("held", READERS)
def test_a_function_holding_only_readers_is_still_served(cash_magics, cash_instance, held):
    """Positive control: the walk does not refuse what only reads."""
    cash_instance.config.call_cost_floor_seconds = 0.0
    log = RunLog()
    # The event's name, not the log: what the log holds changes as it fills.
    cash_magics.shell.user_ns["EVENT"] = log.event
    body = (
        "import sys, pandas as pd\n"
        f"def mk():\n    note = functools.partial(sys.audit, EVENT)\n    {READERS[held]}\n"
        "    def g(x):\n        note(x)\n        note2(x)\n        return x\n    return g\nf = mk()"
    )
    cells = [INIT + body, f"y = (f(1), {SLEEP})", "seen = y[0]"]
    assert _run_twice(cash_magics, cells) == [1, 1]
    assert log == [1], "the second run after the restart was served"


def test_a_writer_is_told_from_a_reader_by_what_it_is():
    """Read off the method itself, not its name: ``Series.add`` and
    ``Index.append`` are Python methods that hand back a new object, which
    a name list of writers refused."""

    def mk(held):
        def g(x):
            return held(x)

        return g

    calls, table = [], {}
    for writer in (
        calls.append,
        table.__setitem__,
        np.zeros(2).fill,
        functools.partial(setattr, object()),
        {"n": calls.append},
    ):
        assert holds_a_stream(mk(writer)) and changes_its_closure(mk(writer)), writer
    for reader in (calls.count, table.get, len, str.join, pd.Series([1]).add, pd.Index([1]).append):
        assert not holds_a_stream(mk(reader)) and not changes_its_closure(mk(reader)), reader
