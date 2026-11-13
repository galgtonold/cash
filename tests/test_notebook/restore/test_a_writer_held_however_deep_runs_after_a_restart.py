"""A function that holds something which changes what it is bound to or
handed runs after a restart, wherever it holds it.

``f = mk()``, then ``y = f(1)``: ``f`` calls a writer it holds, and the
trace records no Python code for it. After a restart plain Python makes the
write; a hit skipped it. A bound ``list.append`` held directly in ``f``'s
own closure was seen; these shapes were served, and the write was lost:

- a method named for no list of writers (``d.__setitem__``, ``arr.fill``);
- a ``partial`` over a builtin function that writes its argument
  (``setattr``, ``operator.setitem``, ``heapq.heappush``);
- a writer held one step further: in a helper's closure, in a dict of hooks,
  on an object's attribute, on a class the notebook names (or in a dict, a
  list or an object such a class holds);
- a ``partial`` handing ``out=`` a mutable array, or over ``np.copyto``;
- an ``operator.methodcaller`` of a method that writes;
- ``defaultdict.__getitem__``, which inserts a missing key.

A library's Python method named as one that writes (``counter.update``,
``series.update``, ``UserList.append``) is kept as a writer: the trace
records the user's code, not a library's.

Every call is kept, so a hit is not left to timing. The controls hold only
readers and count their runs through a note a hit may skip
(`tests._run_log`): they are still served.
"""

import sys

import pytest

from tests._cell_driver import run_cash_cell
from tests._run_log import RunLog
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

SLEEP = f"(time.sleep({F}), 0)[1]"
FACTORY = "def mk():\n    note = {held}\n    def g(x):\n        note({args})\n        return x\n    return g\nf = mk()"
INIT = (
    "import time, functools, operator, heapq, collections\nimport numpy as np, pandas as pd\n"
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
    "on_a_named_class": (
        "class H:\n    note = calls.append\ndef mk():\n    h = H\n    def g(x):\n        h.note(x)\n        return x\n"
        "    return g\nf = mk()",
        "calls",
    ),
    "in_a_dict_on_a_named_class": (
        "class H:\n    hooks = {'n': calls.append}\ndef mk():\n    h = H\n    def g(x):\n        h.hooks['n'](x)\n"
        "        return x\n    return g\nf = mk()",
        "calls",
    ),
    "in_a_list_on_a_named_class": (
        "class H:\n    hooks = [calls.append]\ndef mk():\n    h = H\n    def g(x):\n        h.hooks[0](x)\n"
        "        return x\n    return g\nf = mk()",
        "calls",
    ),
    "on_an_object_on_a_named_class": (
        "sink = R()\nsink.w = calls.append\nclass H:\n    s = sink\ndef mk():\n    h = H\n    def g(x):\n"
        "        h.s.w(x)\n        return x\n    return g\nf = mk()",
        "calls",
    ),
    "counter_update": (
        "c = collections.Counter()\n" + FACTORY.format(held="c.update", args="'ab'"),
        "sorted(c.items())",
    ),
    "userlist_append": ("ul = collections.UserList()\n" + FACTORY.format(held="ul.append", args="x"), "list(ul)"),
    "series_update": (
        "ser = pd.Series([1, 2])\n" + FACTORY.format(held="ser.update", args="pd.Series([9], index=[0])"),
        "ser.tolist()",
    ),
    "partial_out": (FACTORY.format(held="functools.partial(np.add, arr, 1, out=arr)", args=""), "arr.tolist()"),
    "partial_copyto": (FACTORY.format(held="functools.partial(np.copyto, arr, 5.0)", args=""), "arr.tolist()"),
    "methodcaller": (FACTORY.format(held="operator.methodcaller('append', 1)", args="calls"), "calls"),
    "partial_of_a_methodcaller": (
        FACTORY.format(held="functools.partial(operator.methodcaller('append', 1), calls)", args=""),
        "calls",
    ),
    "defaultdict_getitem": (
        "dd = collections.defaultdict(int)\n" + FACTORY.format(held="dd.__getitem__", args="x"),
        "dict(dd)",
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
    "a_library_method_in_python": "note2 = pd.Series([1, 2]).get",
    "an_arrays_reader": "note2 = np.arange(4).searchsorted",
    "a_numpy_scalars_method": "note2 = np.float64(2.0).__mul__",
    "a_struct_reader": "note2 = (lambda u: lambda x: u(b'\\x01\\x00\\x00\\x00'))(struct.Struct('<i').unpack)",
    "a_methodcaller_of_a_reader": "note2 = (lambda m: lambda x: m({1: 2}))(operator.methodcaller('get', 1))",
}


#: A reader held through a lambda's closure is told from a writer by what
#: ran, which only sys.monitoring records: below Python 3.12 it runs again.
_READ_BY_WHAT_RAN = {"a_struct_reader", "a_methodcaller_of_a_reader"}
needs_trace = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="sys.monitoring needs Python 3.12: below it a reader held through a closure runs again",
)


@pytest.mark.parametrize(
    "held", [pytest.param(held, marks=needs_trace if held in _READ_BY_WHAT_RAN else ()) for held in READERS]
)
def test_a_function_holding_only_readers_is_still_served(cash_magics, cash_instance, held):
    """Positive control: the walk does not refuse what only reads."""
    cash_instance.config.call_cost_floor_seconds = 0.0
    log = RunLog()
    # The event's name, not the log: what the log holds changes as it fills.
    cash_magics.shell.user_ns["EVENT"] = log.event
    body = (
        "import sys, struct\n"
        f"def mk():\n    note = functools.partial(sys.audit, EVENT)\n    {READERS[held]}\n"
        "    def g(x):\n        note(x)\n        note2(x)\n        return x\n    return g\nf = mk()"
    )
    cells = [INIT + body, f"y = (f(1), {SLEEP})", "seen = y[0]"]
    assert _run_twice(cash_magics, cells) == [1, 1]
    assert log == [1], "the second run after the restart was served"


#: Held ``arr.__getitem__``: a slice of it is a view of ``arr``, which a
#: stored result is not. Each way round: the array written after the call,
#: and the result written.
WRITTEN_AFTER = {
    "the_array": ("arr[0] = 99.0", "repr(y[0].tolist())"),
    "the_result": ("y[0][...] = 7.0", "repr(arr.tolist())"),
}


@pytest.mark.parametrize("after", WRITTEN_AFTER)
def test_a_held_array_lookup_hands_back_a_view_after_a_restart(cash_magics, cash_instance, after):
    cash_instance.config.call_cost_floor_seconds = 0.0
    write, probe = WRITTEN_AFTER[after]
    body = "def mk():\n    gi = arr.__getitem__\n    def g(x):\n        return gi(slice(0, 2))\n    return g\nf = mk()"
    cells = [INIT + body, f"y = (f(1), {SLEEP})", write, f"seen = {probe}"]
    plain: dict = {}
    for cell in cells:
        exec(cell, plain)
    assert _run_twice(cash_magics, cells) == [plain["seen"]] * 2
