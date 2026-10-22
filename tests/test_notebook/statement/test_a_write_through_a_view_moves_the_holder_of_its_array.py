"""A write through a numpy view moves on the variable that holds the array it
sits on, whatever the array is held in.

``w = slow(data['x'][2:5])`` then ``w[:] = 5`` writes into ``data['x']``. The
write moved the lineage of a variable bound to the array itself (``a``), but
not of one holding it in a dict, list, tuple or attribute: ``data`` kept its
lineage, so after an edit to the write (``w[:] = 7``) and Run All,
``r = slow(data['x'].sum())`` was served the 15 of the first run instead of
21. Each case edits the write, runs all again, and must match plain Python.
"""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

SETUP = (
    "import time, types\n"
    "import numpy as np\n"
    "class Lib:\n"
    "    def __init__(self, x):\n"
    "        self.x = x\n"
    "Lib.__module__ = 'somelib'\n"
    "def slow(x):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return x"
)

#: (cell binding the holder, the view taken of the array in it, the read).
HOLDERS = {
    "named": ("a = np.zeros(10)", "a[2:5]", "a"),
    "dict": ("data = {'x': np.zeros(10)}", "data['x'][2:5]", "data['x']"),
    "list": ("data = [np.zeros(10)]", "data[0][2:5]", "data[0]"),
    "tuple": ("data = (np.zeros(10), 1)", "data[0][2:5]", "data[0]"),
    "attribute": ("data = types.SimpleNamespace(x=np.zeros(10))", "data.x[2:5]", "data.x"),
    "nested": ("data = {'k': [np.zeros(10)]}", "data['k'][0][2:5]", "data['k'][0]"),
    "view of a view": ("data = {'x': np.zeros(10)}", "data['x'][1:8][1:4]", "data['x']"),
    # An object of a library's class, which the share check reads as one
    # closed value: read down as the garbage collector sees it.
    "library object": ("data = Lib(np.zeros(10))", "data.x[2:5]", "data.x"),
}


def _cells(bind: str, view: str, read: str, value: int) -> list[str]:
    return [SETUP, bind, f"w = slow({view})", f"w[:] = {value}", f"r = slow(float({read}.sum()))"]


def _plain(cells: list[str]) -> float:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns["r"]


@pytest.mark.parametrize("holder", list(HOLDERS))
def test_an_edited_write_through_the_view_is_read_back(cash_magics, monkeypatch, holder):
    # A loaded machine stores the cheap steps too.
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    ns = cash_magics.shell.user_ns
    for value in (5, 7, 5):
        cells = _cells(*HOLDERS[holder], value)
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["r"] == _plain(cells) == 3 * value


def test_a_column_series_held_in_a_dict_reads_what_was_written(cash_magics, monkeypatch):
    """Under copy-on-write (pandas 3) a write to the Series leaves the frame
    as it was; before it, the Series is a view of the frame's block and the
    write lands in ``data``. Either way the read gives what plain Python
    gives: on pandas 2 it was served the first run's 50 for 70."""
    pytest.importorskip("pandas")
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    ns = cash_magics.shell.user_ns
    for value in (5, 7):
        cells = [
            SETUP + "\nimport pandas as pd",
            "data = {'df': pd.DataFrame({'c': np.zeros(10)})}",
            "s = slow(data['df']['c'])",
            f"s[:] = {value}",
            "r = slow(float(data['df']['c'].sum()))",
            "t = slow(float(s.sum()))",
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (ns["r"], ns["t"]) == (_plain(cells), 10.0 * value)


def test_a_holder_the_statement_does_not_read_moves_too(cash_magics, monkeypatch):
    """``cache`` got the array from a function, as a side effect: no link
    shows it holds ``a``, and ``v = a[2:5]`` does not read it. Walking only
    what the statement read missed it, and ``r`` was served 21 for 27."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [
        "import numpy as np\ncache = {}\ndef f(x):\n    cache['k'] = x\n    return x",
        "a = f(np.zeros(10))",
        "v = a[2:5]",
        "v[:] = 7",
        "r = float(cache['k'].sum())",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[3] = "v[:] = 9"
    for cell in cells[3:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["r"] == _plain(cells) == 27.0


def test_a_weak_value_dictionary_holding_the_array_moves_too(cash_magics, monkeypatch):
    """The collector sees a weak reference hold nothing, yet ``reg['k']``
    reads the array: ``r`` was served 21 for 27."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [
        "import numpy as np, weakref\nreg = weakref.WeakValueDictionary()\ndef f(x):\n    reg['k'] = x\n    return x",
        "a = f(np.zeros(10))",
        "v = a[2:5]",
        "v[:] = 7",
        "r = float(reg['k'].sum())",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[3] = "v[:] = 9"
    for cell in cells[3:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["r"] == 27.0


def test_an_object_two_variables_hold_is_walked_once(monkeypatch):
    """Every variable is walked for the array, in one walk: what two of
    them hold is read once."""
    from cash.notebook import shared_objects

    a = np.arange(10.0)
    shared = [[i] for i in range(100)] + [a]
    view = a[2:5]
    walked = []
    real = shared_objects.children_of

    def counting(value):
        walked.append(id(value))
        return real(value)

    monkeypatch.setattr(shared_objects, "children_of", counting)
    ns = {"a": a, "one": {"s": shared}, "two": {"s": shared}, "view": view}
    assert shared_objects.view_base_names([view], ns) == {"a", "one", "two", "view"}
    assert walked.count(id(shared)) == 1


def test_a_list_nested_deeper_than_python_recurses_is_no_error(cash_magics):
    cell = (
        "import numpy as np\nbig = []\ncur = big\nfor _ in range(20_000):\n"
        "    nxt = []\n    cur.append(nxt)\n    cur = nxt\ncur.append(np.zeros(3))"
    )
    run_cash_cell(cash_magics, cell, cells=[cell])
    run_cash_cell(cash_magics, "v = cur[0][1:]", cells=[cell, "v = cur[0][1:]"])
    assert cash_magics.shell.user_ns["v"].shape == (2,)
