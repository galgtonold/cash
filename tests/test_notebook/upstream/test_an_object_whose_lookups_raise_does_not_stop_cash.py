"""An object whose ``__getattr__`` raises something other than
AttributeError does not make cash fail.

A record class reading its fields from a dict raises KeyError for a name it
does not hold. cash's own questions about such an object -- is it an
estimator (``fit``), a function (``__code__``), one of cash's wrappers
(``_cash_cached``) -- went through ``getattr(obj, name, default)``, whose
default covers AttributeError only, and the cell failed with KeyError where
plain Python runs it. A copy that runs the object's own lookups
(``copy.deepcopy`` asks for ``__deepcopy__``) failed the store the same way;
now the value is not stored.
"""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

CLASSES = """__name__ = '__main__'
import numpy as np
class Record:
    def __init__(self, arr):
        object.__setattr__(self, "arr", arr)
    def __getattr__(self, name):
        raise KeyError(name)
class Loud(Record):
    def __getattr__(self, name):
        raise RuntimeError("no " + name)
"""


def _plain(cells: list[str]) -> dict:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


@pytest.mark.parametrize("cls", ["Record", "Loud"])
@pytest.mark.parametrize("held", ["{c}(np.zeros(10))", "[{c}(np.zeros(10))]"], ids=["bound", "in-a-list"])
def test_a_notebook_over_such_an_object_runs_as_plain_python(cash_magics, monkeypatch, cls, held):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    reach = "h" if held.startswith("{c}") else "h[0]"
    ns = cash_magics.shell.user_ns
    for value in (3, 4):
        cells = [
            CLASSES,
            "h = " + held.format(c=cls),
            f"v = {reach}.arr[2:5]",
            f"v[:] = {value}",
            f"r = float({reach}.arr.sum())",
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["r"] == _plain(cells)["r"] == 3.0 * value
