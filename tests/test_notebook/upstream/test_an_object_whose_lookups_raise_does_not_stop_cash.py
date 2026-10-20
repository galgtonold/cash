"""An object whose ``__getattr__`` raises something other than
AttributeError does not make cash fail.

A record class reading its fields from a dict raises KeyError for a name it
does not hold. cash's own questions about such an object -- is it an
estimator (``fit``), a function (``__code__``), one of cash's wrappers
(``_cash_cached``) -- went through ``getattr(obj, name, default)``, whose
default covers AttributeError only, and the cell failed with KeyError where
plain Python runs it. A copy that runs the object's own lookups
(``copy.deepcopy`` asks for ``__deepcopy__``) failed the store the same way;
now the value is not stored, and STORE-FAILED says so.

Every such question in the notebook check goes through ``attribute_or``
(`test_attribute_probes_cannot_raise`): one asked with plain ``getattr``
(``writable``, asked of each variable a cell run alone starts from) failed
the cell again with UpstreamStateError.
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


RECORDS = """__name__ = '__main__'
class Rec:
    def __init__(self, **kw):
        self.__dict__['_d'] = dict(kw)
    def __getattr__(self, k):
        return self._d[k]
class Loud(Rec):
    def __getattr__(self, k):
        if k in self._d:
            return self._d[k]
        raise RuntimeError(k)
"""


@pytest.mark.parametrize("cls", ["Rec", "Loud"])
def test_a_cell_run_alone_over_such_an_object_starts_from_the_cells_above(cash_magics, monkeypatch, cls):
    """Run alone, ``s = r.a`` starts from the ``r`` the cells above give,
    though a cell below changed it: the check of what the cell starts from
    asks ``r`` questions of its own."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [RECORDS, f"r = {cls}(a=1)", "s = r.a", "r._d['a'] = 5"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.shell.user_ns["s"] == _plain(cells[:3])["s"] == 1


#: Where such an object reaches code outside the notebook check: hashing an
#: output that holds it, an argument, a loop, a function default.
FLOWS = {
    "in an output": ["rs = [Rec(a=i) for i in range(3)]", "t = sum(r.a for r in rs)"],
    "an argument": ["def g(r):\n    return r.a + 1", "x = g(Rec(a=4))", "y = x * 2"],
    "a dict's value": ["d = {'k': Rec(a=2)}", "z = d['k'].a"],
    "a loop": ["out = []\nfor r in [Rec(a=1), Rec(a=2)]:\n    out.append(r.a)", "n = len(out)"],
    "a default": ["def h(r=Rec(a=3)):\n    return r.a", "q = h()"],
}


@pytest.mark.parametrize("flow", list(FLOWS))
def test_such_an_object_anywhere_in_a_notebook_runs_as_plain_python(cash_magics, monkeypatch, flow):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [RECORDS, *FLOWS[flow]]
    plain = _plain(cells)
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    names = {
        name for name, value in plain.items() if isinstance(value, (int, float, list)) and not name.startswith("_")
    }
    assert {name: ns[name] for name in names if not isinstance(plain[name], list)} == {
        name: plain[name] for name in names if not isinstance(plain[name], list)
    }
