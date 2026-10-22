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

An object that raises for ``__class__`` itself (``Cls``) failed cash's
``isinstance`` checks the same way: ``isinstance`` asks for it when the type
does not match. Those checks go through ``instance_of``, which answers by
the object's real type.
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
class Cls:
    # Raises for __class__ itself: isinstance asks for it when the type
    # does not match, and so does an ABC's check.
    def __init__(self, a):
        object.__setattr__(self, 'a', a)
    def __getattribute__(self, k):
        if k == '__class__':
            raise KeyError(k)
        return object.__getattribute__(self, k)
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
    "hiding its class": ["c = Cls(5)", "w = c.a + 1", "c.a = 9", "w2 = c.a"],
    "hiding its class, in a list": ["cs = [Cls(1), Cls(2)]", "tot = cs[0].a + cs[1].a"],
    # cash checks values against a library's types only once it is loaded.
    "hiding its class, libraries loaded": [
        "import importlib\nfor _m in ('pandas', 'polars', 'pyarrow', 'matplotlib.pyplot', 'scipy.sparse', 'sklearn.base'):\n"
        "    try:\n        importlib.import_module(_m)\n    except ImportError:\n        pass",
        "c = Cls(5)",
        "w = c.a + 1",
        "c.a = 9",
        "w2 = c.a",
    ],
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
    names = {name for name, value in plain.items() if type(value) in (int, float, list) and not name.startswith("_")}
    assert {name: ns[name] for name in names if type(plain[name]) is not list} == {
        name: plain[name] for name in names if type(plain[name]) is not list
    }


#: A class whose ``__getattribute__`` raises for every name but ``v``, in a
#: module: called, mutated, held in a list, a dict, a tuple, a default and a
#: closure. Each cell runs as plain Python, edited or after a restart.
HOSTILE = """
class Hostile:
    def __init__(self, v):
        object.__setattr__(self, 'v', v)
    def __getattribute__(self, k):
        if k != 'v':
            raise KeyError(k)
        return object.__getattribute__(self, k)
    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
def getv(o):
    return o.v
def bump(o):
    o.v += 1
"""
SHAPES = {
    "an attribute as argument": ["c = Hostile(5)", "w = getv(Hostile(c.v))"],
    "the object as argument": ["c = Hostile(5)", "w = getv(c)"],
    "a call that changes it": ["c = Hostile(5)", "bump(c)", "w = c.v"],
    "a loop": ["cs = [Hostile(1), Hostile(2)]", "w = 0\nfor x in cs:\n    w += getv(x)"],
    "a dict holding it": ["c = Hostile(5)", "d = {'k': c}", "w = getv(d['k'])"],
    "a function by another name": ["c = Hostile(5)", "f = getv", "w = f(c)"],
    "a tuple output": ["c = Hostile(5)", "t = (c, c.v)", "w = t[1]"],
    "a default": ["c = Hostile(5)", "def h(o=c):\n    return o.v", "w = h()"],
    "a closure": ["c = Hostile(5)", "def mk():\n    return lambda: c.v", "w = mk()()"],
}


@pytest.mark.parametrize("restart", [False, True], ids=["rerun", "restart"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_an_object_raising_for_every_lookup_runs_as_plain_python(cash_magics, monkeypatch, tmp_path, shape, restart):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    (tmp_path / "hostile_mod.py").write_text(HOSTILE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    cells = ["from hostile_mod import Hostile, getv, bump", *SHAPES[shape]]
    plain = _plain(cells)
    ns = cash_magics.shell.user_ns
    for run in range(2):
        if run and restart:
            cash_magics.tracking_state.reset_session_state()
            cash_magics._upstream_checker.simulator.cache.reset()
            for name in plain:
                ns.pop(name, None)
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["w"] == plain["w"]


COUNTED = """
import traceback
CALLS = {}
ASKED_CLASS = []
class Counted:
    def __init__(self, v):
        object.__setattr__(self, 'v', v)
    def __getattribute__(self, k):
        CALLS[k] = CALLS.get(k, 0) + 1
        if k == '__class__':
            ASKED_CLASS.append(''.join(traceback.format_stack(limit=8)[:-1]))
        return object.__getattribute__(self, k)
"""


def test_cash_asks_such_an_object_nothing_but_to_pickle_it(cash_magics, tmp_path, monkeypatch):
    """Its ``__getattribute__`` is the user's code: cash's own questions
    (``isinstance`` asking for ``__class__``, ``fit``, ``__code__``) read
    its real type and Python's generic lookup instead. Storing it pickles
    it, and pickle asks."""
    (tmp_path / "counted_mod.py").write_text(COUNTED, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    cells = [
        "from counted_mod import Counted, CALLS, ASKED_CLASS",
        "w = 0",
        "c = Counted(5)",
        "w = c.v + 1",
        "w2 = c.v",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    pickling = {"__class__", "__reduce_ex__", "__reduce__", "__getstate__"}
    asked = {name: n for name, n in cash_magics.shell.user_ns["CALLS"].items() if name not in pickling | {"v"}}
    assert asked == {}
    asked_class = cash_magics.shell.user_ns["ASKED_CLASS"]
    assert len(asked_class) <= 1, "\n----\n".join(asked_class)  # the store's one pickle
