"""A value that finds another object by its identity -- a weak reference
to it, or its ``id()`` -- is not restored on its own: restored, it pointed
at the object as it was stored, while the statement binding that object
bound a new one, and ``wr() is p.child`` gave False, ``reg.get(id(p.seen))``
None, where plain Python finds the object. Such a statement runs again."""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

MAKE = "import weakref\nclass Q:\n    pass\nclass P:\n    def __init__(self):\n        self.seen = []\n        self.child = Q()\np = P()"
HOLDERS = {
    "a weak reference": ("wr = weakref.ref(p.child)", "r = wr() is p.child"),
    "a WeakValueDictionary": ("d = weakref.WeakValueDictionary()\nd['c'] = p.child", "r = d.get('c') is p.child"),
    "a weak reference in an object": (
        "class H:\n    pass\nh = H()\nh.ref = weakref.ref(p.child)",
        "r = h.ref() is p.child",
    ),
    "an id key": ("reg = {id(p.seen): 'seen'}", "r = reg.get(id(p.seen)) == 'seen'"),
    "an id in a list": ("ids = [id(p.child)]", "r = ids[0] == id(p.child)"),
    "an id alone": ("k = id(p)", "r = k == id(p)"),
    "its own id": ("p.me = id(p)", "r = p.me == id(p)"),
    "an id in a deque": ("from collections import deque\nids = deque([id(p.child)])", "r = ids[0] == id(p.child)"),
    "an id key in a dict subclass": (
        "from collections import OrderedDict\nreg = OrderedDict({id(p.seen): 'seen'})",
        "r = reg.get(id(p.seen)) == 'seen'",
    ),
    "an id in a SimpleNamespace": (
        "from types import SimpleNamespace\nns = SimpleNamespace(k=id(p.child))",
        "r = ns.k == id(p.child)",
    ),
    "a weak reference in a SimpleNamespace": (
        "from types import SimpleNamespace\nh = SimpleNamespace(wr=weakref.ref(p.child))",
        "r = h.wr() is p.child",
    ),
    "an id in an integer array": (
        "import numpy as np\narr = np.array([id(p.child)], dtype=np.int64)",
        "r = int(arr[0]) == id(p.child)",
    ),
    "an id in a slots class": (
        "class H:\n    __slots__ = ('k',)\n    def __init__(self, k):\n        self.k = k\nh = H(id(p.child))",
        "r = h.k == id(p.child)",
    ),
    "an id of a part of a SimpleNamespace": (
        "from types import SimpleNamespace\nns2 = SimpleNamespace(child=[1])\nreg = {id(ns2.child): 'c'}",
        "r = reg.get(id(ns2.child)) == 'c'",
    ),
    "an id inside a big container of records": (
        "big = [(j, j) for j in range(20000)] + [('c', id(p.child))]",
        "r = big[-1][1] == id(p.child)",
    ),
    "an id on a list subclass": (
        "class C(list):\n    pass\nc = C([1])\nc.meta = id(p.child)",
        "r = c.meta == id(p.child)",
    ),
    "a weak reference on a dict subclass": (
        "class C(dict):\n    pass\nc = C(a=1)\nc.meta = weakref.ref(p.child)",
        "r = c.meta() is p.child",
    ),
    "an id in a named tuple": (
        "from typing import NamedTuple\nclass T(NamedTuple):\n    k: int\nt = T(id(p.child))",
        "r = t.k == id(p.child)",
    ),
    "an id in a tuple subclass": ("class T(tuple):\n    pass\nt = T([id(p.child)])", "r = t[0] == id(p.child)"),
    "a weak reference subclass": (
        "class R(weakref.ref):\n    pass\nr0 = [R(p.child)]",
        "r = r0[0]() is p.child",
    ),
    "an id inside a big list of objects": (
        "class R:\n    pass\nbig = [R() for _ in range(12000)]\nbig[-1].k = id(p.child)",
        "r = big[-1].k == id(p.child)",
    ),
}


def _stored_however_cheap(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.mark.parametrize("holder", list(HOLDERS))
def test_a_run_all_finds_the_new_object(cash_magics, cash_instance, holder):
    _stored_however_cheap(cash_instance)
    hold, read = HOLDERS[holder]
    cells = [MAKE, hold, read + "  # @cash:no-cache"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] is True


@pytest.mark.parametrize(
    "value",
    [
        "k = 12345",
        "reg = {'n': len(p.seen)}",
        "stamps = [1_700_000_000_000 + 8 * i for i in range(50)]",
        "class C(dict):\n    pass\nreg = C(n=len(p.seen))",
        "ns = [1_700_000_000_000_000_000 + 8 * i for i in range(50)]",
        "big = [(j, j * 8) for j in range(20000)]",
        "import numpy as np\narr = np.arange(0, 800_000, 8)",
    ],
)
def test_a_value_not_finding_an_object_by_identity_is_restored(cash_magics, cash_instance, value):
    _stored_however_cheap(cash_instance)
    cells = [MAKE, value]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert "RESTORED" in statuses[-1], statuses


RAISING = (
    "class D(dict):\n"
    "    def keys(self):\n        raise RuntimeError('keys')\n"
    "    def values(self):\n        raise RuntimeError('values')\n"
    "    def items(self):\n        raise RuntimeError('items')\n"
    "    def __iter__(self):\n        raise RuntimeError('iter')\n"
    "    def __len__(self):\n        raise RuntimeError('len')\n"
    "class L(list):\n"
    "    def __iter__(self):\n        raise RuntimeError('iter')\n"
    "    def __len__(self):\n        raise RuntimeError('len')\n"
    "class S(set):\n"
    "    def __iter__(self):\n        raise RuntimeError('iter')\n"
    "class R(weakref.ref):\n"
    "    def __call__(self):\n        raise RuntimeError('call')\n"
)


@pytest.mark.parametrize(
    ("hold", "read"),
    [
        ("reg = D(k=id(p.child))", "r = dict.__getitem__(reg, 'k') == id(p.child)"),
        ("reg = L([id(p.child)])", "r = list.__getitem__(reg, 0) == id(p.child)"),
        ("reg = S([id(p.child)])", "r = id(p.child) in set(set.__iter__(reg))"),
        ("reg = [R(p.child)]", "r = weakref.ref.__call__(reg[0]) is p.child"),
    ],
    ids=["a dict subclass", "a list subclass", "a set subclass", "a weak reference subclass"],
)
def test_a_holder_whose_own_methods_raise_is_read_without_them(cash_magics, cash_instance, hold, read):
    """The check reads a container subclass through the builtin type's
    methods and a weak reference through ``weakref.ref``'s own call, never
    the class's: plain Python runs none of them here, and an error in one
    failed the cell."""
    _stored_however_cheap(cash_instance)
    cells = [MAKE + "\n" + RAISING, hold, read + "  # @cash:no-cache"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] is True


@pytest.mark.parametrize(
    ("holder", "name", "kind"),
    [("a weak reference", "wr", "a weak reference"), ("an id key", "reg", "an id()")],
)
def test_it_runs_again_with_one_warning_naming_the_variable(cash_magics, cash_instance, holder, name, kind):
    _stored_however_cheap(cash_instance)
    hold, _read = HOLDERS[holder]
    cells = [MAKE, hold]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
    held = [str(w.message) for w in caught if "NOTEBOOK-IDENTITY-HELD" in str(w.message)]
    assert len(held) == 1, held
    assert f"'{name}' holds {kind}" in held[0]
