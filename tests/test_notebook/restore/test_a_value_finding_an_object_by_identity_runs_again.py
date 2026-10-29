"""A value that finds another object by its identity -- a weak reference
to it, or its ``id()`` -- is not restored on its own: restored, it pointed
at the object as it was stored, while the statement binding that object
bound a new one, and ``wr() is p.child`` gave False, ``reg.get(id(p.seen))``
None, where plain Python finds the object. Such a statement runs again."""

from __future__ import annotations

import sys
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
    "an id key in a dict too big to look inside": (
        "reg = {id(p.seen): 'seen', **{i: i for i in range(20_000)}}",
        "r = reg.get(id(p.seen)) == 'seen'",
    ),
    "an id in each of 20,000 dicts": (
        "regs = [{'pid': id(p.seen)} for _ in range(20_000)]",
        "r = regs[0]['pid'] == id(p.seen)",
    ),
    "a DataFrame column of ids": (
        "import pandas as pd\ndf = pd.DataFrame({'pid': [id(p.seen)], 'v': [1]})",
        "r = bool(df['pid'][0] == id(p.seen))",
    ),
    "a Series indexed by ids": (
        "import pandas as pd\nser = pd.Series(['seen'], index=[id(p.seen)])",
        "r = ser.get(id(p.seen)) == 'seen'",
    ),
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
    "an id of a part of an item of a big list": (
        "objs = [P() for _ in range(12000)]\nreg = {id(objs[7].child): 'c'}",
        "r = reg.get(id(objs[7].child)) == 'c'",
    ),
    "an id in a pandas column of objects": (
        "import pandas as pd\ndf = pd.DataFrame({'ref': [id(p.child)]}, dtype=object)",
        "r = df['ref'][0] == id(p.child)",
    ),
    "an id in a pandas series of integers": (
        "import pandas as pd\ns = pd.Series([id(p.child)])",
        "r = int(s[0]) == id(p.child)",
    ),
    "a weak reference in a pandas series": (
        "import pandas as pd\ns = pd.Series([weakref.ref(p.child)], dtype=object)",
        "r = s[0]() is p.child",
    ),
    "an id in a partial": ("import functools\nf = functools.partial(max, id(p.child))", "r = f(0) == id(p.child)"),
    "an id in an argparse namespace": (
        "import argparse\nns = argparse.Namespace(k=id(p.child))",
        "r = ns.k == id(p.child)",
    ),
    "a deque in an argparse namespace": (
        "import argparse\nfrom collections import deque\nns = argparse.Namespace(q=deque([id(p.child)]))",
        "r = ns.q[0] == id(p.child)",
    ),
    "an id in an array.array": ("import array\na = array.array('q', [id(p.child)])", "r = a[0] == id(p.child)"),
    "an id in a structured array": (
        "import numpy as np\narr = np.array([(1, id(p.child))], dtype=[('a', 'i8'), ('b', 'i8')])",
        "r = int(arr['b'][0]) == id(p.child)",
    ),
    "an id as hex text": ("reg = {hex(id(p.child)): 'c'}", "r = reg.get(hex(id(p.child))) == 'c'"),
    "an id as decimal text": ("key = str(id(p.child))", "r = key == str(id(p.child))"),
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


BIG = "data = [[{'k': i} for i in range(9000)] for _ in range(25)]"


@pytest.mark.parametrize(
    "target",
    ["P()", "{'k': 1}", "'x' * 40 + str(len(data))"],
    ids=["an object", "a dict the collector does not track", "a str"],
)
@pytest.mark.parametrize(
    ("hold", "read"),
    [
        ("key = id(holder['a']['b'][0])", "r = key == id(holder['a']['b'][0])"),
        ("reg = {'k': id(holder['a']['b'][0])}", "r = reg['k'] == id(holder['a']['b'][0])"),
    ],
    ids=["bare", "in a dict"],
)
def test_an_id_beside_more_objects_than_the_walk_reads_runs_again(cash_magics, cash_instance, target, hold, read):
    """The look for the object stops short in a big notebook. The number
    was then ruled out when no object the collector tracks had it as its
    id, which a dict of plain values or a str never is; and a bare one was
    let through whatever it matched. Both were served stale and silently."""
    _stored_however_cheap(cash_instance)
    cells = [BIG, "class P:\n    pass\nholder = {'a': {'b': [" + target + "]}}", hold, read + "  # @cash:no-cache"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
            assert cash_magics.shell.user_ns["r"] is True
    assert [w for w in caught if "IDENTITY-HELD" in str(w.message)]


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
        "import pandas as pd\ndf = pd.DataFrame({'a': range(5), 'b': ['x', 'y', 'z', 'u', 'v']})",
        "import argparse\nns = argparse.Namespace(k=3, name='run')",
        "words = ['a word', '1234', 'deadbeef', '0x12']",
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


@pytest.mark.skipif(sys.platform == "win32", reason="no lowest mapped address is read on Windows")
@pytest.mark.parametrize("value", ["cfg = {'size': 65536, 'name': 'a'}", "x = 1 << 20"])
def test_a_round_number_beside_more_objects_than_the_walk_reads_is_restored(cash_magics, cash_instance, value):
    """65536 and 1 << 20 are multiples of 8 as an id() is, but below any
    address the process has mapped, so not taken for one even with more
    objects in the notebook than the look for one reads. (``data`` itself,
    too big to look through, runs again with its own warning.)"""
    _stored_however_cheap(cash_instance)
    cells = ["data = [[{'k': i} for i in range(9000)] for _ in range(25)]", value]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert "RESTORED" in statuses[-1], statuses
    name = value.split(" =", 1)[0]
    assert not [w for w in caught if "IDENTITY-HELD" in str(w.message) and f"'{name}'" in str(w.message)]


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


@pytest.mark.parametrize(
    "held",
    ["numpy.True_", "()", "bool"],
)
def test_the_id_of_an_object_compiled_into_a_c_image_is_found(held):
    """Such an object keeps only a pointer's alignment, not the heap's 16
    bytes: ``np.True_`` (``(arr >= 0).all()``) sits 8 bytes off on 64-bit
    builds. Its ``id()`` was not looked for, and a registry keyed by it
    was restored with an address a restarted process does not reuse."""
    import numpy

    from cash.notebook.statement.identity_holders import identity_held_reason

    ok = eval(held, {"numpy": numpy})
    assert identity_held_reason("seen", {id(ok): "ok"}, {"ok": ok}) is not None


def test_a_number_the_notebook_is_too_big_to_look_through_for_is_not_called_an_id():
    """Past the walk's budget the number is not known to be an ``id()``:
    the warning says cash could not look, not that the value holds one."""
    from cash.notebook.statement.identity_holders import identity_held_reason

    lost = object()
    held = identity_held_reason("reg", {id(lost): "x"}, {"rows": [[i] for i in range(250_000)]})
    assert held is not None
    assert held.kind == "a container cash cannot read"
    assert "more objects than cash looks through" in held
