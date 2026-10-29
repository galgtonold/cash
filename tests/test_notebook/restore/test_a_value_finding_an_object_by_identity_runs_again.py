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
    "an id four levels down in each of 20,000 dicts": (
        "regs = [{'a': {'b': {'pid': id(p.seen)}}} for _ in range(20_000)]",
        "r = regs[0]['a']['b']['pid'] == id(p.seen)",
    ),
    "an id in each of 20,000 records": (
        "class R:\n    def __init__(self, pid):\n        self.pid = pid\nregs = [R(id(p.seen)) for _ in range(20_000)]",
        "r = regs[0].pid == id(p.seen)",
    ),
    "an id nested past what is sampled": (
        "regs = [[[[[[[[[[id(p.seen)]]]]]]]]] for _ in range(20_000)]",
        "r = regs[0][0][0][0][0][0][0][0][0][0] == id(p.seen)",
    ),
    "an object DataFrame column of ids": (
        "import pandas as pd\ndf = pd.DataFrame({'pid': pd.Series([id(p.seen)], dtype=object), 'v': [1]})",
        "r = bool(df['pid'][0] == id(p.seen))",
    ),
    "a nullable DataFrame column of ids": (
        "import pandas as pd\ndf = pd.DataFrame({'pid': pd.array([id(p.seen), None], dtype='Int64'), 'v': [1, 2]})",
        "r = bool(df['pid'][0] == id(p.seen))",
    ),
    "a DataFrame column of ids past 1,024 integer columns": (
        "import pandas as pd\ndf = pd.DataFrame({**{f'c{i}': [i] for i in range(1024)}, 'pid': [id(p.seen)]})",
        "r = bool(df['pid'][0] == id(p.seen))",
    ),
    "a DataFrame column of ids": (
        "import pandas as pd\ndf = pd.DataFrame({'pid': [id(p.seen)], 'v': [1]})",
        "r = bool(df['pid'][0] == id(p.seen))",
    ),
    "a Series indexed by ids": (
        "import pandas as pd\nser = pd.Series(['seen'], index=[id(p.seen)])",
        "r = ser.get(id(p.seen)) == 'seen'",
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
    objects in the notebook than the look for one reads."""
    _stored_however_cheap(cash_instance)
    cells = ["data = [[{'k': i} for i in range(9000)] for _ in range(25)]", value]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert "RESTORED" in statuses[-1], statuses
    assert not [w for w in caught if "IDENTITY-HELD" in str(w.message)]


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
