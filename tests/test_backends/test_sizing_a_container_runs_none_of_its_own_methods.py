"""Sizing a container of a class of the user's runs none of its methods.

A dict, list or set subclass whose ``__len__`` or ``values()`` prints and
counts showed ``CALLED __len__`` in the cell that bound it and in the one
that read it, and its counter went up, where plain Python never asked:
cash sized it (for the RAM tier and the restore budget) through the
class's own methods. It is read by the builtin type's methods now.
"""

from __future__ import annotations

import pytest

from cash.notebook.upstream.changed_below import _held_directly
from cash.sizing import estimate_object_size, memory_footprint

CALLS: list[str] = []


def _counting(base: type) -> type:
    def method(name):
        def run(self, *args, **kwargs):
            CALLS.append(name)
            return getattr(base, name)(self, *args, **kwargs)

        return run

    names = ["__iter__", "__len__", "__contains__", "__getitem__", "__reversed__", "copy"]
    if base is dict:
        names += ["keys", "values", "items", "get"]
    return type(f"Counting{base.__name__.title()}", (base,), {n: method(n) for n in names if hasattr(base, n)})


def _filled(base: type):
    made = _counting(base)()
    if base is dict:
        for k in range(5):
            dict.__setitem__(made, k, [k])
    elif base is list:
        for k in range(5):
            list.append(made, [k])
    else:
        for k in range(5):
            set.add(made, (k,))
    return made


@pytest.mark.parametrize("base", [dict, list, set])
def test_sizing_and_the_look_below_call_nothing_of_the_class(base):
    value = _filled(base)
    CALLS.clear()
    assert estimate_object_size(value) > 0
    assert memory_footprint(value) > 0
    held, whole = _held_directly(value)
    assert whole and len(held) == 5
    assert CALLS == []


class _Elsewhere(dict):
    """Keeps its items in an attribute; its builtin storage stays empty."""

    def __init__(self):
        super().__init__()
        self._d = {}

    def __setitem__(self, key, value):
        self._d[key] = value

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)

    def values(self):
        return self._d.values()


class _Extra(list):
    __slots__ = ("extra",)


def test_a_container_keeping_its_items_in_its_attributes_is_sized_by_them():
    """Sized by its builtin storage alone, a dict subclass holding two
    8 MB arrays in ``self._d`` read as 80 bytes, and the look below it saw
    nothing and called the walk complete."""
    np = pytest.importorskip("numpy")
    value = _Elsewhere()
    value["a"] = np.zeros(1_000_000)
    value["b"] = np.ones(1_000_000)
    assert estimate_object_size(value) > 15_000_000
    assert memory_footprint(value) > 15_000_000
    held, _whole = _held_directly(value)
    assert any(item is value._d for item in held)
    slotted = _Extra()
    slotted.extra = np.zeros(1_000_000)
    assert estimate_object_size(slotted) > 7_000_000
    assert memory_footprint(slotted) > 7_000_000


@pytest.mark.parametrize("base", [dict, list])
def test_attributes_sharing_a_child_are_sized_in_bounded_time(base):
    """Six levels of a subclass, each with 100 attributes all naming the
    next: summing every attribute at each level took 40 s a call."""
    import time

    kind = type("Shared", (base,), {})
    nxt = kind()
    for _level in range(6):
        cur = kind()
        for i in range(100):
            setattr(cur, f"a{i}", nxt)
        nxt = cur
    start = time.perf_counter()
    estimate_object_size(cur)
    assert time.perf_counter() - start < 1.0


def test_an_attribute_that_is_also_an_item_is_counted_once():
    """``self[key] = self.key = value`` (an EasyDict) holds each value once."""

    class EasyDict(dict):
        pass

    big = list(range(100_000))
    d = EasyDict()
    dict.__setitem__(d, "rows", big)
    d.rows = big
    alone = estimate_object_size({"rows": big})
    assert estimate_object_size(d) < 1.5 * alone


def test_the_look_below_reads_a_subclass_s_slots():
    class Slotted(list):
        __slots__ = ("extra",)

    held = [1]
    value = Slotted([[0]])
    value.extra = held
    items, _whole = _held_directly(value)
    assert any(item is held for item in items)


def _counting_cell(base: str, names: list[str], slots: str = "") -> str:
    body = "".join(
        f"    def {m}(self, *a, **k):\n        CALLS.append('{m}')\n        return {base}.{m}(self, *a, **k)\n"
        for m in names
    )
    return f"import collections, time\nCALLS = []\nclass C({base}):\n{slots}{body}"


_SEQ = ["__iter__", "__len__", "__getitem__", "__contains__", "__eq__"]
_MAP = [*_SEQ, "keys", "values", "items", "get", "copy"]


@pytest.mark.parametrize(
    ("cls", "make", "read"),
    [
        (_counting_cell("tuple", _SEQ), "c = C([[1], 2])", "tuple.__getitem__(c, 1)"),
        (_counting_cell("frozenset", _SEQ), "c = C([1, 2])", "frozenset.__len__(c)"),
        (
            _counting_cell("collections.Counter", _MAP),
            "c = C()\ndict.__setitem__(c, 'k', 3)",
            "dict.__getitem__(c, 'k')",
        ),
        (
            _counting_cell("list", _SEQ, "    __slots__ = ('extra',)\n"),
            "c = C()\nlist.append(c, [1])\nc.extra = [2]",
            "(list.__len__(c), c.extra)",
        ),
    ],
    ids=["tuple", "frozenset", "Counter", "list with slots"],
)
def test_storing_and_restoring_a_container_runs_none_of_its_methods(cash_magics, cash_instance, cls, make, read):
    """Copied and pickled by the class's own route, a tuple, frozenset,
    Counter or slotted list subclass had its ``__iter__``, ``__len__``,
    ``keys`` or ``__getitem__`` run as cash stored it, restored it and
    put it back into its holder."""
    from tests._cell_driver import run_cash_cell

    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [cls, "time.sleep(0.05)\n" + make, f"seen = ({read}, len(CALLS))  # @cash:no-cache"]
    ns = cash_magics.shell.user_ns
    statuses = []
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
            if cell is cells[1]:
                statuses.append([str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]])
        assert ns["seen"][1] == 0, ns["CALLS"]
        assert type(ns["c"]).__name__ == "C"
    assert "RESTORED" in str(statuses[-1][0]), statuses  # ``c = C(...)``, stored and restored


@pytest.mark.parametrize(
    ("cls", "make", "read"),
    [
        (
            "class C(list):\n    __slots__ = 'tag'\n    def append(self, x):\n        list.append(self, x)",
            "c = C(range(5))\nc.tag = 'T'",
            "(list(c), getattr(c, 'tag', 'LOST'))",
        ),
        (
            "class C(dict):\n    __slots__ = ('__x',)\n    def keys(self):\n        return dict.keys(self)\n"
            "    def setx(self, v):\n        self.__x = v",
            "c = C(a=1)\nc.setx(5)",
            "(dict(c), getattr(c, '_C__x', 'LOST'))",
        ),
    ],
    ids=["a slot named by one string", "a private slot"],
)
def test_a_container_subclass_keeps_its_slots_through_a_restore(cash_magics, cash_instance, cls, make, read):
    """``__slots__ = 'tag'`` is one slot, not three, and ``'__x'`` is stored
    as ``_C__x``: read by name from ``__slots__``, both came back unset."""
    from tests._cell_driver import run_cash_cell

    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = ["import time\n" + cls, "time.sleep(0.05)\n" + make, f"seen = {read}  # @cash:no-cache"]
    ns = cash_magics.shell.user_ns
    expected = None
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        expected = expected or ns["seen"]
        assert ns["seen"] == expected
    assert "LOST" not in ns["seen"]
