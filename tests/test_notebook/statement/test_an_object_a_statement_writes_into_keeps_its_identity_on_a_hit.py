"""A statement that writes into a variable keeps its object on a hit.

``arr[3] = 7`` changes the array ``arr`` is bound to; plain Python keeps
that object. A hit that bound ``arr`` to the stored copy put a new object
there, so ``reg = {id(arr): 'a'}`` or ``weakref.ref(arr)``, taken in a cell
run after the entry was stored, no longer found it: ``reg.get(id(arr))``
gave None where plain Python gives ``'a'``. Nothing checked at the hit,
because nothing held the array by identity when the entry was stored.

A hit now writes the stored state into the live object when it can (an
array, a frame or Series, a list, dict or set of plain values, a frame
it added or dropped columns of), so the statement stays served; any other
object is bound to the copy only while the notebook holds neither it nor
an object it holds by identity, and the statement runs otherwise.
"""

from __future__ import annotations

import weakref

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

MAKE = (
    "import weakref\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "class Box:\n"
    "    pass\n"
    "arr = np.zeros(10)\n"
    "d = {'a': 1}\n"
    "lst = [1, 2]\n"
    "df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})\n"
    "df['c'] = 0.0\n"
    "o = Box()\n"
    "o.child = [1]\n"
)

#: (statement, how the later cell holds the object, what reads it back, served?)
CASES = [
    ("arr[3] = 7", "reg = {id(arr): 'a'}", "r = reg.get(id(arr)) == 'a' and arr[3] == 7", True),
    ("arr[3] = 7", "w = weakref.ref(arr)", "r = w() is arr and arr[3] == 7", True),
    (
        "for i in range(10):\n    arr[i] = i + 1",
        "reg = {id(arr): 'a'}",
        "r = reg.get(id(arr)) == 'a' and arr[9] == 10",
        True,
    ),
    ("arr += 1", "reg = {id(arr): 'a'}", "r = reg.get(id(arr)) == 'a' and arr[0] == 1", True),
    ("d['k'] = 7", "reg = {id(d): 'a'}", "r = reg.get(id(d)) == 'a' and d['k'] == 7", True),
    ("lst[0] = 7", "ids = [id(lst)]", "r = ids[0] == id(lst) and lst[0] == 7", True),
    ("df.loc[1, 'a'] = 9.0", "reg = {id(df): 'a'}", "r = reg.get(id(df)) == 'a' and df.loc[1, 'a'] == 9.0", True),
    (
        "df['b'] = df.a * 2",
        "reg = {id(df): 'a'}",
        "r = reg.get(id(df)) == 'a' and list(df.columns) == ['a', 'c', 'b'] and df.b[2] == 6.0",
        True,
    ),
    ("o.x = 5", "reg = {id(o): 'a'}", "r = reg.get(id(o)) == 'a' and o.x == 5", False),
    ("o.x = 5", "reg = {id(o.child): 'a'}", "r = reg.get(id(o.child)) == 'a' and o.x == 5", False),
]


def _statuses(magics) -> dict[str, str]:
    return {
        m.get("code"): m.get("status") for m in magics.cash_status("dict")["last_cell"]["statements"] if m.get("code")
    }


@pytest.mark.parametrize(("body", "take", "check", "served"), CASES)
def test_an_id_or_weak_reference_taken_after_the_store_still_finds_the_object(
    cash_magics, cash_instance, body, take, check, served
):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    ns = cash_magics.shell.user_ns
    cells = [MAKE, body]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    cells = [MAKE, take, body, check]
    var = take.split("(")[-1].split(")")[0].split(".")[0]
    for _ in range(2):
        for cell in cells:
            if cell == body:
                # Weakly, where the kind allows it, so the test holds nothing
                # itself; it rules out a copy given the freed object's id.
                held = weakref.ref(ns[var]) if type(ns[var]).__weakrefoffset__ else None
            run_cash_cell(cash_magics, cell, cells=cells)
            if cell == body:
                statuses = set(_statuses(cash_magics).values())
                assert held is None or held() is ns[var]
        assert bool(ns["r"]) is True
    # Where the state can be written into the object, the entry is still served.
    assert (statuses == {CacheStatus.RESTORED}) is served, statuses


def test_a_copy_is_still_bound_when_nothing_holds_the_object_by_identity(cash_magics, cash_instance):
    """The positive control: with no id() or weak reference taken, an object
    no restore goes into is bound to the stored copy, served as before."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [MAKE, "o.x = 5"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert _statuses(cash_magics).get("o.x = 5") == CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["o"].x == 5
