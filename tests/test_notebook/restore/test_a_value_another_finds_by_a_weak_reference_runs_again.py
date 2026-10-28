"""A value another variable finds by a weak reference is not restored: the
restore binds a copy, the object the reference points at goes, and the
reference finds nothing where plain Python finds the value.

``box = Box(BIG)`` keeping ``weakref.ref(BIG)``, then ``BIG[0].id = -5``
restored on the next Run All: ``box.h()`` gave None, and ``box.total()``
raised ``TypeError: 'NoneType' object is not iterable``. Such a statement
runs again, and NOTEBOOK-IDENTITY-HELD says so."""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

MAKE = "import weakref\nclass Row:\n    def __init__(self, i):\n        self.id = i\nclass Bag(list):\n    pass\nBIG = Bag(Row(i) for i in range(50))"
HOLDERS = {
    "a weak reference in an object": (
        "class Box:\n    def __init__(self, rows):\n        self.h = weakref.ref(rows)\nbox = Box(BIG)",
        "r = sum(row.id for row in box.h())",
    ),
    "a weak reference": ("wr = weakref.ref(BIG)", "r = sum(row.id for row in wr())"),
    "a WeakValueDictionary of one of its items": (
        "d = weakref.WeakValueDictionary()\nd['first'] = BIG[0]",
        "r = d['first'].id + len(BIG)",
    ),
}


@pytest.mark.parametrize("holder", list(HOLDERS))
def test_a_run_all_reads_the_value_through_the_reference(cash_magics, cash_instance, holder):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    hold, read = HOLDERS[holder]
    cells = [MAKE, hold, "BIG[0].id = -5", read + "  # @cash:no-cache"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
            assert cash_magics.shell.user_ns["r"] == (sum(range(1, 50)) - 5 if "sum" in read else -5 + 50)
    assert any("NOTEBOOK-IDENTITY-HELD" in str(w.message) for w in caught)


def test_a_value_only_cash_refers_to_weakly_is_restored(cash_magics, cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [MAKE, "BIG[0].id = -5"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses == ["RESTORED"], statuses
