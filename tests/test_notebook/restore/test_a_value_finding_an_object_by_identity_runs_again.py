"""A value that finds another object by its identity -- a weak reference
to it, or its ``id()`` -- is not restored on its own: restored, it pointed
at the object as it was stored, while the statement binding that object
bound a new one, and ``wr() is p.child`` gave False, ``reg.get(id(p.seen))``
None, where plain Python finds the object. Such a statement runs again."""

from __future__ import annotations

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
