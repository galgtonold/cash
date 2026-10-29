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


@pytest.mark.parametrize("value", ["cfg = {'size': 65536, 'name': 'a'}", "x = 1 << 20"])
def test_a_round_number_beside_more_objects_than_the_walk_reads_is_restored(cash_magics, cash_instance, value):
    """65536 and 1 << 20 are multiples of 8 as an id() is, but below any
    address the process uses; and with more objects in the notebook than
    the look for one reads, a bare number is still not taken for an id()."""
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
