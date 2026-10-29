"""A change made by code that keeps a variable's part elsewhere reaches the
statement reading the variable, also where the read of the heap for what
holds it cannot answer.

On Python 3.12.0-3.12.5 without the C extension, with ``_asyncio`` loaded,
every read of the heap answers "not known" (`shared_objects._TRAVERSE_LOOPS`),
as one out of time does: the notebook still runs, and ``Pt.bump_held(5)``
changing ``data`` in place is seen by the statement reading it.
"""

from __future__ import annotations

import asyncio
import gc

import pytest

from cash.exceptions import CashWarning
from cash.notebook import shared_objects
from tests._cell_driver import run_cash_cell
from tests.test_notebook.values.test_a_watched_variable_changed_in_place_is_seen import (
    CHANGES,
    POKE,
    READ,
    SETUP,
    _expected,
    watched_notebook,  # noqa: F401 - the fixture
)


def _unreadable(monkeypatch) -> None:
    asyncio.run(asyncio.sleep(0))  # ``_asyncio`` loaded, as in a kernel
    monkeypatch.setattr(shared_objects, "_REFERRERS_IN", None)
    monkeypatch.setattr(shared_objects, "_TRAVERSE_LOOPS", True)
    monkeypatch.setattr(shared_objects, "_UNREADABLE", [False])


def test_the_change_reaches_the_reader(watched_notebook, monkeypatch):  # noqa: F811 - the fixture
    _unreadable(monkeypatch)
    magics = watched_notebook
    ns = magics.shell.user_ns
    change = CHANGES["by code holding it elsewhere"]
    unchanged = [*SETUP, POKE, READ]
    cells = [*SETUP, POKE, *change, READ]
    for cell in SETUP:
        run_cash_cell(magics, cell, cells=unchanged)
    for _ in range(3):
        run_cash_cell(magics, POKE, cells=unchanged)
        run_cash_cell(magics, READ, cells=unchanged)
    before = ns["result"]
    assert before == _expected(ns)
    for cell in change:
        run_cash_cell(magics, cell, cells=cells)
    run_cash_cell(magics, READ, cells=cells)
    assert ns["result"] == _expected(ns) != before


def test_a_holder_search_cut_short_warns(watched_notebook, monkeypatch):  # noqa: F811 - the fixture
    """Out of time, or not known, the search for what keeps a variable
    handed to code (``Pt.hold(data['meta'])``) left the rest unlinked
    silently. Now the statement is not stored, and says so."""
    from cash.notebook.statement import processor

    real = processor.share_group
    searched = []

    def cut_short(names, values, user_ns, *args, **kwargs):
        if values is user_ns:  # the search from the handed variables up
            searched.append(names)
            raise shared_objects.WalkBudgetExceeded
        return real(names, values, user_ns, *args, **kwargs)

    monkeypatch.setattr(processor, "share_group", cut_short)
    magics = watched_notebook
    hold = CHANGES["by code holding it elsewhere"][0]
    cells = [*SETUP, hold]
    for cell in SETUP:
        run_cash_cell(magics, cell, cells=cells)
    with pytest.warns(CashWarning, match="NOTEBOOK-SHARE-UNCHECKED"):
        run_cash_cell(magics, hold, cells=cells)
    assert searched == [["data"]]
    run_cash_cell(magics, hold, cells=cells)
    assert str(magics._last_cell_metrics["statements"][-1]["status"]) == "COMPUTED"


def test_the_search_for_what_holds_a_value_does_not_read_gc_directly(monkeypatch):
    """``gc.get_referrers`` reads the module's traverse too, and looped as
    gc's other reads do: the search for a value's holders answers "not
    known", as one out of time does."""
    _unreadable(monkeypatch)

    def looped(*objs):
        raise AssertionError("gc.get_referrers read the heap")

    monkeypatch.setattr(gc, "get_referrers", looped)
    target = [1]
    holder = [target]
    with pytest.raises(shared_objects.WalkBudgetExceeded):
        shared_objects._holders_from_referrers([target], set(), {"holder": holder}, set(), shared_objects.GaveUp())
    assert shared_objects.heap_unreadable()


def test_the_warning_names_the_python_not_a_time(watched_notebook, monkeypatch):  # noqa: F811 - the fixture
    """Not known is no matter of time: the warning and the reason say what
    cannot be read, and on which Python."""
    _unreadable(monkeypatch)
    magics = watched_notebook
    hold = CHANGES["by code holding it elsewhere"][0]
    cells = [*SETUP, hold]
    for cell in SETUP:
        run_cash_cell(magics, cell, cells=cells)
    with pytest.warns(CashWarning, match="NOTEBOOK-SHARE-UNCHECKED") as caught:
        run_cash_cell(magics, hold, cells=cells)
    said = " ".join(str(w.message) for w in caught)
    assert "with asyncio loaded cannot read what holds an object" in said, said
    assert "within" not in said, said
    from cash.notebook.statement import processor

    why = processor._cut_short_why("finding what keeps the variables it hands to code", 2.0)
    assert "cannot be done on Python" in why and "longer than" not in why, why


def test_whether_functions_or_objects_exist_is_a_yes_when_the_heap_cannot_be_read(monkeypatch):
    """The checks for a notebook function, or an object of a class of the
    user's, read ``gc.get_referrers`` too: not known, their answer is yes,
    as for a heap ``gc.freeze`` hid."""
    from cash.notebook import held_functions

    _unreadable(monkeypatch)

    def looped(*objs):
        raise AssertionError("gc.get_referrers read the heap")

    monkeypatch.setattr(gc, "get_referrers", looped)
    monkeypatch.setattr(held_functions, "_NO_FUNCTIONS", {})
    assert held_functions.notebook_functions_exist({"x": 1}) is True


class _NoObjectMade:
    pass


def test_whether_an_object_of_a_users_class_exists_is_a_yes_when_the_heap_cannot_be_read(monkeypatch):
    from cash.notebook import held_functions

    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [_NoObjectMade])
    monkeypatch.setattr(held_functions, "_users_module_functions", list)
    monkeypatch.setattr(held_functions, "_OBJECTS_SEEN", [None])
    assert held_functions.module_class_objects_exist() is False  # none made, read as it is
    _unreadable(monkeypatch)

    def looped(*objs):
        raise AssertionError("gc.get_referrers read the heap")

    monkeypatch.setattr(gc, "get_referrers", looped)
    assert held_functions.module_class_objects_exist() is True
