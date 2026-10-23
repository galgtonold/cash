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


def test_the_change_reaches_the_reader(watched_notebook, monkeypatch):  # noqa: F811 - the fixture
    asyncio.run(asyncio.sleep(0))  # ``_asyncio`` loaded, as in a kernel
    monkeypatch.setattr(shared_objects, "_REFERRERS_IN", None)
    monkeypatch.setattr(shared_objects, "_TRAVERSE_LOOPS", True)
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
