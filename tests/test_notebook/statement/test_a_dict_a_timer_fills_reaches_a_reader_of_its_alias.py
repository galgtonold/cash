"""``v = D['k']`` is refused as an alias of ``D``'s item, which keeps the
lineage ``D`` gives it. A ``threading.Timer`` started in a cell below fills
``D`` where cash cannot see it: the statement runs again and finds 2, and
the reader below must not be served what 1 gave.

The same holds where ``D`` is read through a function or a copy, which a
hit would otherwise serve, and where the thread that filled ``D`` was one
cash never saw start (the console, a library's callback): an alias that
binds another value over the same inputs is keyed by its value.
"""

from __future__ import annotations

import time

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

#: How the statement reads ``D['k']``.
READS = {
    "an alias": "v = D['k']",
    "a function": "def get():\n    return D['k']\nv = get()",
    "a copy": "v = dict(D)['k']",
    "an expression": "v = D['k'] + 0",
}


@pytest.mark.parametrize("read", list(READS))
def test_a_reader_sees_what_the_timer_put(cash_magics, monkeypatch, read):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [
        "import threading\nD = {'k': 1}",
        READS[read],
        "w = [v, 'read']",
        "tm = threading.Timer(0.1, lambda: D.__setitem__('k', 2))\ntm.start()",
    ]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns["tm"].join(5)
    time.sleep(0.05)
    for cell in cells[1:3]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["v"] == 2
    assert ns["w"] == [2, "read"], "the reader was served what the old item gave"


def test_a_change_by_a_thread_cash_never_saw_reaches_a_reader_of_the_alias(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = ["D = {'k': 1}", "v = D['k']", "w = [v, 'read']"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    exec("import threading as _th\n_t = _th.Thread(target=lambda: D.update(k=2))\n_t.start()\n_t.join()", ns)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["v"] == 2
    assert ns["w"] == [2, "read"], "the reader was served what the old item gave"


@pytest.mark.parametrize(
    "work",
    [
        pytest.param("th = threading.Thread(target=lambda: D.__setitem__('k', 2))\nth.start()\nth.join()", id="Thread"),
        pytest.param("tm = threading.Timer(0.01, lambda: D.__setitem__('k', 2))\ntm.start()\ntm.join()", id="Timer"),
    ],
)
def test_a_call_reading_what_a_joined_thread_changed_is_not_served(cash_magics, monkeypatch, work):
    """``z = get()`` ran, but the call ``get()`` inside it was served from
    the call cache: keyed on lineages, and no lineage moved when the
    thread set ``D['k']``, it handed back the 1 that ``v = get()`` stored."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [
        "import threading, time\nD = {'k': 1}\ndef get():\n    time.sleep(0.02)\n    return D['k']",
        "v = get()",
        work,
        "z = get()",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["z"] == 2
