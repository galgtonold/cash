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
