"""``v = D['k']`` is refused as an alias of ``D``'s item, which keeps the
lineage ``D`` gives it. A ``threading.Timer`` started in a cell below fills
``D`` where cash cannot see it: the statement runs again and finds 2, and
the reader below must not be served what 1 gave.
"""

from __future__ import annotations

import time

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell


def test_a_reader_of_the_alias_sees_what_the_timer_put(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [
        "import threading\nD = {'k': 1}",
        "v = D['k']",
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
