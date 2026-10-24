"""A closure whose cell a call rebinds away from a list no longer moves
with that list.

``draw, rebind = make(BIG)`` links both closures to ``BIG``; ``rebind([0])``
rebinds the cell they share, and binds no name, so the edges stayed: a
change to ``BIG`` below moved ``draw`` on, and running ``y = draw(1)`` again
rebuilt ``draw`` from the cell above without the ``rebind`` after it, which
gave 2000 where the cells above give 1. Whether ``draw`` got the edge at all
followed the hash seed (seeds 1 and 2 here).
"""

from __future__ import annotations

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.core

CELLS = [
    "import time\nBIG = [{'id': i} for i in range(2000)]\n"
    "def make(rows):\n    def draw(k):\n        time.sleep(0.2)\n        return len(rows)\n"
    "    def rebind(r):\n        nonlocal rows\n        rows = r\n    return draw, rebind\n"
    "draw, rebind = make(BIG)",
    "rebind([0])",
    "y = draw(1)",
]


@pytest.mark.parametrize("change", ["BIG.append({'id': -1})", "BIG[0]['id'] = -1"])
def test_a_change_to_the_list_does_not_run_the_reader_again(cash_magics, change):
    cells = [*CELLS, change]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.shell.user_ns["y"] == 1
    status = cash_magics.cash_status("dict")["last_cell"]["statements"][-1].get("status")
    assert status == CacheStatus.RESTORED, status
