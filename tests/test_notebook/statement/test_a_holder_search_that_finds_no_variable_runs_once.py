"""A share check whose search up `gc.get_referrers` finds no variable holding
the outputs' objects is not asked again with anchors.

A refused check is asked a second time with anchors: variables the
statement left as they were, bound again to their live objects on a
restore. An anchor is a variable the search found; when it found none, the
second check takes the same steps to the same refusal, and each climb reads
the whole heap -- 0.1-0.5 s a statement in a session with a few hundred
thousand objects.
"""

from __future__ import annotations

from cash.notebook import shared_objects
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

CELLS = [
    "import time\nimport pandas as pd",
    "items = [[1], [2], [3]]\na = pd.DataFrame({'o': items, 'n': [0, 1, 2]})\nb = pd.DataFrame({'o': items})\ndel items",
    f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\ndf2 = a[a.n >= 0]",
]


def test_the_climb_runs_once(cash_magics, monkeypatch):
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    climbs = []
    real = shared_objects._holders_from_referrers

    def counted(*args, **kwargs):
        found = real(*args, **kwargs)
        climbs.append(found)
        return found

    monkeypatch.setattr(shared_objects, "_holders_from_referrers", counted)
    run_cash_cell(cash_magics, CELLS[2], cells=CELLS)
    assert climbs and not any(climbs), climbs
    assert len(climbs) == 1
