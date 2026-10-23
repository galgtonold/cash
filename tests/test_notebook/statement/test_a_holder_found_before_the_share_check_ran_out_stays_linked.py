"""A variable the share check found holding an output's objects stays
linked when the check runs out of its budget on the next round.

``h.map = {i: r for i, r in enumerate(recs)}``: the first round finds
``recs`` holding the records ``h`` holds, and the round after it, with
``recs`` joined, ran out of its budget in a real kernel some runs and not
others. Then the search up from ``h`` stood in for it, which finds no
variable sharing ``h``'s parts, and ``recs`` was dropped: a change to a
record through ``df`` moved ``recs`` but not ``h``, and a reader of
``h.map`` was served the record from before the change, with no warning.
"""

from __future__ import annotations

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import WalkBudgetExceeded
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
MADE = """recs = [{'id': i, 'meta': {'k': i}} for i in range(20)]
df = pd.DataFrame({'id': range(20), 'rec': recs})
class H:
    pass
h = H()
h.map = HOLD
def setk(d):
    if d['id'] == 3:
        d['meta']['k'] = Z"""
HOLDS = {
    "dict": "{i: r for i, r in enumerate(recs)}",
    "list": "list(recs)",
}


@pytest.fixture
def second_round_runs_out(monkeypatch):
    """The round of the check with ``recs`` joined runs out of its budget."""
    check = shared_objects._check_group

    def checked(group, *args, **kwargs):
        if {"h", "recs"} <= set(group):
            raise WalkBudgetExceeded
        return check(group, *args, **kwargs)

    monkeypatch.setattr(shared_objects, "_check_group", checked)


@pytest.mark.parametrize("hold", list(HOLDS))
def test_the_reader_after_a_change_through_the_frame_sees_it(cash_magics, second_round_runs_out, hold):
    for z in (1, 2, 1):
        cells = [
            "import time\nimport pandas as pd",
            MADE.replace("HOLD", HOLDS[hold]),
            f"Z = {z}",
            f"_ = ({SLOW}, df['rec'].apply(setk))[1]",
            f"g = ({SLOW}, str(h.map[3]))[1]",
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["g"] == str({"id": 3, "meta": {"k": z}}), z
