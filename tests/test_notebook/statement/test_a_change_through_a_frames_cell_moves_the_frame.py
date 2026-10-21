"""A change made through a variable bound to a frame's cell moves the frame.

``l = df['o'].iloc[1]`` and then ``l.append(9)`` changes the list ``df``
holds in row 1, as in plain Python. The share check links ``l`` with the
variables holding its object, but an object column is a numpy array, which
tells the garbage collector nothing of what it holds: no holder was found,
``df`` kept its lineage, and a filter on the cells' length was served from
before the change. The frames' object columns are read by address now.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FRAME = "df = pd.DataFrame({'o': [[i] for i in range(10)], 'n': range(10)})"
SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
BODIES = {
    "call": [f"def mk(d):\n    {SLOW}\n    return d[d['o'].map(len) == 1]", "w = mk(df)"],
    "statement": [f"w = ({SLOW}, df[df['o'].map(len) == 1])[1]"],
}


@pytest.mark.parametrize("kind", list(BODIES))
def test_the_filter_after_the_change_is_not_the_one_before(cash_magics, kind):
    body = BODIES[kind]
    cells = [
        "import time\nimport pandas as pd",
        FRAME,
        *body,
        "l = df['o'].iloc[1]",
        "l.append(9)",
        body[-1] + "  # again",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert len(ns["w"]) == 9
    assert ns["w"]["o"].iloc[0] is ns["df"]["o"].iloc[0]
