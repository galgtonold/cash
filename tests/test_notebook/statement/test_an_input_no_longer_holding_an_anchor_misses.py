"""A statement entry bound to an anchor is a miss when an input that held
the anchor's object holds it no more.

``x = [1]`` and ``df`` holding ``x`` in both rows: ``w = mk(df)`` returns
row 1, ``x`` itself, and the entry binds ``w``'s cell to the live ``x`` (an
anchor). A new ``df`` of two equal lists has the same key, and its hit
bound ``w``'s cell to ``x`` as well, where plain Python's ``w`` holds the new
frame's own list: a change through ``df['o'].iloc[1]`` did not show in
``w``. The entry records which anchors each input it left as it was holds,
and a hit asks it again.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SEL1 = f"def mk(d):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return d[d.n == 1]"
TARGETS = {"statement": ("w = mk(df)", "w"), "call in a dict": ("w = {'r': mk(df)}", "w['r']")}


@pytest.mark.parametrize("kind", list(TARGETS))
def test_the_new_frames_own_list_is_the_result(cash_magics, kind):
    line, result = TARGETS[kind]
    cells = [
        "import time\nimport pandas as pd",
        "x = [1]\ndf = pd.DataFrame({'o': [x, x], 'n': [0, 1]})",
        SEL1,
        line,
        "df = pd.DataFrame({'o': [[1], [1]], 'n': [0, 1]})",
        line + "  # again",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    w = eval(result, {}, ns)
    assert w["o"].iloc[0] is ns["df"]["o"].iloc[1]
    assert w["o"].iloc[0] is not ns["x"]
