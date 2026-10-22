"""A change made through a list or dict holding a frame's cells moves the
frame.

``L = list(df['o'])`` holds the dicts ``df`` holds, and ``L[3]['z'] = 2``
changes the one in row 3, as in plain Python. Edited from ``z = 1``, the
change left ``df`` at its lineage from before, and a statement or call
reading ``df['o']`` was served what it read then. So was it through
``df['o'].tolist()``, a dict holding one cell, a ``map`` changing the cells,
and a variable bound to one cell once a list held the cells too.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FRAME = "df = pd.DataFrame({'x': np.arange(10.), 'o': [{'i': i} for i in range(10)]})"
SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
CHANGES = {
    "list": ("L = list(df['o'])", "L[3]['z'] = {z}"),
    "tolist": ("L = df['o'].tolist()", "L[3]['z'] = {z}"),
    "dict": ("D = {'k': df['o'].iloc[3]}", "D['k']['z'] = {z}"),
    "map": ("", "_ = df['o'].map(lambda d: d.update(z={z}) if d['i'] == 3 else None)"),
    "cell-and-list": ("L = list(df['o'])\nc3 = df['o'].iloc[3]", "c3['z'] = {z}"),
}
READERS = {
    "statement": [f"g = ({SLOW}, df['o'].map(str).tolist())[1]"],
    "call": [f"def strs(d):\n    {SLOW}\n    return d['o'].map(str).tolist()", "g = strs(df)"],
}


@pytest.mark.parametrize("reader", list(READERS))
@pytest.mark.parametrize("change", list(CHANGES))
def test_the_read_after_an_edited_change_is_not_the_one_before(cash_magics, change, reader):
    hold, edit = CHANGES[change]
    for z in (1, 2):
        cells = [
            "import time\nimport numpy as np, pandas as pd",
            f"{FRAME}\n{hold}",
            edit.format(z=z),
            *READERS[reader],
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        ns = cash_magics.shell.user_ns
        assert ns["df"]["o"].iloc[3] == {"i": 3, "z": z}
        assert ns["g"][3] == str({"i": 3, "z": z}), (z, ns["g"][3])
