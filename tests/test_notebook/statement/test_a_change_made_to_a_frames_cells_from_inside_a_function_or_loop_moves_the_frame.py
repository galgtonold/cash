"""A change made to a frame's object cells moves the frame, whatever code
makes it: a function of the user's handed to ``.map`` or ``.apply``,
``operator.setitem`` in a lambda, a loop over the column, or ``iterrows``.

Each changes the dict in row 3 in place, as in plain Python, and none of
them names ``df`` as what it changes: the statement or call reading
``df['o']`` afterwards was served what it read before the change was
edited, with no warning.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FRAME = "df = pd.DataFrame({'x': np.arange(10.), 'o': [{'i': i} for i in range(10)]})"
SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
CHANGES = {
    "def-in-map-lambda": (
        "def upd(d, z):\n    if d['i'] == 3:\n        d['z'] = z",
        "_ = df['o'].map(lambda d: upd(d, {z}))",
    ),
    "def-to-apply": ("def upd(d):\n    if d['i'] == 3:\n        d['z'] = Z", "Z = {z}\n_ = df['o'].apply(upd)"),
    "def-to-map": ("def upd(d):\n    if d['i'] == 3:\n        d['z'] = Z", "Z = {z}\n_ = df['o'].map(upd)"),
    "setitem-lambda": (
        "import operator",
        "_ = df['o'].map(lambda d: operator.setitem(d, 'z', {z}) if d['i'] == 3 else None)",
    ),
    "for-loop": ("", "for d in df['o']:\n    if d['i'] == 3:\n        d['z'] = {z}"),
    "for-loop-update": ("", "for d in df['o']:\n    if d['i'] == 3:\n        d.update(z={z})"),
    "iterrows": ("", "for _, row in df.iterrows():\n    if row['o']['i'] == 3:\n        row['o']['z'] = {z}"),
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
