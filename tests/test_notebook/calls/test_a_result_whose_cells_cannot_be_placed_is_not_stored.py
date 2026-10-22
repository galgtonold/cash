"""A statement whose outputs hold cells of a frame that cash cannot record
where to put back is not stored, and says so (NOTEBOOK-CELLS-SHARED).

``f = flt(df)`` with ``flt`` returning ``df[mask]``: ``f``'s rows are
``df``'s own lists, and a hit puts them back as ``df``'s by where they sit
(`cell_links`). Above the number of cells it records, or with one list in
two rows of ``df``, it cannot say. Above the number, the statement was
stored with copies of the lists, and ``df['o'].iloc[25].append(1)`` after a
hit did not show in ``f``; it runs every time instead, and says so.
"""

from __future__ import annotations

import warnings

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from cash.notebook import frame_cells
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FRAMES = {
    "too many cells": "df = pd.DataFrame({'x': np.arange(40), 'o': [[i] for i in range(40)]})",
    "one list in two rows": (
        "_cells = [[i] for i in range(40)]\n_cells[30] = _cells[31]\n"
        "df = pd.DataFrame({'x': np.arange(40), 'o': _cells})\ndel _cells"
    ),
}


@pytest.mark.parametrize("frame", list(FRAMES))
def test_a_change_through_the_frame_reaches_the_result_on_every_run(cash_magics, monkeypatch, frame):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    if frame == "too many cells":
        monkeypatch.setattr(frame_cells, "_MAX_LINKED", 10)
    cells = [
        "import time\nimport numpy as np, pandas as pd",
        FRAMES[frame],
        f"def flt(d):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return d[d.x > 5]",
        "f = flt(df)",
        "df['o'].iloc[25].append(1)",
    ]
    ns = cash_magics.shell.user_ns
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["f"]["o"].loc[25] is ns["df"]["o"].loc[25]
        assert ns["f"]["o"].loc[25] == [25, 1]
        if frame == "too many cells":
            assert any("NOTEBOOK-CELLS-SHARED" in str(w.message) for w in caught)
