"""A frame whose object column holds the very cells an output frame holds is
not stored with it: the entry records where the output holds them, and a
hit takes them back from the live frame.

``w = df[df['n'] % 2 == 0]`` keeps ``df``'s own lists, and a later
``df['t'].iloc[2].append(5)`` shows through ``w``. Stored whole, ``df``
went into the entry beside ``w``: a million-row frame was copied again on
every store and every hit. The frame is current (its lineage is the one the
entry recorded), so its cells are the ones the statement would take again.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cash.notebook.holder_patches import FrameCellsPatch, apply_patch, frame_cells_patch
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"import time\nimport pandas as pd\ndef slow(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x"
CELLS = [
    SETUP,
    "df = pd.DataFrame({'t': [['a', str(i)] for i in range(1000)], 'n': range(1000)})",
    "w = slow(df[df['n'] % 2 == 0])",
]


@pytest.fixture(autouse=True)
def time_to_find_the_holders(monkeypatch):
    """A busy machine may take longer than a quick statement is given to
    find the holders, and the statement is then not stored at all."""
    from cash.notebook.statement import processor

    monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 30.0)


def test_the_entry_holds_where_the_cells_are_not_the_frame(cash_magics, statement_processor):
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    metrics = statement_processor.process_statement(CELLS[2])
    _metadata, payload = cash_magics._cash_instance.backend.get(metrics["cache_key"])
    assert isinstance(payload["variables"]["df"], FrameCellsPatch)


def test_a_hit_holds_the_frames_own_cells(cash_magics):
    ns = cash_magics.shell.user_ns
    for run in ("first", "second"):
        for cell in CELLS:
            run_cash_cell(cash_magics, cell, cells=CELLS)
        assert ns["w"]["t"].iloc[1] is ns["df"]["t"].iloc[2], f"{run} Run All"
        assert ns["w"]["n"].tolist() == list(range(0, 1000, 2))


def test_a_frame_holding_more_of_the_output_than_cells_is_not_patched():
    df = pd.DataFrame({"t": [["a", str(i)] for i in range(10)]})
    rows = df["t"].tolist()
    w = df.iloc[::2].copy()
    # Its cells, and the frame itself in a column: more than cells.
    w["frame"] = [df] * len(w)
    assert frame_cells_patch(df, {"w": w}) is None
    del rows


def test_a_frame_without_the_rows_any_more_fails_the_restore():
    df = pd.DataFrame({"t": [["a", str(i)] for i in range(10)]})
    patch = frame_cells_patch(df, {"w": df.iloc[5:].copy()})
    assert patch is not None
    restored = {"w": df.iloc[5:].copy(deep=True)}
    with pytest.raises(KeyError):
        apply_patch(df.iloc[:3].copy(), patch, restored)
