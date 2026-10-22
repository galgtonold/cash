"""``mg = df.merge(other, on='g')`` holds each of ``other``'s dicts in many
rows. The share check read such a cell once per row, with only that row's
slots allowed for it, so each one looked held by something else: the merge,
and every statement or helper call on ``mg`` after it, re-ran every time.
Read once with the slots of all its rows allowed, it is held by the frames
alone: stored, and served still holding ``other``'s own dicts."""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
SETUP = "import time\nimport numpy as np, pandas as pd"
FRAMES = (
    "df = pd.DataFrame({'g': np.arange(1000) % 7, 'o': [{'i': i} for i in range(1000)]})\n"
    "other = pd.DataFrame({'g': np.arange(7), 'lab': [{'g': g} for g in range(7)]})"
)
MERGES = {
    "statement": [f"mg = ({SLOW}, df.merge(other, on='g'))[1]"],
    "call": [f"def mrg(a, b):\n    {SLOW}\n    return a.merge(b, on='g')", "mg = mrg(df, other)"],
    "repeated_column": [f"mg = ({SLOW}, pd.DataFrame({{'lab': other['lab'].tolist() * 3}}))[1]"],
}


@pytest.fixture(autouse=True)
def time_to_find_the_holders(monkeypatch):
    from cash.notebook.statement import processor

    monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 30.0)


def _statuses(cash_magics):
    return [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]


@pytest.mark.parametrize("merge", list(MERGES))
def test_the_result_is_served_holding_the_other_frames_cells(cash_magics, merge):
    cells = [SETUP, FRAMES, *MERGES[merge]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    assert _statuses(cash_magics)[0] == "RESTORED", _statuses(cash_magics)
    ns = cash_magics.shell.user_ns
    lab = ns["other"]["lab"]
    assert all(cell is lab.iloc[cell["g"]] for cell in ns["mg"]["lab"])
    lab.iloc[3]["z"] = 1
    assert {"g": 3, "z": 1} in ns["mg"]["lab"].tolist()


def test_a_call_on_the_first_frame_after_the_merge_is_served(cash_magics):
    """``cf = flt(df)`` shares ``df``'s cells with ``mg``, which holds
    ``other``'s too. The read of the frames sharing ``cf``'s cells listed
    ``other``'s dicts for ``mg``; read again with ``other`` joined, that
    first read's lists counted as one more holder of each, and every such
    call re-ran."""
    cells = [
        SETUP,
        FRAMES,
        "mg = df.merge(other, on='g')",
        f"def flt(d):\n    {SLOW}\n    return d[d.g > 0]",
        "cf = flt(df)",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    assert _statuses(cash_magics) == ["RESTORED"]
    ns = cash_magics.shell.user_ns
    assert ns["cf"]["o"].iloc[0] is ns["df"]["o"].iloc[1]
    ns["df"]["o"].iloc[1]["z"] = 1
    assert ns["cf"]["o"].iloc[0] == {"i": 1, "z": 1}
