"""A statement reading a name the cell's plan binds again runs after it.

Running a cell again whose frame a later loop changed in place, the plan
(``NotebookSimulator.plan_cell_run``) restores ``df = load()`` and skips what
is current by lineage. ``held = {id(df): 'df'}`` below it was skipped too,
and kept the ``id()`` of the frame the restore replaced: plain Python keys
the new frame. Only when ``df``'s statement ran long enough to be stored
(past the store floor), so the one test that met it failed on a slow CI
runner alone.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pd = pytest.importorskip("pandas")

#: ``df``'s statement sleeps past the store floor, as a slow machine's
#: ``pd.DataFrame(...)`` did; the list bound to ``held`` first gives the
#: cell a run the plan can jump.
MAKE = (
    "import time\nimport pandas as pd\nheld = [(i, str(i)) for i in range(1000)]\n"
    f"df = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), pd.DataFrame({{'a': range(300), 'b': 0.0}}))[1]\n"
)
LOOP = "for i in range(300):\n    df.loc[i, 'b'] = df.loc[i, 'a'] * 2"


@pytest.mark.parametrize(
    ("tail", "finds"),
    [
        ("held = {id(df): 'df'}", "held.get(id(df)) == 'df'"),
        ("held = id(df)", "held == id(df)"),
        ("held = hex(id(df))", "held == hex(id(df))"),
    ],
)
def test_what_keeps_the_id_of_a_restored_frame_finds_the_new_one(cash_magics, mock_shell, tail, finds):
    make = MAKE + tail
    cells = [make, LOOP]
    for _ in range(3):
        run_cash_cell(cash_magics, make, cells=cells)
        assert eval(finds, dict(mock_shell.user_ns))
        run_cash_cell(cash_magics, LOOP, cells=cells)
        assert eval(finds, dict(mock_shell.user_ns))


def test_what_reads_no_rebound_name_is_still_skipped(cash_magics, mock_shell):
    """The plan still jumps: the statements above the restore are skipped."""
    make = MAKE + "held = {id(df): 'df'}"
    cells = [make, LOOP]
    run_cash_cell(cash_magics, make, cells=cells)
    run_cash_cell(cash_magics, LOOP, cells=cells)
    run_cash_cell(cash_magics, make, cells=cells)

    statuses = [str(m["status"]) for m in cash_magics._last_cell_metrics["statements"]]
    assert statuses[:3] == ["SKIPPED", "SKIPPED", "SKIPPED"]
    assert statuses[3] == "RESTORED"
    assert mock_shell.user_ns["held"].get(id(mock_shell.user_ns["df"])) == "df"
