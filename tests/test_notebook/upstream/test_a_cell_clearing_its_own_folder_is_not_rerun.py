"""A cell that clears its output folder and redraws it is not re-run by an
unrelated cell below it.

``for old in OUT.glob('*.png'): old.unlink()`` opens no file, so the cell's
run left no record that the loop had run, and the planner took it for a
writer never run this session: every later cell, even ``x = 41 + 1``,
re-ran the deletion and the redraw first.
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

CLEARS = {
    # A path the planner can resolve: the loop variable is an entry of OUT.
    "glob": "for old in OUT.glob('*.txt'):\n    old.unlink()",
    # One it cannot: only the record of the run vouches for it.
    "listdir": "for name in os.listdir(OUT):\n    os.remove(os.path.join(OUT, name))",
}


@pytest.mark.parametrize("clear", sorted(CLEARS))
def test_an_unrelated_cell_leaves_the_folder_alone(cash_magics, tmp_path, clear):
    out = tmp_path / "charts"
    cells = [
        f"import os\nfrom pathlib import Path\nOUT = Path(r'{out}')\nOUT.mkdir(exist_ok=True)\n"
        + CLEARS[clear]
        + "\nfor k in range(3):\n    (OUT / f'c{k}.txt').write_text(str(k))",
        "x = 41 + 1",
    ]
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(cash_magics, cells[0], cells=cells)
        run_cash_cell(cash_magics, cells[1], cells=cells)
        # Not the cell's own file: a re-run of the chart cell deletes it.
        (out / "kept.txt").write_text("kept")
        for _ in range(3):
            run_cash_cell(cash_magics, cells[1], cells=cells)

    assert (out / "kept.txt").exists(), "the unrelated cell re-ran the cell that clears the folder"
    assert sorted(p.name for p in out.iterdir()) == ["c0.txt", "c1.txt", "c2.txt", "kept.txt"]


@pytest.mark.parametrize("CLEAR", sorted(CLEARS))
def test_an_edited_clearing_cell_still_reruns_for_a_cell_that_reads_the_folder(cash_magics, tmp_path, CLEAR):
    """Control: the record says the cell ran, not that its code is current. An
    edit to the cell re-runs it for a cell that reads what it writes."""
    out = tmp_path / "charts"
    chart = (
        f"import os\nfrom pathlib import Path\nOUT = Path(r'{out}')\nOUT.mkdir(exist_ok=True)\n"
        + CLEARS[CLEAR]
        + "\nfor k in range(N):\n    (OUT / f'c{k}.txt').write_text(str(k))"
    )
    reader = "names = sorted(os.listdir(OUT))"
    cells = ["N = 2", chart, reader]
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for code in cells:
            run_cash_cell(cash_magics, code, cells=cells)
        assert cash_magics.shell.user_ns["names"] == ["c0.txt", "c1.txt"]
        cells[0] = "N = 3"
        run_cash_cell(cash_magics, cells[2], cells=cells)

    assert cash_magics.shell.user_ns["names"] == ["c0.txt", "c1.txt", "c2.txt"]
