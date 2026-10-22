"""A file a magic's line read in Python changes, and only a cell below is
run: the kernel keeps the old value, as a plain one would, and says so
(NOTEBOOK-MAGIC-STALE). The magic's files are in its cell's snapshot, so the
check redoes that cell. Before, it warned only when the check redid the cell
for another reason. A cell of magics alone runs without cash, which does not
see what it read."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

CELLS = {
    "py_after": "x = %sx echo {len(open('f').read())}\nq = len(x)",
    "py_before": "x = 0\nx = %sx echo {len(open('f').read())}",
    "listing": "q = 0\nx = !echo {len(os.listdir('d'))}",
}


@pytest.mark.parametrize("cell", list(CELLS))
def test_a_cell_below_run_alone_warns(nb_runner, cell):
    work = nb_runner.work_dir
    (work / "f").write_text("abc")
    (work / "d").mkdir()
    (work / "d" / "1").write_text("")
    nb_runner.create_notebook(["import os", CELLS[cell], "print('X', x)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "NOTEBOOK-MAGIC-STALE" not in nb_runner.get_raw_output(3)
    (work / "f").write_text("abcdef")
    (work / "d" / "2").write_text("")
    nb_runner.run_cells([3])
    shown = nb_runner.get_raw_output(3)
    assert "NOTEBOOK-MAGIC-STALE" in shown, shown
    assert "X ['3']" in shown or "X ['1']" in shown, shown
