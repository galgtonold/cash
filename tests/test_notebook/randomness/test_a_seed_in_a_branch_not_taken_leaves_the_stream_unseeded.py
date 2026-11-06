"""A seed in a branch not taken leaves the stream unseeded for the draws below.

mlbook ch08 shuffles its frame under a pandas version check whose ``else``
seeds the global stream, draws in ``gs.fit(X, y)`` below it and seeds the
stream further down. After an edit between the check and the fit, the fit
ran again and was not stored: the cell on disk keeps the indent of a blank
line in the block, which IPython empties before it runs a cell, so the cell
never counted as run and its ``else`` seed counted as one above; and the
upstream check read the ``if`` as seeding whichever branch ran. A run from
the top has the stream unseeded at the fit, so the fit is keyed and stored
as an unseeded draw, and the edited cells run again serve it.
"""

from __future__ import annotations

import pytest

from cash.tracking.randomness import rng_virtual_var
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

NP = rng_virtual_var("numpy.random")

IF_CELLS = {
    "plain": "if flag:\n    w = 1\nelse:\n    np.random.seed(0)\n    w = 2",
    "indented_blank_line": "if flag:\n    w = 1\n    \nelse:\n    np.random.seed(0)\n    w = 2",
}


def _as_ipython_runs(cell: str) -> str:
    """*cell* as IPython runs it: every whitespace-only line emptied."""
    return "\n".join("" if line.isspace() else line for line in cell.split("\n"))


def _setup(log: str) -> str:
    return (
        "import os, time\nimport numpy as np\nflag = True\n"
        "def make(n):\n"
        f"    fd = os.open({log!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return np.random.rand(n)"
    )


@pytest.mark.parametrize("if_cell", sorted(IF_CELLS))
def test_a_draw_run_again_after_an_edit_above_is_stored(cash_magics, mock_shell, tmp_path, if_cell):
    log = tmp_path / "draws.log"

    def draws() -> int:
        return len(log.read_bytes()) if log.exists() else 0

    cells = [
        _setup(str(log)),
        IF_CELLS[if_cell],
        "k = 10",
        "X = make(3) + k",
        "np.random.seed(0)\nz = np.random.rand()",
    ]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, _as_ipython_runs(cell), cells=cells)
    assert draws() == 1

    cells[2] = "k = 8"
    lineage = cash_magics.tracking_state.variable_lineage
    for _ in range(2):
        for cell in cells[2:]:
            run_cash_cell(cash_magics, _as_ipython_runs(cell), cells=cells)
            if cell.startswith("X ="):
                assert NP not in lineage, "the draw read the stream the seed below left"
    assert draws() == 2, "the draw under the edit was not stored"
    assert min(mock_shell.user_ns["X"]) >= 8
