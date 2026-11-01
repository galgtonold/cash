"""A seed below a draw, run on an earlier pass, does not key the draw.

Restart and Run All is what a cached notebook must agree with. There a draw
from a global stream that no cell above seeds reads an unseeded stream: it
is keyed as an unseeded draw and served frozen. In the same kernel, a seed
in a cell below (``np.random.seed(0)`` under ``gs.fit(X, y)`` in mlbook
ch08) had left the stream seeded when Run All came round to the draw again,
so the draw was keyed on that seed: it ran again on every pass, and the
entry it stored was one a restart never looked up. The stream now counts as
unseeded for a cell no cell above seeds it in, as after a restart. A seed
above the draw still keys it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash.tracking.randomness import rng_virtual_var
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

NP = rng_virtual_var("numpy.random")


def _setup(log: str) -> str:
    return (
        "import os, time\nimport numpy as np\n"
        "def make(n):\n"
        f"    fd = os.open({log!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return np.random.rand(n)"
    )


@pytest.fixture
def notebook(cash_magics, mock_shell, tmp_path):
    log = tmp_path / "draws.log"

    def run_all(cells: list[str]) -> list[str]:
        statuses = []
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
            statuses += [str(m.get("status")) for m in cash_magics.cash_status("dict")["last_cell"]["statements"]]
        return statuses

    run_all.setup = _setup(str(log))
    run_all.draws = lambda: len(log.read_bytes()) if log.exists() else 0
    run_all.ns = mock_shell.user_ns
    run_all.lineage = cash_magics._cell_executor.tracking_state.variable_lineage
    return run_all


@pytest.mark.parametrize("draw", ["X = make(3)", "X = make(3) + np.random.rand(3) * 0"])
def test_a_draw_with_a_seed_only_below_is_served_on_the_next_pass(notebook, draw):
    cells = [notebook.setup, draw, "np.random.seed(0)\nnoise = np.random.rand(2)"]
    notebook(cells)
    first = notebook.ns["X"].tolist()
    drawn = notebook.draws()
    notebook(cells)
    assert notebook.draws() == drawn, "the draw ran again on the second pass"
    assert notebook.ns["X"].tolist() == first


def test_the_stream_takes_no_lineage_from_a_seed_below(notebook):
    cells = [notebook.setup, "y = 1", "np.random.seed(0)"]
    notebook(cells)
    assert NP in notebook.lineage
    notebook(cells[:2])
    assert NP not in notebook.lineage


def test_a_seed_below_still_seeds_the_draws_below_it(notebook):
    cells = [notebook.setup, "X = make(3)", "np.random.seed(0)", "Y = make(2)"]
    notebook(cells)
    notebook(cells)
    np.random.seed(0)
    assert notebook.ns["Y"].tolist() == np.random.rand(2).tolist()


def test_a_seed_above_keys_the_draw(notebook):
    cells = [notebook.setup, "np.random.seed(0)", "X = make(3)"]
    notebook(cells)
    drawn = notebook.draws()
    assert notebook(cells)[-1] == "RESTORED"
    assert notebook.draws() == drawn
    reseeded = [notebook.setup, "np.random.seed(1)", "X = make(3)"]
    notebook(reseeded)
    np.random.seed(1)
    assert notebook.ns["X"].tolist() == np.random.rand(3).tolist()
