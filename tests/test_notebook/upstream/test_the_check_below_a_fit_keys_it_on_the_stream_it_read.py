"""The check below a fit keys it on the random stream the fit read.

A fit that draws from numpy's global stream unseen (``gs.fit(X, y)`` with a
solver that draws) reads the stream's lineage into its key. With a seed in a
cell below it, run on an earlier pass of the same kernel, the stream has a
lineage, and the fit moves it on. The check before the next cell simulated
the fit on the stream as it stood after the fit, missed the entry the fit
had just stored, took the variables stored with the fit for stale and ran
the fit again (a 25 s grid search, twice per Run All).
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("sklearn")


def _cells(log: str) -> list[str]:
    return [
        "import os, time\nimport numpy as np\n"
        "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import GridSearchCV",
        "X = np.arange(40.0).reshape(20, 2) % 7\ny = np.arange(20) % 2",
        "def score(est, X, y):\n"
        f"    fd = os.open({log!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S} / 4)\n"
        "    return est.score(X, y)",
        "Cs = [1.0, 10.0]\ngrid = {'C': Cs}",
        "gs = GridSearchCV(LogisticRegression(solver='liblinear'), grid, scoring=score, cv=2)",
        "gs.fit(X, y)",
        "print(gs.best_params_)",
        "np.random.seed(0)\nnoise = np.random.rand(2)",
    ]


@pytest.fixture
def notebook(cash_magics, tmp_path):
    log = tmp_path / "fits.log"
    cells = _cells(str(log))

    def fits() -> int:
        return len(log.read_bytes()) if log.exists() else 0

    def run(index: int) -> None:
        run_cash_cell(cash_magics, cells[index], cells=cells)

    run.fits = fits
    run.cells = cells
    return run


def test_the_cell_below_a_fit_does_not_run_it_again(notebook):
    for _ in range(2):
        for index in range(len(notebook.cells)):
            before = notebook.fits()
            notebook(index)
            if index == 6:
                assert notebook.fits() == before, "the check before the cell below the fit ran the fit again"
