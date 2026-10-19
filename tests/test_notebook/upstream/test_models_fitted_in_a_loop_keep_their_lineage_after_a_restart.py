"""Models fitted in a loop leave it with the lineage they left it with
before a restart, so what is built on them is served after one.

``for m in models: m.fit(X, y)`` moved the fitted models on only on the
pass that saw the fit and learnt the loop body's verdict. After a restart
the verdict is recalled, nothing is seen, and the models left the loop
with other lineages: E8's ``voting_clf.fit`` on them ran again (31 s).
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("sklearn")

DATA = "import time\nimport numpy as np\nX = np.arange(40.0).reshape(20, 2)\ny = X.sum(axis=1)"
MODELS = (
    "from sklearn.linear_model import LinearRegression, Ridge\n"
    "a = LinearRegression()\nb = Ridge(alpha=0.5)\nmodels = [a, b]"
)
LOOP = "for m in models:\n    print('fitting', m)\n    m.fit(X, y)\n    time.sleep(0.6)"
USE = f"total = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), float(a.coef_.sum() + b.coef_.sum()))[1]"
CELLS = [DATA, MODELS, LOOP, USE]
NAMES = ["time", "np", "X", "y", "LinearRegression", "Ridge", "a", "b", "models", "m", "total"]


@pytest.fixture
def clean_backend(tmp_path):
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    # A new kernel asks the records again for the verdicts it has not seen.
    processor = magics._statement_processor
    processor._mutation_routing._unrecorded.clear()
    processor._in_loop._unrecorded.clear()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _run(magics) -> list[str | None]:
    out = []
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)
        out.append(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status"))
    return out


def test_the_fitted_models_keep_their_lineage_and_what_reads_them_is_served(cash_magics, clean_backend):
    _run(cash_magics)
    lineage = cash_magics.tracking_state.variable_lineage
    before = (lineage["a"], lineage["b"])
    _restart(cash_magics)

    statuses = _run(cash_magics)
    assert (lineage["a"], lineage["b"]) == before
    assert statuses[3] == CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["total"] == pytest.approx(4.0, abs=0.5)
