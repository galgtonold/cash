"""Review of ecfef08: a fit served early links what its later place linked
whenever those names are bound already, though they share nothing here.

``parts`` exists before the first ``vote.fit(X, y)`` but the first vote does
not hold it. The record of the text is the second fit's, which linked the
parts the second vote holds, so the early hit links ``vote`` to ``parts``;
the loop fitting the parts then moves ``vote`` on where a run does not, and
the second fit and the prediction run again after a restart. Fails on
ecfef08 (lineages differ; the second fit is COMPUTED).
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("sklearn")

DATA = "import time\nimport numpy as np\nX = np.arange(400.0).reshape(200, 2)\ny = X.sum(axis=1)"
FIRST = (
    "from sklearn.linear_model import LinearRegression, Ridge\n"
    "from sklearn.ensemble import VotingRegressor\n"
    "vote = VotingRegressor([('lr', LinearRegression()), ('rg', Ridge())])"
)
FIT = "vote.fit(X, y)"
PARTS = "parts = [LinearRegression(), Ridge(alpha=0.5)]"
LOOP = "for part in parts:\n    part.fit(X, y)"
SECOND = "vote = VotingRegressor([('a', parts[0]), ('b', parts[1])])"
USE = f"total = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), float(vote.predict(X[:3]).sum()))[1]"
CELLS = [
    DATA,
    "from sklearn.linear_model import LinearRegression, Ridge\n" + PARTS,
    FIRST,
    FIT + "\nfirst = 1",
    LOOP,
    SECOND,
    FIT + "\nsecond = 2",
    USE,
]
UNSTORED = (4, 5, 6)  # LOOP, SECOND and the second fit
NAMES = [
    "time",
    "np",
    "X",
    "y",
    "LinearRegression",
    "Ridge",
    "VotingRegressor",
    "vote",
    "parts",
    "part",
    "first",
    "second",
    "total",
]


@pytest.fixture
def clean_backend(tmp_path):
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    processor = magics._statement_processor
    processor._mutation_routing._unrecorded.clear()
    processor._in_loop._unrecorded.clear()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _run(magics, config) -> list[str | None]:
    """The loop, the second vote and its fit are never stored: the early hit
    then takes the record its text left last, the second fit's, while that
    fit runs. Left to the floor, the second fit (a few ms) is stored in one
    run and not the other under load."""
    floor = config.min_execution_time_to_cache_seconds
    out = []
    for i, cell in enumerate(CELLS):
        config.min_execution_time_to_cache_seconds = 3600.0 if i in UNSTORED else floor
        run_cash_cell(magics, cell, cells=CELLS)
        out.append(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status"))
    return out


def test_the_early_hit_links_no_bound_name_its_run_did_not_link(cash_magics, cash_instance, clean_backend):
    _run(cash_magics, cash_instance.config)
    lineage = cash_magics.tracking_state.variable_lineage
    before = {name: lineage.get(name) for name in ("vote", "parts", "part")}
    _restart(cash_magics)

    statuses = _run(cash_magics, cash_instance.config)
    assert {name: lineage.get(name) for name in before} == before
    assert statuses[-1] == CacheStatus.RESTORED
