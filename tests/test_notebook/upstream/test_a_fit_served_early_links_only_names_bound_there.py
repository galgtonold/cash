"""A statement served from the cache links only names bound where it runs.

What a statement linked is recorded by its text, so ``vote.fit(X, y)``
written twice keeps what the later one linked: the parts the second
``vote`` holds. A hit on the first one, before ``parts`` exists, linked
them too, where its run had linked nothing; the later names moved on
where a run does not move them, and E8's second ``voting_clf.fit`` ran
again after a restart (31 s).
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
CELLS = [DATA, FIRST, FIT + "\nfirst = 1", PARTS, LOOP, SECOND, FIT + "\nsecond = 2", USE]
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
    """The loop, the second vote and its fit are never stored, as the
    second fit (a few ms) is left to the floor: stored in one run and not
    the other under load."""
    floor = config.min_execution_time_to_cache_seconds
    out = []
    for i, cell in enumerate(CELLS):
        config.min_execution_time_to_cache_seconds = 3600.0 if i in UNSTORED else floor
        run_cash_cell(magics, cell, cells=CELLS)
        out.append(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status"))
    return out


def test_the_early_hit_links_nothing_the_later_fit_linked(cash_magics, cash_instance, clean_backend):
    _run(cash_magics, cash_instance.config)
    lineage = cash_magics.tracking_state.variable_lineage
    before = {name: lineage.get(name) for name in ("vote", "parts", "part")}
    _restart(cash_magics)

    statuses = _run(cash_magics, cash_instance.config)
    assert {name: lineage.get(name) for name in before} == before
    assert statuses[-1] == CacheStatus.RESTORED
