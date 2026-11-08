"""A run that takes no digest moves what an output holds as one that did.

``cv_res.sort_values(..., inplace=True)`` is never stored (its share check
gives up), so it runs on every Run All, and it changes ``grid_search`` in place, and so,
maybe, the variables holding the pipelines the search holds. The first run
digested them before and after and kept their lineages, recording the move
each would have taken as an alias of the lineage it kept
(`keep_held_lineage`), which the upstream simulation resolves. A later run
whose share check ran out of time -- more objects in the namespace by then
-- took no digest and moved them all on. The simulation kept them, so the
next cell re-ran upstream every cell that fits on them: 130 s of refits on a
same-code Run All of E6 (homl ch02).

A run that takes no digest now moves a holder as the simulation does; one
whose digests saw it change still moves it on. Here ``vote.fit`` stands in:
it fits clones of the trees ``a`` and ``b`` it holds, and runs again with no
entry to serve it.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell

pytest.importorskip("sklearn")

DEFINE = """import os
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
LOG = LOG_PATH
def mark():
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"f")
    os.close(fd)"""

TREE = """class Tree(DecisionTreeClassifier):
    def fit(self, X, y, sample_weight=None, check_input=True):
        mark()
        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)"""

CELLS = [
    TREE,
    "X = np.array([[i % 7, i % 5] for i in range(60)])\ny = np.array([i % 2 for i in range(60)])",
    "a = Tree(max_depth=2, random_state=0)\nb = Tree(max_depth=3, random_state=0)",
    "a.fit(X, y)",
    "b.fit(X, y)",
    "vote = VotingClassifier([('a', a), ('b', b)])",
    "vote.fit(X, y)",
    "from sklearn.base import clone\nagain = clone(a).fit(X, y)\ndepth = again.get_depth()",
]
FIT = CELLS.index("vote.fit(X, y)") + 1  # DEFINE first


@pytest.fixture(autouse=True)
def _digests_in_time(monkeypatch):
    """The first run's digests finish whatever the machine's load."""
    monkeypatch.setattr(processor, "HOLDER_DIGEST_FLOOR_S", 30.0)
    monkeypatch.setattr(processor, "HOLDER_DIGEST_BUDGET_S", 30.0)


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0


def _no_digest(monkeypatch):
    """The run takes no digest of the holders, as when its share check runs
    out of time or the digests run past their budget in a loaded namespace:
    it knows neither that they kept their content nor that they changed."""
    monkeypatch.setattr(processor.StatementProcessor, "_holder_digests", lambda self, run: {})
    monkeypatch.setattr(processor.StatementProcessor, "_unchanged_holders", lambda self, run, holders: set())


def _forget_the_fit(cash_instance) -> None:
    """Drop the ensemble fit's entry, and only it: the fit runs again on the
    next Run All, as a statement too quick to store does."""
    gone = [
        meta["key"]
        for meta in list(cash_instance.backend.list_entries() or ())
        if str(meta.get("code", "")).strip() == "vote.fit(X, y)"
    ]
    assert gone
    for key in gone:
        cash_instance.backend.delete(key)


def _fits(log) -> int:
    n = len(log.read_bytes()) if log.exists() else 0
    log.unlink(missing_ok=True)
    return n


def _run_all(cash_magics, tmp_path):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "fits.log"
    cells = [DEFINE.replace("LOG_PATH", repr(str(log))), *CELLS]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    return log, cells


@pytest.mark.usefixtures("store_everything")
def test_a_run_all_with_no_digest_keeps_the_lineages_the_first_kept(cash_magics, cash_instance, tmp_path, monkeypatch):
    log, cells = _run_all(cash_magics, tmp_path)
    assert _fits(log) == 5  # a, b, the ensemble's two clones, and again
    lineage = cash_magics.tracking_state.variable_lineage
    kept = {name: lineage[name] for name in ("a", "b", "again")}
    depth = cash_magics.shell.user_ns["depth"]

    _forget_the_fit(cash_instance)
    _no_digest(monkeypatch)
    _run_all(cash_magics, tmp_path)

    # ``again`` is served: its key reads ``a``'s lineage.
    assert {name: lineage[name] for name in ("a", "b", "again")} == kept
    assert cash_magics.shell.user_ns["depth"] == depth
    assert cash_magics.shell.user_ns["vote"].estimators[0][1] is cash_magics.shell.user_ns["a"]
