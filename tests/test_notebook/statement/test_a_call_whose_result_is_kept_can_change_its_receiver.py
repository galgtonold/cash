"""``history = net.fit(X)`` changes ``net`` as much as a bare ``net.fit(X)``.

A training or stepping call that returns something (a history, a loss) was
cached like a pure capture (``m = df.mean()``): the next Run All restored
``history`` and never trained ``net``, so later predictions came from an
untrained model, with no warning. The receiver of a method call whose result
is bound is now watched on the statement's first run, like the argument of a
bare call: when the call changed it, the statement runs every time.
"""

from __future__ import annotations

import time

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


class Net:
    """A model whose ``fit`` trains it in place and returns a history."""

    def __init__(self):
        self.w = 0.0

    def fit(self, X, epochs=1):
        hist = []
        for _ in range(epochs):
            time.sleep(ABOVE_PERSISTENCE_FLOOR_S / 2)
            self.w += sum(X)
            hist.append(self.w)
        return {"loss": hist}

    def score(self, X):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return self.w * sum(X)

    def predict(self, x):
        return self.w * x


CELLS = ["X = [1.0, 2.0]\nnet = Net()", "history = net.fit(X, epochs=3)", "pred = net.predict(10)"]


def _run_all(magics, cells):
    magics.shell.user_ns["Net"] = Net
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


def test_a_second_run_all_trains_the_model_again(cash_magics):
    ns = cash_magics.shell.user_ns
    _run_all(cash_magics, CELLS)
    assert ns["pred"] == 90.0, "first Run All"
    _run_all(cash_magics, CELLS)
    assert ns["pred"] == 90.0, "second Run All: history was restored and net never trained"


def test_the_statement_says_what_it_changed(cash_magics, statement_processor):
    _run_all(cash_magics, CELLS[:1])
    metrics = statement_processor.process_statement(CELLS[1])
    assert metrics["status"] == CacheStatus.COMPUTED
    assert "net" in metrics["evaluated_vars"], metrics["evaluated_vars"]
    assert any("net" in r for r in metrics.get("uncacheable_reasons") or []), metrics.get("uncacheable_reasons")


def test_a_call_that_leaves_its_receiver_alone_is_stored(cash_magics, statement_processor):
    """Control: ``s = net.score(X)`` reads ``net`` only, so it caches."""
    _run_all(cash_magics, CELLS[:1])
    metrics = statement_processor.process_statement("s = net.score(X)")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert not metrics.get("uncacheable_reasons"), metrics.get("uncacheable_reasons")
    again = statement_processor.process_statement("s = net.score(X)")
    assert again["status"] == CacheStatus.RESTORED
