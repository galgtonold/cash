"""``print(m.fit_transform(x))`` changes ``m`` as much as a bare ``m.fit_transform(x)``.

A cell that builds a fresh model and fits it inside a ``print`` (or any
other wrapping call: ``print(len(m.fit(x)))``) cached the print like a pure
read: on a re-run ``m = Model()`` ran again, the print was replayed from the
cache, and ``m`` stayed unfitted, with no warning. A method call nested in a
bare call is now watched like one whose result is bound, so when it changes
its receiver the statement runs every time.
"""

from __future__ import annotations

import time

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


class Model:
    """A model whose ``fit_transform`` fits it in place and returns rows."""

    def __init__(self):
        self.coef = None

    def fit_transform(self, x):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        self.coef = sum(x)
        return [v * 2 for v in x]

    def score(self, x):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return sum(x)


def _run_all(magics, cells):
    magics.shell.user_ns["Model"] = Model
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


@pytest.mark.parametrize(
    "fit",
    ["print(m.fit_transform([1, 2, 3]))", "print(len(m.fit_transform([1, 2, 3])))"],
    ids=["print", "print-len"],
)
def test_a_rerun_leaves_the_model_fitted(cash_magics, fit):
    cell = f"m = Model()\n{fit}"
    _run_all(cash_magics, [cell])
    assert cash_magics.shell.user_ns["m"].coef == 6, "first run"
    _run_all(cash_magics, [cell])
    assert cash_magics.shell.user_ns["m"].coef == 6, "re-run: the print was restored and m never fitted"


def test_a_read_inside_a_print_is_still_stored(cash_magics, statement_processor):
    """Control: ``print(m.score(x))`` leaves ``m`` alone, so it caches."""
    _run_all(cash_magics, ["m = Model()"])
    metrics = statement_processor.process_statement("print(m.score([1, 2]))")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert not metrics.get("uncacheable_reasons"), metrics.get("uncacheable_reasons")
    again = statement_processor.process_statement("print(m.score([1, 2]))")
    assert again["status"] == CacheStatus.RESTORED
