"""A cell run again on its own that changes what it reads (``w -= 1``)
starts from the value the cells above give it. When the cell's own cache
entry restores that value, the upstream check leaves the variable for the
entry and keys the statement as if restored. Should the lookup refuse the
entry after all, for a reason the check could not foresee, the statement ran
on the -1 it had written already, gave -2, and stored it under the key of
the restored w: every later hit served -2.

Now it warns (NOTEBOOK-INPUT-NOT-REBUILT), and stores nothing.
"""

from __future__ import annotations

import warnings

import numpy as np

from cash.backends.persistence_policy import PersistencePolicy
from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell

CELLS = ["import numpy as np", "w = np.zeros(3)", "w -= 1"]


def _codes(magics, cell) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(magics, cell, cells=CELLS)
    return [c for w in caught if (c := getattr(w.message, "code", None))]


def test_a_refused_hit_warns_and_stores_nothing(cash_magics, monkeypatch):
    # A loaded machine stores the cheap steps too.
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    ns = cash_magics.shell.user_ns
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    assert np.array_equal(ns["w"], [-1.0, -1.0, -1.0])

    # Run again alone, its own entry would restore w; the lookup refuses it.
    real = processor.StatementProcessor._live_objects_shared

    def refuse(self, run, metadata):
        return True if run.code == "w -= 1" else real(self, run, metadata)

    monkeypatch.setattr(processor.StatementProcessor, "_live_objects_shared", refuse)
    codes = _codes(cash_magics, "w -= 1")
    assert "NOTEBOOK-INPUT-NOT-REBUILT" in codes
    assert np.array_equal(ns["w"], [-2.0, -2.0, -2.0]), "control: it ran on the value in memory"

    # Nothing was stored under the key of the restored w: served now, it is -1.
    monkeypatch.setattr(processor.StatementProcessor, "_live_objects_shared", real)
    run_cash_cell(cash_magics, "w -= 1", cells=CELLS)
    assert np.array_equal(ns["w"], [-1.0, -1.0, -1.0])
    statuses = [str(s["status"]) for s in cash_magics.cash_status("dict")["last_cell"]["statements"]]
    assert statuses == ["CacheStatus.RESTORED"] or statuses == ["RESTORED"], statuses
