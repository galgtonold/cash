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
import pytest

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


def test_a_served_rerun_alone_is_served_again(cash_magics, monkeypatch):
    """The entry recorded what ``w -= 1`` was built on as w's lineage after
    it ran: its own output's. Served, w then counted as built on the -1 it
    holds, the next run alone did not start from the base, and gave -2."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    ns = cash_magics.shell.user_ns
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    for _ in range(3):
        run_cash_cell(cash_magics, "w -= 1", cells=CELLS)
        assert np.array_equal(ns["w"], [-1.0, -1.0, -1.0])


ALONE = {
    "an array": (["import numpy as np", "w = np.zeros(3)", "w += 1"], lambda ns: ns["w"].tolist(), [1.0, 1.0, 1.0]),
    "a dict": (["d = {'n': 0}", "d['n'] += 1"], lambda ns: dict(ns["d"]), {"n": 1}),
    "a list": (["xs = [0, 0, 0]", "for i in range(3):\n    xs[i] += 1"], lambda ns: list(ns["xs"]), [1, 1, 1]),
    "a frame": (
        ["import pandas as pd", "m = pd.DataFrame({'a': [0, 0, 0]})", "for i in range(3):\n    m.loc[i, 'a'] += 1"],
        lambda ns: ns["m"]["a"].tolist(),
        [1, 1, 1],
    ),
}


@pytest.mark.parametrize("name", list(ALONE))
def test_a_cell_that_changes_what_it_reads_run_alone_again_and_again_starts_from_its_base(
    cash_magics, monkeypatch, name
):
    """Served every other time, computed from its own output in between:
    {'n': 1}, {'n': 2}, {'n': 1}."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells, read, want = ALONE[name]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    seen = []
    for _ in range(3):
        run_cash_cell(cash_magics, cells[-1], cells=cells)
        seen.append(read(cash_magics.shell.user_ns))
    assert seen == [want] * 3
