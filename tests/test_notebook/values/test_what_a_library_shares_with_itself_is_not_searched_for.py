"""What a fitted model shares with its library only is not searched for.

A fitted ``HistGradientBoostingRegressor`` holds the interpreter's one empty
tuple ``()`` and, through its loss, the ``Interval`` sklearn's link classes
keep as a class attribute. The check for other variables holding the model's
objects climbed their referrers: through the thousands of objects that hold
``()``, and past the link classes into their subclasses, methods and module.
It read the heap seven times over, about 3 s, never ended, and the fit was
refused -- as too costly to check on a slower machine, as shared on a faster
one -- every run. Neither is the notebook's: nothing in ``()`` can change,
and a library's class is found again by its name, as its module is.
"""

from __future__ import annotations

import pytest

import cash.notebook.shared_objects as shared_objects
from tests._cell_driver import run_cash_cell

pytest.importorskip("sklearn")


def test_a_boosting_fit_is_stored_without_reading_the_heap(cash_magics, monkeypatch):
    reads = []
    real = shared_objects._referrers_of
    monkeypatch.setattr(shared_objects, "_referrers_of", lambda objs, stop: reads.append(len(objs)) or real(objs, stop))
    run_cash_cell(cash_magics, "import numpy as np\nfrom sklearn.ensemble import HistGradientBoostingRegressor")
    run_cash_cell(
        cash_magics,
        "rng = np.random.default_rng(0)\nX = rng.normal(size=(20_000, 4))\ny = X[:, 0] + rng.normal(size=20_000)",
    )
    run_cash_cell(cash_magics, "m = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(X, y)")

    assert reads == [], reads  # the heap was read for objects only the library holds
    row = cash_magics._last_cell_metrics["statements"][-1]
    assert str(row["status"]) == "COMPUTED"
    assert not row.get("uncacheable_reasons"), row.get("uncacheable_reasons")
