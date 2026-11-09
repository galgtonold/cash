"""The restore-cost check of an entry that keeps a slice variable as an
anchor weighs what a hit reads back and does: the output with a reference
in the slice's place, not the slice's data, plus the time the content
check of the slice takes.
"""

from __future__ import annotations

import pytest

from cash.notebook.anchors import AnchorRef
from cash.notebook.statement.store import StatementStore
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

FIT = f"model = Keeper().fit(X_train, {ABOVE_PERSISTENCE_FLOOR_S})"
CELLS = [
    "import time\nimport numpy as np\n"
    "class Keeper:\n"
    "    def fit(self, X, wait):\n"
    "        time.sleep(wait)\n"
    "        self.X_ = X\n"
    "        return self",
    "X = np.arange(400_000.0).reshape(1000, 400).copy()",
    "X_train = X[:800]",
    FIT,
]


def test_the_slice_is_weighed_as_a_reference_with_its_check(cash_magics, monkeypatch):
    asked: list[tuple[dict, float]] = []
    real = StatementStore.should_skip_large_object_caching

    def weigh(self, values, *args, check_seconds=0.0, **kwargs):
        asked.append((dict(values), check_seconds))
        return real(self, values, *args, check_seconds=check_seconds, **kwargs)

    monkeypatch.setattr(StatementStore, "should_skip_large_object_caching", weigh)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)

    values, check_seconds = asked[-1]
    assert isinstance(values["model"].X_, AnchorRef), values
    assert check_seconds > 0
