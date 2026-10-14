"""An array read by two slow statements is cut into block digests once
(``CACHE-PER-ITEM-RERUN``): the fingerprint is kept by value and lineage
for the next statement that reads the same value."""

from __future__ import annotations

import cash.notebook.per_item_reruns as per_item_reruns
from tests._cell_driver import run_cash_cell

CELLS = [
    "import numpy as np, time",
    "imgs = np.arange(3000 * 16, dtype=np.int64).reshape(3000, 16)",
    "a = [imgs * 2, time.sleep(1.1)][0]",
    "b = [imgs * 3, time.sleep(1.1)][0]",
]


def test_the_second_statement_reuses_the_fingerprint(cash_magics, monkeypatch):
    taken = []
    real = per_item_reruns.fingerprint

    def counting(value, **kw):
        taken.append(value.shape)
        return real(value, **kw)

    monkeypatch.setattr(per_item_reruns, "fingerprint", counting)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell)

    assert taken == [(3000, 16)], taken
    assert len(cash_magics.tracking_state.per_item_reruns._records) == 2, "both statements keep a record"
