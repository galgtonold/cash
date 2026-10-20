"""A store the backend refuses warns STORE-FAILED.

A full disk or a cache_dir that cannot be written fails ``backend.set`` with
whatever exception the backend raises; only the debug log said so, and the
next run of the notebook missed for no reason the user could see. The
statement's own result is unchanged.
"""

from __future__ import annotations

import warnings

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell


class DiskSaidNo(Exception):
    pass


@pytest.mark.parametrize("exc", [DiskSaidNo, RuntimeError, KeyError, MemoryError, OSError])
def test_a_refused_store_warns(cash_magics, cash_instance, monkeypatch, exc):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)

    def refuse(*args, **kwargs):
        raise exc("no space left")

    cells = ["import numpy as np", "x = np.arange(5.0) * 2", "y = float(x.sum())"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    monkeypatch.setattr(cash_instance.backend, "set", refuse)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells[1:]:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert "STORE-FAILED" in [getattr(w.message, "code", None) for w in caught]
    assert cash_magics.shell.user_ns["y"] == 20.0
