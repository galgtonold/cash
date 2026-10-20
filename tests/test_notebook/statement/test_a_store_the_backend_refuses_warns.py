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


def test_a_value_cash_cannot_keep_by_reference_says_so(cash_magics, monkeypatch):
    """``hooks = {'add': results.append}`` holds ``results``, which cash keeps
    as a reference to the variable; the copy that puts it in place rebuilds
    ``results.append`` as ``getattr(reference, 'append')`` and fails. That
    is cash's own step, not the backend: it has its own code and fix."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = ["results = []", "hooks = {'add': results.append, 'n': sum(range(10))}"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[1], cells=cells)
    codes = [getattr(w.message, "code", None) for w in caught]
    assert "STORE-REF-UNSUPPORTED" in codes and "STORE-FAILED" not in codes, codes
    cash_magics.shell.user_ns["hooks"]["add"](5)
    assert cash_magics.shell.user_ns["results"] == [5]


def _store_warnings(cash_magics, cells: list[str]) -> dict[str, str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    return {w.message.code: str(w.message) for w in caught if str(getattr(w.message, "code", "")).startswith("STORE")}


def test_the_warning_names_only_the_variable_it_could_not_rebuild_around(cash_magics, monkeypatch):
    """``other`` is held as it is and kept by reference; only ``results``,
    held through a method bound to it, stops the store."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = ["results = [1]", "other = [2]", "hooks = {'add': results.append, 'o': other}"]
    found = _store_warnings(cash_magics, cells)
    assert list(found) == ["STORE-REF-UNSUPPORTED"], found
    assert "'results'" in found["STORE-REF-UNSUPPORTED"] and "'other'" not in found["STORE-REF-UNSUPPORTED"]


def test_a_value_whose_own_pickling_raises_next_to_a_held_variable_is_store_failed(cash_magics, monkeypatch):
    """Its own ``__reduce__`` raised, as it would with nothing held: not a
    form cash cannot keep as a reference."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    bad = "class Bad:\n    def __reduce__(self):\n        raise ValueError('Bad refuses pickling')\n"
    cells = [bad, "results = [1]", "h = {'r': results, 'b': Bad()}"]
    found = _store_warnings(cash_magics, cells)
    assert list(found) == ["STORE-FAILED"], found
    assert "Bad refuses pickling" in found["STORE-FAILED"]
