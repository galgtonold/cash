"""An integer array of at least `_BOUNDS_FIRST` numbers is first told by its
lowest and highest number: when no mapping an object can lie in overlaps
them, none is an ``id()`` (``np.arange(10_000_000)``: two passes, not five).
One ``id()`` among small numbers still spans the heap, and is found."""

from __future__ import annotations

import pytest

from cash.notebook.statement import identity_holders

np = pytest.importorskip("numpy")


class _Thing:
    pass


def test_an_id_among_small_numbers_is_still_found():
    thing = _Thing()
    column = np.arange(identity_holders._BOUNDS_FIRST * 3, dtype=np.int64)
    column[len(column) // 2] = id(thing)
    reason = identity_holders.identity_held_reason("v", column, {"thing": thing})
    assert reason is not None and reason.kind == "an id()"


def test_small_numbers_alone_are_passed_by_their_bounds(monkeypatch):
    thing = _Thing()
    column = np.arange(identity_holders._BOUNDS_FIRST * 3, dtype=np.int64)
    numpy_compress = np.compress
    calls = []
    monkeypatch.setattr(np, "compress", lambda *a, **k: calls.append(1) or numpy_compress(*a, **k))
    assert identity_holders.identity_held_reason("v", column, {"thing": thing}) is None
    assert calls == [], "the whole array was masked though its bounds hold no address"
