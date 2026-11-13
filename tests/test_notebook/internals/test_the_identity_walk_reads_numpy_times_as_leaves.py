"""The identity walk takes numpy ``timedelta64`` and ``datetime64`` as
leaves and never as an ``int``: ``timedelta64`` subclasses
``signedinteger``, and ``int()`` of one in seconds is a ``TypeError``,
which broke the user's loop collecting them. Whatever else the walk raises
refuses the value with NOTEBOOK-IDENTITY-HELD's reason rather than the
user's cell. Pins the work, not the behaviour: a refactor may expect this
to fail."""

from __future__ import annotations

import numpy as np
import pytest

from cash.notebook.statement.identity_holders import IdentityHeld, _value_kinds, _Walk, identity_held_reason


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_a_list_of_timedeltas_holds_no_id(unit):
    deltas = list(np.diff(np.arange(0, 50_000, 3).astype(f"timedelta64[{unit}]")))
    deltas.append(np.timedelta64("NaT", unit))
    assert identity_held_reason("deltas", deltas, {}) is None
    assert identity_held_reason("deltas", deltas[:3], {}) is None


def test_a_list_of_datetimes_holds_no_id():
    moments = list(np.arange(0, 50_000).astype("datetime64[s]"))
    assert identity_held_reason("moments", moments, {}) is None


def test_times_are_leaves_and_not_integers():
    leaves, ints = _value_kinds()
    assert {np.timedelta64, np.datetime64} <= leaves
    assert np.timedelta64 not in ints and np.int64 in ints


def test_an_error_in_the_walk_refuses_the_value(monkeypatch):
    def fail(walk, items):
        raise RuntimeError("boom")

    monkeypatch.setattr(_Walk, "run", fail)
    held = identity_held_reason("deltas", [1, 2], {})
    assert isinstance(held, IdentityHeld) and held.name == "deltas"
    assert "could not look through 'deltas'" in held.what and "boom" in held.what
