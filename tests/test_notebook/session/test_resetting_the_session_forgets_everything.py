"""Resetting the session state forgets every field, not a chosen few.

The integration harness simulates a fresh kernel with
``TrackingState.reset_session_state()``. A reset that cleared only some of the
fields would leave mutation verdicts, RNG states or control outcomes behind,
and a test of "a fresh session" would run against a half-remembered one. The
check below walks the dataclass's own field list, so a field added later is
covered without anyone remembering to add it here.
"""

from __future__ import annotations

import dataclasses

import pytest

from cash.notebook.lineage_store import LineageStore
from cash.notebook.recorded_reads import ReadRecord
from cash.notebook.tracking_state import TrackingState


def _mark(state: TrackingState) -> dict[str, object]:
    """Give every field a non-default value; return the containers it holds."""
    held = {}
    for f in dataclasses.fields(state):
        value = getattr(state, f.name)
        if isinstance(value, LineageStore):
            value.record("x", "h")
        elif isinstance(value, ReadRecord):
            value.watched.add(("env", "X"))
            value.known[("env", "X")] = "d"
        elif isinstance(value, dict):
            value["x"] = {"y"}
        elif isinstance(value, set):
            value.add("x")
        elif isinstance(value, int):
            setattr(state, f.name, value + 7)
        elif value is None:
            setattr(state, f.name, frozenset({"x"}))
        else:
            pytest.fail(f"{f.name}: no way to mark a {type(value).__name__}; extend _mark")
        if hasattr(getattr(state, f.name), "clear"):
            held[f.name] = getattr(state, f.name)
    return held


def test_every_field_goes_back_to_its_default():
    state = TrackingState()
    _mark(state)
    fresh = TrackingState()
    assert any(getattr(state, f.name) != getattr(fresh, f.name) for f in dataclasses.fields(state))

    state.reset_session_state()

    left = [f.name for f in dataclasses.fields(state) if getattr(state, f.name) != getattr(fresh, f.name)]
    assert not left, f"reset_session_state left these fields set: {left}"


def test_a_container_another_component_holds_is_emptied_too():
    """The components share the containers, so a reset must not swap them out."""
    state = TrackingState()
    held = _mark(state)

    state.reset_session_state()

    stale = [name for name, container in held.items() if len(container)]
    assert not stale, f"a held reference still sees the old contents: {stale}"
    swapped = [name for name, container in held.items() if getattr(state, name) is not container]
    assert not swapped, f"replaced rather than emptied: {swapped}"
