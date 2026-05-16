"""Tests for LineageStore — the single seam for reading/writing variable lineage."""
from __future__ import annotations

import hashlib

import pytest

from cash.notebook.lineage_store import LineageStore


class _Box:
    """A trivial container so we can attach ``_cash_lineage_hash`` to it."""


class TestGet:
    def test_returns_none_for_unknown_variable(self):
        store = LineageStore()
        assert store.get("x") is None

    def test_returns_recorded_hash(self):
        store = LineageStore()
        store.record("x", "abc123")
        assert store.get("x") == "abc123"

    def test_contains_protocol(self):
        store = LineageStore()
        store.record("x", "abc")
        assert "x" in store
        assert "y" not in store

    def test_iter_yields_recorded_names(self):
        store = LineageStore()
        store.record("x", "h1")
        store.record("y", "h2")
        assert set(iter(store)) == {"x", "y"}


class TestRecordCoordinatesAttribute:
    """``record(var, hash, value=val)`` must set both the dict entry and ``_cash_lineage_hash``.

    This is the core invariant: today the dict and the attribute are written separately
    in 6 places inside notebook/ and they drift if one is forgotten.
    """

    def test_record_without_value_only_updates_dict(self):
        store = LineageStore()
        store.record("x", "abc")
        assert store.get("x") == "abc"

    def test_record_with_value_sets_attribute(self):
        store = LineageStore()
        box = _Box()
        store.record("x", "deadbeef", value=box)
        assert box._cash_lineage_hash == "deadbeef"
        assert store.get("x") == "deadbeef"

    def test_record_with_value_that_cannot_take_attributes_does_not_raise(self):
        """Builtins like int / str can't hold attributes — must not crash."""
        store = LineageStore()
        store.record("x", "abc", value=42)
        assert store.get("x") == "abc"  # dict still updated


class TestResetTo:
    """``reset_to`` is the *resynchronisation* writer — distinct from ``record``.

    Used when downstream advancement leaves the actual lineage 'ahead' of where
    simulation says it should be (upstream.py l.572, l.1597). The name is the
    documentation: searching the codebase for ``reset_to`` finds every resync site.
    """

    def test_overwrites_existing_lineage(self):
        store = LineageStore()
        store.record("x", "advanced")
        store.reset_to("x", "virtual")
        assert store.get("x") == "virtual"

    def test_does_not_touch_value_attribute(self):
        """reset_to is a state-machine correction; the value's _cash_lineage_hash
        belongs to the value's actual computation, not to the simulator's view.

        The value isn't passed to reset_to (by design) — but we verify here that
        a pre-existing attribute on a value in user_ns is not invalidated, since
        reset_to has no access to it.
        """
        store = LineageStore()
        box = _Box()
        box._cash_lineage_hash = "value_hash"
        store.record("x", "advanced")  # no value: attribute untouched
        store.reset_to("x", "virtual")
        assert box._cash_lineage_hash == "value_hash"
        assert store.get("x") == "virtual"


class TestResolvePriorityLadder:
    """The priority ladder pulled out of ``cache_key._resolve_input_lineage``.

    Order: virtual → store → value._cash_lineage_hash → compute_hash_fn → str(val).
    """

    def test_virtual_lineage_wins_over_store(self):
        store = LineageStore()
        store.record("x", "store_hash")
        result = store.resolve("x", value=None, virtual={"x": "virtual_hash"})
        assert result == "virtual_hash"

    def test_store_wins_when_no_virtual(self):
        store = LineageStore()
        store.record("x", "store_hash")
        result = store.resolve("x", value=None, virtual={})
        assert result == "store_hash"

    def test_falls_back_to_value_attribute(self):
        store = LineageStore()
        box = _Box()
        box._cash_lineage_hash = "value_hash"
        result = store.resolve("x", value=box, virtual={})
        assert result == "value_hash"

    def test_falls_back_to_compute_hash_fn(self):
        store = LineageStore()
        result = store.resolve(
            "x", value=42, virtual={},
            compute_hash_fn=lambda v: f"computed:{v}",
        )
        assert result == "computed:42"

    def test_falls_back_to_str_hash_when_no_compute_fn(self):
        store = LineageStore()
        expected = hashlib.sha256(b"42").hexdigest()
        result = store.resolve("x", value=42, virtual={})
        assert result == expected

    def test_returns_none_when_value_is_none_and_not_in_store(self):
        store = LineageStore()
        result = store.resolve("x", value=None, virtual={})
        assert result is None

    def test_recursionerror_during_compute_is_swallowed(self):
        """Mirrors the existing ``except (AttributeError, TypeError, RecursionError)``
        behaviour in cache_key._compute_val_lineage."""
        store = LineageStore()

        def boom(_):
            raise RecursionError

        result = store.resolve("x", value=object(), virtual={}, compute_hash_fn=boom)
        assert result is None


class TestTrackingStateWiring:
    """``TrackingState.lineage`` is a LineageStore that shares state with
    ``TrackingState.variable_lineage``. During migration the dict view and the
    store view must observe each other's writes."""

    def test_state_exposes_lineage_store(self):
        from cash.notebook._protocols import TrackingState
        state = TrackingState()
        assert isinstance(state.lineage, LineageStore)

    def test_dict_writes_visible_through_store(self):
        from cash.notebook._protocols import TrackingState
        state = TrackingState()
        state.variable_lineage["x"] = "h1"
        assert state.lineage.get("x") == "h1"

    def test_store_writes_visible_through_dict(self):
        from cash.notebook._protocols import TrackingState
        state = TrackingState()
        state.lineage.record("x", "h1")
        assert state.variable_lineage["x"] == "h1"


class TestBackingDictCompatibility:
    """During migration, callers that still want the raw dict (cache_key.py's
    ``CacheKeyContext.variable_lineage`` field, persistence layer) must keep working.
    """

    def test_as_dict_exposes_live_view(self):
        store = LineageStore()
        store.record("x", "h1")
        d = store.as_dict()
        assert d == {"x": "h1"}

    def test_as_dict_reflects_subsequent_writes(self):
        store = LineageStore()
        d = store.as_dict()
        store.record("x", "h1")
        assert d.get("x") == "h1"  # live view, not a snapshot
