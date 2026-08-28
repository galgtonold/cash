"""Registering a hasher for a type cash already fingerprints.

Cash content-hashes numpy arrays and pandas / polars / PyArrow / modin /
dask objects itself, and those run before any registered hasher. A
``register_hasher(np.ndarray, ...)`` therefore succeeded, was stored, and was
never consulted -- with no warning at registration and none at call time. The
only visible symptom was that nothing got faster.

That silence cost a design partner an afternoon, and the effectiveness
warning made it worse: it diagnosed the cause correctly ("a large argument is
being hashed in full on every call") and then named the one remedy that
cannot work for those types.

Two changes, pinned here:

* a bare registration on such a type warns, at the moment the user can still
  act on it;
* ``override=True`` makes the hasher win, because the escape hatch cash used
  to recommend -- wrap the array in a thin type and hash a version field --
  carries exactly the same risk while costing a signature change through the
  whole call chain. Refusing the direct route bought no safety, only work.

The risk it opens is real and is asserted below rather than hedged: an
overriding hasher IS the identity of the value, so two different values that
hash alike share one entry. That is the trade the flag names.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import cash
from cash.exceptions import CashCacheIneffectiveWarning


@pytest.fixture
def c(tmp_path):
    return cash.Cash(cache_dir=str(tmp_path / "cache"))


def _register(instance, type_, fn, **kwargs):
    """Register, returning the warnings it produced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        instance.register_hasher(type_, fn, **kwargs)
    return [w.message for w in caught]


class Tagged:
    """A value carrying cash's notebook lineage marker."""

    def __init__(self, payload, lineage=None):
        self.payload = payload
        if lineage is not None:
            self._cash_lineage_hash = lineage


# ---------------------------------------------------------------------------
# The warning
# ---------------------------------------------------------------------------


def test_registering_on_a_content_hashed_type_warns(c):
    caught = _register(c, np.ndarray, lambda a: "v1")
    assert len(caught) == 1, "the silent no-op is the whole bug"
    assert isinstance(caught[0], CashCacheIneffectiveWarning)
    text = str(caught[0])
    assert "numpy" in text
    assert "override=True" in text, "a warning with no way out is just noise"


def test_the_warning_is_about_a_real_no_op(c):
    """Not a style complaint: assert the hasher genuinely never runs.

    Without this the warning could be firing on something that works fine.
    """
    calls = []
    _register(c, np.ndarray, lambda a: calls.append(1) or "v1")

    @c.cache(assume_safe=True)
    def f(arr):
        return arr[0, 0]

    f(np.zeros((4, 4)))
    f(np.zeros((4, 4)))
    assert calls == [], "the hasher ran, so there was nothing to warn about"


def test_registering_on_an_ordinary_type_is_silent(c):
    """The control. A warning that fires on everything says nothing."""
    assert _register(c, Tagged, lambda t: str(t.payload)) == []


def test_the_warning_does_not_repeat_per_registration(c):
    _register(c, np.ndarray, lambda a: "v1")
    assert _register(c, np.ndarray, lambda a: "v2") == [], "warn once, not per call"


@pytest.mark.parametrize(
    ("type_", "family"),
    [(np.ndarray, "numpy"), (Tagged, None), (dict, None), (str, None)],
)
def test_builtin_hashed_family_names_the_claiming_hasher(type_, family):
    assert cash.Cash.builtin_hashed_family(type_) is family


def test_pandas_frames_are_claimed_too():
    pd = pytest.importorskip("pandas")
    assert cash.Cash.builtin_hashed_family(pd.DataFrame) == "pandas"
    assert cash.Cash.builtin_hashed_family(pd.Series) == "pandas"


# ---------------------------------------------------------------------------
# The override
# ---------------------------------------------------------------------------


def test_override_makes_the_hasher_win(c):
    calls = []
    _register(c, np.ndarray, lambda a: calls.append(1) or "v1", override=True)

    @c.cache(assume_safe=True)
    def f(arr):
        return arr[0, 0]

    f(np.zeros((4, 4)))
    assert calls, "override=True and the hasher still was not consulted"


def test_override_does_not_warn(c):
    """You asked for it; saying so again on every registration is noise."""
    assert _register(c, np.ndarray, lambda a: "v1", override=True) == []


def test_override_really_replaces_the_content_hash(c):
    """The consequence, stated openly rather than left implicit.

    Two arrays that share nothing but the hasher's verdict share one entry.
    This is what the flag buys and what it costs; a test that avoided saying
    so would be pretending the trade is free.
    """
    _register(c, np.ndarray, lambda a: "same-for-everything", override=True)

    @c.cache(assume_safe=True)
    def f(arr):
        return float(arr.sum())

    first = f(np.zeros((4, 4)))
    second = f(np.ones((4, 4)))
    assert first == second == 0.0, "the override did not reach the cache key"


def test_a_faithful_override_still_distinguishes_values(c):
    """The other half: an honest hasher keeps distinct values distinct."""
    _register(c, np.ndarray, lambda a: str(a.sum()), override=True)

    @c.cache(assume_safe=True)
    def f(arr):
        return float(arr.sum())

    assert f(np.zeros((4, 4))) == 0.0
    assert f(np.ones((4, 4))) == 16.0


def test_editing_an_overriding_hasher_invalidates(c):
    """The hasher's own source is part of the key, on this path too."""
    _register(c, np.ndarray, lambda a: "v1", override=True)

    @c.cache(assume_safe=True)
    def f(arr):
        return arr.sum()

    f(np.zeros((4, 4)))
    hits_before = f.cache_info()["hits"]
    f(np.zeros((4, 4)))
    assert f.cache_info()["hits"] > hits_before

    # Same returned value, different hasher body: the entry must not be reused.
    _register(c, np.ndarray, lambda a: "v1" + "", override=True)
    hits = f.cache_info()["hits"]
    f(np.zeros((4, 4)))
    assert f.cache_info()["hits"] == hits, "a different hasher reused the old entry"


def test_override_beats_a_lineage_hash(c):
    """Documented ordering: overriding hashers come before everything.

    A notebook value short-circuits to its lineage hash ahead of ordinary
    registered hashers, so this is the one place the two rules could disagree.
    """
    calls = []
    _register(c, Tagged, lambda t: calls.append(1) or str(t.payload), override=True)

    @c.cache(assume_safe=True)
    def f(value):
        return value.payload

    f(Tagged(1, lineage="lineage-abc"))
    assert calls, "the lineage hash short-circuited an overriding hasher"


# ---------------------------------------------------------------------------
# One registration per type
# ---------------------------------------------------------------------------


def test_re_registering_without_override_drops_the_override(c):
    """Otherwise the old entry keeps winning from the other registry."""
    _register(c, np.ndarray, lambda a: "override", override=True)
    calls = []
    _register(c, np.ndarray, lambda a: calls.append(1) or "plain")

    @c.cache(assume_safe=True)
    def f(arr):
        return arr[0, 0]

    f(np.zeros((4, 4)))
    assert calls == [], "the discarded override was still being consulted"


def test_re_registering_with_override_replaces_the_plain_one(c):
    _register(c, np.ndarray, lambda a: "plain")
    calls = []
    _register(c, np.ndarray, lambda a: calls.append(1) or "override", override=True)

    @c.cache(assume_safe=True)
    def f(arr):
        return arr[0, 0]

    f(np.zeros((4, 4)))
    assert calls, "the plain registration shadowed the new override"


def test_a_late_registration_is_not_shadowed_by_the_memo(c):
    """The arg-hash memo caches whichever hasher was in effect when it filled.

    Registering after a value has already been hashed once has to invalidate
    that, or the new hasher silently does nothing for exactly the objects the
    user has been passing all along.
    """
    @c.cache(assume_safe=True)
    def f(value):
        return value.payload

    value = Tagged(1, lineage="lineage-abc")
    f(value)

    calls = []
    _register(c, Tagged, lambda t: calls.append(1) or str(t.payload), override=True)
    f(value)
    assert calls, "the memo served a hash computed before the registration"


# ---------------------------------------------------------------------------
# The advice that sent someone to a dead end
# ---------------------------------------------------------------------------


def test_the_effectiveness_warning_names_the_override(tmp_path):
    """It correctly diagnosed a big argument, then named an unusable remedy."""
    from cash.effectiveness import EffectivenessLedger

    ledger = EffectivenessLedger(waste_threshold_seconds=0.01)
    message = None
    for _ in range(5):
        message = ledger.record(
            "mod.f", overhead_seconds=1.0, body_seconds=0.0, was_hit=True,
        ) or message

    assert message is not None, "this workload should have been flagged"
    assert "register_hasher" in message
    assert "override=True" in message, (
        "the remedy is unreachable for numpy/pandas arguments without it, "
        "which is the case this warning fires on most"
    )
