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

* a bare registration on such a type is REJECTED with ``ValueError``, at the
  moment the user can still act on it. A warning was the first version and it
  is not enough: this is not suboptimal-but-working code, it is code with no
  configuration in which it does anything, and warnings scroll away;
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
# The rejection
# ---------------------------------------------------------------------------


def test_registering_on_a_content_hashed_type_raises(c):
    """A warning was the first version, and it is not enough.

    This is not a suboptimal-but-working registration -- there is no
    configuration in which it does anything. Dead code that the user clearly
    meant to be live is an error, and setup time is the safe place to say so:
    it fires once, deterministically, before any computation depends on it.
    Cash already reserves this shape for the untrackable-dependency case,
    which raises unless the user passes ``assume_safe=True``.
    """
    with pytest.raises(ValueError) as excinfo:
        c.register_hasher(np.ndarray, lambda a: "v1")
    text = str(excinfo.value)
    assert "numpy" in text
    assert "override=True" in text, "a refusal with no way forward is a wall"
    assert "drop the registration" in text, "removing it is the other remedy"


def test_the_rejection_names_the_collision_it_is_protecting_against(c):
    """The message has to say what override COSTS, not just how to pass it."""
    with pytest.raises(ValueError) as excinfo:
        c.register_hasher(np.ndarray, lambda a: "v1")
    assert "share one cache entry" in str(excinfo.value)


@pytest.mark.parametrize("family", ["numpy", "pandas"])
def test_every_content_hashed_family_is_rejected(c, family):
    """Not a numpy special case -- the rule is the whole built-in table."""
    if family == "numpy":
        types = [np.ndarray]
    else:
        pd = pytest.importorskip("pandas")
        types = [pd.DataFrame, pd.Series]
    for type_ in types:
        with pytest.raises(ValueError, match=family):
            c.register_hasher(type_, lambda v: "v1")


def test_a_rejected_registration_changes_nothing(c):
    """Raise before mutating, or a refusal quietly drops a good registration."""
    _register(c, np.ndarray, lambda a: "kept", override=True)
    with pytest.raises(ValueError):
        c.register_hasher(np.ndarray, lambda a: "rejected")

    @c.cache(assume_safe=True)
    def f(arr):
        return arr[0, 0]

    f(np.zeros((4, 4)))
    assert np.ndarray in c._override_hashers, "the refusal ate a valid override"


def test_registering_on_an_ordinary_type_is_accepted(c):
    """The control. A rule that rejects everything protects nothing."""
    assert _register(c, Tagged, lambda t: str(t.payload)) == []
    assert Tagged in c._type_hashers


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


def test_pyarrow_tables_are_claimed_too():
    pa = pytest.importorskip("pyarrow")
    assert cash.Cash.builtin_hashed_family(pa.Table) == "pyarrow"
    assert cash.Cash.builtin_hashed_family(pa.RecordBatch) == "pyarrow"


def test_a_subclass_in_your_own_module_is_still_yours(c):
    """The family check is by module PREFIX, and that is load-bearing.

    Cash claims ``numpy.ndarray`` itself, but a subclass defined in user code
    is dispatched by its own name and module -- so registering a plain hasher
    for it has always worked, and must keep working.
    """
    class MyArray(np.ndarray):
        pass

    assert cash.Cash.builtin_hashed_family(MyArray) is None
    assert _register(c, MyArray, lambda a: "mine") == []


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
    """Otherwise the old entry keeps winning from the other registry.

    Uses an ordinary type: a plain re-registration on a content-hashed one is
    rejected outright, so this rule can only be observed where both forms are
    legal.
    """
    _register(c, Tagged, lambda t: "override", override=True)
    calls = []
    _register(c, Tagged, lambda t: calls.append(1) or "plain")

    @c.cache(assume_safe=True)
    def f(value):
        return value.payload

    f(Tagged(1))
    assert calls, "the discarded override was still being consulted"


def test_re_registering_with_override_replaces_the_plain_one(c):
    _register(c, Tagged, lambda t: "plain")
    calls = []
    _register(c, Tagged, lambda t: calls.append(1) or "override", override=True)

    @c.cache(assume_safe=True)
    def f(value):
        return value.payload

    f(Tagged(1))
    assert calls, "the plain registration shadowed the new override"
    assert Tagged not in c._type_hashers


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
