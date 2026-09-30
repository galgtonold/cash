"""``__cash_key__``: a class names what identifies its instances in a key.

A class holding large frames, keyed by content, costs a full read of them on
every call, even for a method that returns their row count. With
``__cash_key__`` the key uses what the method returns instead, as ``self``,
as an argument and inside a container, and a background check warns when one
key stands for two different contents.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cash

pytestmark = pytest.mark.core


class Store:
    """Big data, identified by a version the owner bumps."""

    def __init__(self, version, data):
        self.version = version
        self.data = np.asarray(data, dtype=float)

    def __cash_key__(self):
        return ("store", self.version)


class Plain(Store):
    """The same data with the key switched off: keyed by content again."""

    __cash_key__ = None


class Looping:
    def __cash_key__(self):
        return self


class Raising:
    def __cash_key__(self):
        raise KeyError("version")


class Registered(Store):
    pass


class Other:
    def __init__(self, version, data):
        self.version = version
        self.data = np.asarray(data, dtype=float)

    def __cash_key__(self):
        return ("store", self.version)


@pytest.fixture
def c(tmp_path):
    return cash.Cash(cache_dir=str(tmp_path / "cache"))


def _codes(rec):
    return [getattr(w.message, "code", None) for w in rec]


def test_the_key_names_the_object_not_its_data(c):
    """Same key, other data: served the first result (the user's promise).
    Another key: recomputed."""

    @c.cache
    def total(store):
        return float(store.data.sum())

    assert total(Store(1, [1, 2])) == 3.0
    assert total(Store(1, [10, 20])) == 3.0
    assert total(Store(2, [10, 20])) == 30.0

    @c.cache
    def name(klass):
        return klass.__name__

    # A class passed as a value is keyed as code, not by the method it
    # defines for its instances.
    assert name(Store) == "Store"
    assert name(Other) == "Other"


def test_self_in_a_method_is_keyed_by_it(c):
    class Holder(Store):
        @c.cache
        def total(self):
            return float(self.data.sum())

    assert Holder(1, [1, 2]).total() == 3.0
    assert Holder(1, [5, 5]).total() == 3.0
    assert Holder(2, [5, 5]).total() == 10.0


def test_inside_a_container_it_is_keyed_by_it(c):
    @c.cache
    def totals(stores):
        return [float(s.data.sum()) for s in stores]

    assert totals([Store(1, [1]), Store(2, [2])]) == [1.0, 2.0]
    assert totals([Store(1, [9]), Store(2, [9])]) == [1.0, 2.0]
    assert totals([Store(1, [9]), Store(3, [9])]) == [9.0, 9.0]


def test_two_classes_returning_one_key_do_not_share_entries(c):
    @c.cache
    def total(store):
        return float(store.data.sum())

    assert total(Store(1, [1])) == 1.0
    assert total(Store(1, [5])) == 1.0
    assert total(Other(1, [7])) == 7.0


def test_none_on_a_subclass_turns_it_off(c):
    @c.cache
    def total(store):
        return float(store.data.sum())

    assert total(Store(1, [1])) == 1.0
    assert total(Store(1, [4])) == 1.0
    assert total(Plain(1, [1])) == 1.0
    assert total(Plain(1, [4])) == 4.0


def test_a_registered_hasher_wins(c):
    c.register_hasher(Registered, lambda r: str(r.data.tolist()))

    @c.cache
    def total(store):
        return float(store.data.sum())

    assert total(Store(1, [1])) == 1.0
    assert total(Store(1, [6])) == 1.0
    assert total(Registered(1, [1])) == 1.0
    assert total(Registered(1, [6])) == 6.0


@pytest.mark.parametrize("value", [Looping(), Raising()], ids=["returns-itself", "raises"])
def test_a_broken_key_runs_uncached_and_names_the_method(c, value):
    runs = []

    @c.cache
    def f(x):
        runs.append(1)
        return 1

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f(value)
        f(value)
    assert len(runs) == 2
    messages = [str(w.message) for w in rec if getattr(w.message, "code", None) == "KEY-UNHASHABLE-ARG"]
    assert messages and "__cash_key__()" in messages[0]


def test_an_unhashable_own_class_is_pointed_at_cash_key(c):
    class Session:
        def __reduce__(self):
            raise TypeError("a live session cannot be pickled")

    @c.cache
    def f(s):
        return 1

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f(Session())
    messages = [str(w.message) for w in rec if getattr(w.message, "code", None) == "KEY-UNHASHABLE-ARG"]
    assert messages and "__cash_key__" in messages[0]


# ---------------------------------------------------------------------------
# The background check
# ---------------------------------------------------------------------------


def _run_checked(c, fn, *values):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for v in values:
            fn(v)
        c._args.key_check.wait()
    return _codes(rec)


def test_one_key_for_two_contents_warns(c):
    @c.cache
    def total(store):
        return float(store.data.sum())

    codes = _run_checked(c, total, Store(1, [1, 2]), Store(1, [3, 4]))
    assert codes.count("KEY-STALE-CASH-KEY") == 1


def test_one_key_for_equal_contents_is_quiet(c):
    @c.cache
    def total(store):
        return float(store.data.sum())

    codes = _run_checked(c, total, Store(1, [1, 2]), Store(1, [1, 2]), Store(2, [5]))
    assert "KEY-STALE-CASH-KEY" not in codes


def test_the_check_remembers_across_processes(tmp_path):
    """A later run with the same key over other data warns: the record is
    beside the cache, not in the process."""

    def make():
        app = cash.Cash(cache_dir=str(tmp_path / "cache"))

        @app.cache
        def total(store):
            return float(store.data.sum())

        return app, total

    app, total = make()
    assert "KEY-STALE-CASH-KEY" not in _run_checked(app, total, Store(1, [1, 2]))
    app, total = make()
    assert "KEY-STALE-CASH-KEY" in _run_checked(app, total, Store(1, [8, 9]))


def test_the_check_can_be_turned_off(tmp_path):
    app = cash.Cash(cache_dir=str(tmp_path / "cache"), check_cash_keys=False)

    @app.cache
    def total(store):
        return float(store.data.sum())

    assert "KEY-STALE-CASH-KEY" not in _run_checked(app, total, Store(1, [1, 2]), Store(1, [3, 4]))
