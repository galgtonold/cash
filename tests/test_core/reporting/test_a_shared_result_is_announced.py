"""A result that shares state with something the caller holds says so.

Found while stress-testing the decorator: on the computing run these
behave like plain Python, and on a HIT they quietly stop:

* ``return base[lo:hi]`` -- the numpy view stops sharing memory with ``base``,
* ``return Wrapper(rows)`` -- the result no longer holds the caller's list,
* ``return CONFIG`` -- the caller's write no longer reaches the module global.

A hit hands back a value rebuilt from the stored bytes, so this is what caching
such a result means. Cash cannot tell whether the caller relies on the sharing,
so it says what will differ and lets the user waive it (``assume_safe=True``).
"""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")


CONFIG: dict = {}


def _warnings_of(fn, *args):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fn(*args)
    return [str(w.message) for w in seen if "CACHE-RESULT-SHARED" in str(w.message)]


def test_a_view_of_an_argument_is_announced(cash_instance):
    @cash_instance.cache
    def window(base, lo, hi):
        return base[lo:hi]

    said = _warnings_of(window, np.arange(5), 1, 4)
    assert said, "no warning for a result sharing memory with an argument"
    assert "base" in said[0], said[0]


def test_a_result_holding_an_argument_is_announced(cash_instance):
    @cash_instance.cache
    def wrap(rows, tag):
        return {"data": rows, "tag": tag}

    assert _warnings_of(wrap, [1, 2, 3], "t")


def test_a_returned_module_global_is_announced(cash_instance):
    @cash_instance.cache
    def config(n):
        return CONFIG

    assert _warnings_of(config, 1)


def test_assume_safe_waives_it(cash_instance):
    @cash_instance.cache(assume_safe=True)
    def window(base, lo, hi):
        return base[lo:hi]

    assert not _warnings_of(window, np.arange(5), 1, 4)


def test_an_independent_result_says_nothing(cash_instance):
    @cash_instance.cache
    def copied(base, lo, hi):
        return base[lo:hi].copy()

    @cash_instance.cache
    def rebuilt(rows, tag):
        return {"data": list(rows), "tag": tag}

    @cash_instance.cache
    def snapshot(n):
        return dict(CONFIG)

    assert not _warnings_of(copied, np.arange(5), 1, 4)
    assert not _warnings_of(rebuilt, [1, 2, 3], "t")
    assert not _warnings_of(snapshot, 1)


ROWS: list = [1, 2]


def test_a_result_holding_a_module_global_is_announced(cash_instance):
    # `{'rows': ROWS}`: plain Python hands back the caller's own list, and an
    # edit to ROWS shows through the result; a hit's copy does not.
    @cash_instance.cache
    def wrap(n):
        return {"rows": ROWS}

    said = _warnings_of(wrap, 1)
    assert said, "no warning for a result holding a module global"
    assert "ROWS" in said[0], said[0]


def test_a_result_holding_a_module_global_in_an_attribute_is_announced(cash_instance):
    import argparse

    @cash_instance.cache
    def wrap(n):
        return argparse.Namespace(rows=ROWS)

    assert _warnings_of(wrap, 1)


def test_a_closure_variable_returned_or_held_is_announced(cash_instance):
    box = [1]

    @cash_instance.cache
    def held(n):
        return {"rows": box}

    @cash_instance.cache
    def returned(n):
        return box

    said = _warnings_of(held, 1)
    assert said and "box" in said[0], said
    assert _warnings_of(returned, 1)


def test_a_held_sentinel_global_says_nothing(cash_instance):
    # Handed back as itself on a hit (`identity_refs`): nothing differs.
    @cash_instance.cache
    def lookup(n):
        return {"rows": _MARK}

    assert not _warnings_of(lookup, 1)


_MARK = object()
