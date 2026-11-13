"""``cache(func)`` of a function already decorated the same way hands back
the wrapper it made then.

Libraries that cache through a factory (``cache(func, memory)(args)``, the
joblib ``Memory.cache`` shape) decorate the same function on every call.
Registered anew each time, the function lost its statistics to a fresh
registration, re-ran the decoration's checks, and, counted as a new
function, dropped every remembered fast hit in the process.
"""

from __future__ import annotations

import warnings

import pytest

RUNS = {"n": 0}


def double(x):
    RUNS["n"] += 1
    return x * 2


def triple(x):
    return x * 3


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_the_same_function_and_options_give_the_same_wrapper(disk_cash):
    first = disk_cash.cache(double)
    assert disk_cash.cache(double) is first
    assert disk_cash.cache(ttl=60)(double) is not first


def test_calls_through_each_decoration_count_together(disk_cash):
    disk_cash.cache(double)(1)
    disk_cash.cache(double)(1)
    disk_cash.cache(double)(1)
    info = disk_cash.cache(double).cache_info()
    assert (info["hits"], info["misses"]) == (2, 1)


def test_other_options_or_new_code_register_anew(disk_cash, monkeypatch):
    """The positive control: what would change the key still does."""
    plain = disk_cash.cache(double)
    assert disk_cash.cache(ignore="x")(double) is not plain

    runs = RUNS["n"]
    again = disk_cash.cache(double)
    assert again(4) == 8
    monkeypatch.setattr(double, "__code__", triple.__code__)
    swapped = disk_cash.cache(double)
    assert swapped is not again
    assert swapped(4) == 12
    assert RUNS["n"] >= runs
