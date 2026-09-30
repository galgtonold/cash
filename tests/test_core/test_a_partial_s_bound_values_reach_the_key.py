"""What a cached ``functools.partial`` binds is keyed per call, by content.

``c.cache(partial(total, arr))`` keyed the bound array by its ``repr``, taken
once at decoration: numpy elides the middle of a long array, so two arrays
differing there shared one entry, and an ordinary object's ``repr`` holds its
address, so a change to it was never seen.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from cash import Cash, FileBackend
from cash.exceptions import CashCacheIneffectiveWarning


@pytest.fixture
def c(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def total(arr, x):
    return float(arr.sum()) + x


class Model:
    def __init__(self, k):
        self.k = k


def scaled(model, x):
    return model.k * x


def test_arrays_differing_in_their_elided_middle_key_apart(c):
    a = np.zeros(5000)
    b = np.zeros(5000)
    b[2500] = 100.0
    assert repr(a) == repr(b)
    assert c.cache(functools.partial(total, a))(1) == 1.0
    assert c.cache(functools.partial(total, b))(1) == 101.0


def test_a_bound_array_changed_in_place_is_seen(c):
    a = np.zeros(10)
    g = c.cache(functools.partial(total, a))
    assert g(1) == 1.0
    a[3] = 5.0
    assert g(1) == 6.0


def test_a_bound_object_changed_in_place_is_seen(c):
    m = Model(3)
    g = c.cache(functools.partial(scaled, m))
    assert g(2) == 6
    m.k = 7
    assert g(2) == 14


def test_the_namespace_does_not_hold_an_address():
    first, second = Model(3), Model(3)  # both alive: two addresses
    one = Cash.get_func_key(functools.partial(scaled, first))
    two = Cash.get_func_key(functools.partial(scaled, second))
    assert one == two


def test_an_unhashable_bound_value_runs_uncached_with_a_warning(c):
    import threading

    lock = threading.Lock()

    def locked(held, x):
        return x + 1

    g = c.cache(functools.partial(locked, lock))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert g(1) == 2
    messages = [str(w.message) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert any("KEY-UNHASHABLE-ARG" in m and "partial binds" in m for m in messages), messages
