"""A result holding a function whose closure holds another part of it is
not cached (STORE-PARTS-SHARED).

``return arr, lambda: arr`` from a module: the function cannot be pickled,
so the RAM tier kept it by reference and copied ``arr``. On a hit the
function returned the array it closed over on the first run, not the one
the call handed back, and a write to one no longer reached the other.
"""

from __future__ import annotations

import warnings

import numpy as np

from cash import Cash

_SHARED = {}


def _make():
    arr = np.zeros(10)
    return arr, (lambda: arr)


def _plain():
    return np.zeros(10), (lambda: 1.0)


def test_the_result_is_returned_whole_and_not_cached(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    make = c.cache(_make)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            arr, getter = make()
            assert getter() is arr
    assert "STORE-PARTS-SHARED" in [getattr(w.message, "code", None) for w in caught]


def test_a_function_closing_over_nothing_of_the_result_is_cached(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), backend="memory", register_magic=False)
    plain = c.cache(_plain)
    plain()
    plain()
    assert plain.cache_info()["hits"] == 1
