"""Correctness of argument hashing for large numpy/pyarrow values.

Large arrays/tables used to be hashed by a SAMPLE (numpy: first/last 128
elements; pyarrow >=10MB: schema + row count only), so two different values
that matched the sample collided into a wrong cache hit - a silent
data-corruption bug for exactly the large arrays a data/ML cache targets.
These tests pin full-content hashing.
"""
from __future__ import annotations

import numpy as np
import pytest

from cash import Cash, FileBackend


def _counter_fn(c):
    calls = {"n": 0}

    @c.cache
    def f(arr):
        calls["n"] += 1
        return float(np.asarray(arr).sum())

    return f, calls


def test_large_numpy_middle_difference_does_not_collide(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    f, calls = _counter_fn(c)
    base = np.ones(4_000_000)              # 32 MB -> old sampling path
    a = f(base)
    other = base.copy()
    other[2_000_000] = 999_999.0           # differs only in the middle
    b = f(other)
    assert calls["n"] == 2, "different arrays must not share a cache entry"
    assert a == base.sum()
    assert b == other.sum()
    assert a != b


def test_numpy_reshape_does_not_collide(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def shape_of(a):
        return tuple(a.shape)

    assert shape_of(np.ones(4)) == (4,)
    assert shape_of(np.ones((2, 2))) == (2, 2)   # same bytes, different shape


def test_equal_numpy_arrays_still_hit(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    f, calls = _counter_fn(c)
    arr = np.arange(1000.0)
    f(arr)
    f(arr.copy())                          # equal value, fresh object -> must hit
    assert calls["n"] == 1


def test_numpy_contiguity_invariant(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def s(a):
        return float(a.sum())

    A = np.arange(20.0).reshape(4, 5)
    non_contig = np.ascontiguousarray(A.T).T   # same values, non-C-contiguous
    assert s.explain(A).cache_key == s.explain(non_contig).cache_key


def test_large_pyarrow_different_data_does_not_collide(tmp_path):
    pa = pytest.importorskip("pyarrow")
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    calls = {"n": 0}

    @c.cache
    def rows(t):
        calls["n"] += 1
        return t.num_rows

    n = 2_000_000
    t1 = pa.table({"a": np.zeros(n), "b": np.ones(n)})        # >10 MB
    t2 = pa.table({"a": np.zeros(n), "b": np.full(n, 2.0)})   # same schema/rows, diff data
    rows(t1)
    rows(t2)
    assert calls["n"] == 2, "tables with different data must not collide"
