"""CAS-86: compute_hash_full — full-content hashing for key discrimination.

``compute_hash`` samples large objects; wherever the hash IS the cache-key
discriminator (per-iteration loop keys), a difference outside the sample
collided two keys. These tests pin that compute_hash_full sees every byte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.notebook.object_hashing import compute_hash, compute_hash_full


def test_ndarray_out_of_sample_difference_distinct():
    a = np.zeros(2000)
    b = np.zeros(2000)
    b[1000] = 5.0                       # outside compute_hash's flat[:100] sample
    assert compute_hash(a) == compute_hash(b), "precondition: sampled hash collides"
    assert compute_hash_full(a) != compute_hash_full(b)


def test_dataframe_tail_difference_distinct():
    df1 = pd.DataFrame({"v": range(1000)})
    df2 = df1.copy()
    df2.iloc[999, 0] = -1               # outside compute_hash's head(5) sample
    assert compute_hash(df1) == compute_hash(df2), "precondition: sampled hash collides"
    assert compute_hash_full(df1) != compute_hash_full(df2)


def test_large_list_middle_difference_distinct():
    l1 = list(range(1000))
    l2 = list(range(1000))
    l2[500] = -1                        # outside the head5/tail5 sample
    assert compute_hash(l1) == compute_hash(l2), "precondition: sampled hash collides"
    assert compute_hash_full(l1) != compute_hash_full(l2)


def test_equal_objects_hash_equal():
    a = np.arange(500.0)
    assert compute_hash_full(a) == compute_hash_full(np.arange(500.0))
    df = pd.DataFrame({"v": [1, 2]})
    assert compute_hash_full(df) == compute_hash_full(df.copy())


def test_object_dtype_array_uses_content_not_pointers():
    a1 = np.array([{"k": 1}, {"k": 22}], dtype=object)
    a2 = np.array([{"k": 1}, {"k": 22}], dtype=object)
    assert compute_hash_full(a1) == compute_hash_full(a2)
    a3 = np.array([{"k": 999}, {"k": 22}], dtype=object)
    assert compute_hash_full(a1) != compute_hash_full(a3)


def test_noncontiguous_array_hashable_and_content_true():
    base = np.arange(1000.0)
    view = base[::2]
    assert compute_hash_full(view) == compute_hash_full(np.arange(1000.0)[::2])


def test_unpicklable_falls_back_gracefully():
    import threading
    h = compute_hash_full(threading.Lock())
    assert isinstance(h, str) and len(h) == 64
