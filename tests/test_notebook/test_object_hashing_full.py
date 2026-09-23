"""compute_hash_full — full-content hashing for key discrimination.

``compute_hash`` samples large objects; wherever the hash IS the cache-key
discriminator (per-iteration loop keys), a difference outside the sample
collided two keys. These tests pin that compute_hash_full sees every byte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cash.object_hashing import compute_hash, compute_hash_full


def test_ndarray_out_of_sample_difference_distinct():
    a = np.zeros(2000)
    b = np.zeros(2000)
    b[1000] = 5.0  # outside compute_hash's flat[:100] sample
    assert compute_hash(a) == compute_hash(b), "precondition: sampled hash collides"
    assert compute_hash_full(a) != compute_hash_full(b)


def test_dataframe_tail_difference_distinct():
    df1 = pd.DataFrame({"v": range(1000)})
    df2 = df1.copy()
    df2.iloc[999, 0] = -1  # outside compute_hash's head(5) sample
    assert compute_hash(df1) == compute_hash(df2), "precondition: sampled hash collides"
    assert compute_hash_full(df1) != compute_hash_full(df2)


def test_large_list_middle_difference_distinct():
    l1 = list(range(1000))
    l2 = list(range(1000))
    l2[500] = -1  # outside the head5/tail5 sample
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


# ---------------------------------------------------------------------------
# The schema reaches the notebook's keys, as it reaches the decorator's
# ---------------------------------------------------------------------------
# `compute_hash_full` builds the per-iteration loop keys. It hashed a frame by
# its values alone and an array by its C-order bytes, so values that are equal
# but are different objects to the code reading them keyed alike: the second
# iteration was served the first one's result. The decorator's hasher had long
# folded in dtypes and memory layout; both paths now use that one hasher.


def _loop_key(value):
    """The key a `for` iteration binding *value* to ``x`` gets."""
    from cash.notebook.control_structures.common import build_iteration_context, compute_context_hash

    return compute_context_hash(build_iteration_context(["x"], {"x": value}, None))


def _distinct_on_the_notebook_path(a, b):
    assert compute_hash_full(a) != compute_hash_full(b)
    assert _loop_key(a) != _loop_key(b)


def test_int64_and_nullable_int64_key_apart():
    """``Int64`` has pd.NA semantics; ``int64`` does not."""
    _distinct_on_the_notebook_path(pd.Series([1, 2, 3], dtype="int64"), pd.Series([1, 2, 3], dtype="Int64"))
    frame = pd.DataFrame({"a": [1, 2, 3]})
    _distinct_on_the_notebook_path(frame, frame.astype({"a": "Int64"}))


def test_tz_naive_and_tz_aware_timestamps_key_apart():
    """The same instants, one with a zone: ``tz_convert`` works on one and
    raises on the other."""
    naive = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D"))
    _distinct_on_the_notebook_path(naive, naive.dt.tz_localize("UTC"))
    _distinct_on_the_notebook_path(naive.to_frame("t"), naive.dt.tz_localize("UTC").to_frame("t"))


def test_c_and_fortran_ordered_arrays_key_apart():
    """Equal values, different memory order: ``np.ravel(x, order="A")`` reads
    them differently."""
    c_order = np.arange(12.0).reshape(3, 4)
    f_order = np.asfortranarray(c_order)
    assert np.array_equal(c_order, f_order), "precondition: equal values"
    _distinct_on_the_notebook_path(c_order, f_order)


def test_equal_values_with_equal_schema_still_share_a_key():
    """The control: the schema must add distinctions, not make every value unique."""
    assert _loop_key(pd.Series([1, 2, 3], dtype="Int64")) == _loop_key(pd.Series([1, 2, 3], dtype="Int64"))
    assert _loop_key(np.asfortranarray(np.ones((3, 4)))) == _loop_key(np.asfortranarray(np.ones((3, 4))))


def test_the_notebook_and_the_decorator_hash_a_frame_alike():
    """One hasher: the full hash of a library value IS the decorator's key for it."""
    from cash.object_hashing import builtin_hash

    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert compute_hash_full(frame) == builtin_hash(frame)
    arr = np.arange(6).reshape(2, 3).T
    assert compute_hash_full(arr) == builtin_hash(arr)
