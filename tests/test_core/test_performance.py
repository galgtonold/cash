"""Performance tests for `cash.notebook.object_hashing.estimate_object_size`.

Originally tested `CashMagics._calculate_memory_size` (extracted to
`object_hashing.calculate_memory_size` in step 1 of the magics deepening,
then consolidated with the more sophisticated `_estimate_object_size`
implementation that was duplicated inside `StatementProcessor`).

Single canonical sizer; tests call the module function directly.
"""
import sys
import time

import numpy as np
import pandas as pd

from cash.notebook.object_hashing import estimate_object_size


def test_dataframe_uses_memory_usage():
    """DataFrame size matches `memory_usage(deep=True).sum()`."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert estimate_object_size(df) == int(df.memory_usage(deep=True).sum())


def test_numpy_array_uses_nbytes():
    """NumPy array size matches `nbytes`."""
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    assert estimate_object_size(arr) == arr.nbytes


def test_series_uses_memory_usage():
    """pandas Series size matches `memory_usage(deep=True)`."""
    series = pd.Series([1, 2, 3, 4, 5])
    assert estimate_object_size(series) == int(series.memory_usage(deep=True))


def test_list_size_grows_with_elements():
    """List recursive size exceeds shallow size."""
    test_list = [1, 2, 3, 'hello', 'world']
    assert estimate_object_size(test_list) > sys.getsizeof(test_list)


def test_dict_size_grows_with_pairs():
    """Dict size accounts for nested structures."""
    test_dict = {'a': 1, 'b': [1, 2, 3], 'c': 'hello'}
    assert estimate_object_size(test_dict) > sys.getsizeof(test_dict)


def test_mixed_types_sum():
    """Per-object sizing composes for a dict of disparate values."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    arr = np.array([1, 2, 3])
    list_data = [1, 2, 3]

    total = (
        estimate_object_size(df)
        + estimate_object_size(arr)
        + estimate_object_size(list_data)
    )
    df_size = int(df.memory_usage(deep=True).sum())
    arr_size = arr.nbytes
    assert total > df_size + arr_size, 'Total size accounts for all variables'


def test_large_dataframe_fast():
    """Sizing a 100k-row DataFrame stays under 100ms."""
    df = pd.DataFrame({
        'a': np.random.rand(100000),
        'b': np.random.rand(100000),
        'c': ['text'] * 100000,
    })

    start = time.time()
    size = estimate_object_size(df)
    elapsed = time.time() - start

    assert elapsed < 0.1, f'Sizing took {elapsed*1000:.1f}ms, should be <100ms'
    assert size == int(df.memory_usage(deep=True).sum())


def test_circular_reference_bounded():
    """Circular references are bounded by the depth cap, not a seen-set."""
    list1 = [1, 2, 3]
    list2 = [4, 5, list1]
    list1.append(list2)

    assert estimate_object_size(list1) > 0  # must not hang or recurse infinitely


def test_unpicklable_object():
    """Lambdas (and other unpicklable values) still produce a sensible size."""
    func = lambda x: x * 2  # noqa: E731
    assert estimate_object_size(func) > 0
