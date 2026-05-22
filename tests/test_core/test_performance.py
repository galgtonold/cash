"""Performance tests for `cash.notebook.object_hashing.calculate_memory_size`.

Originally tested `CashMagics._calculate_memory_size`. Moved to module-level
function in `object_hashing.py`; tests now call the function directly without
needing an IPython shell fixture.
"""
import sys
import time

import numpy as np
import pandas as pd

from cash.notebook.object_hashing import _recursive_getsizeof, calculate_memory_size


def test_memory_calculation_dataframe():
    '''DataFrame memory calculation uses memory_usage().'''
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    memory_size = calculate_memory_size({'df': df})

    expected_size = df.memory_usage(deep=True).sum()
    assert memory_size == expected_size, 'DataFrame memory size should match memory_usage()'


def test_memory_calculation_numpy_array():
    '''NumPy array memory calculation uses nbytes.'''
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    memory_size = calculate_memory_size({'arr': arr})

    assert memory_size == arr.nbytes, 'NumPy array memory size should match nbytes'


def test_memory_calculation_series():
    '''pandas Series memory calculation uses memory_usage().'''
    series = pd.Series([1, 2, 3, 4, 5])
    memory_size = calculate_memory_size({'series': series})

    expected_size = series.memory_usage(deep=True)
    assert memory_size == expected_size, 'Series memory size should match memory_usage()'


def test_memory_calculation_list():
    '''List memory calculation uses recursive getsizeof.'''
    test_list = [1, 2, 3, 'hello', 'world']
    memory_size = calculate_memory_size({'test_list': test_list})

    shallow_size = sys.getsizeof(test_list)
    assert memory_size > shallow_size, 'Recursive size should be greater than shallow size'


def test_memory_calculation_dict():
    '''Dict memory calculation uses recursive getsizeof.'''
    test_dict = {'a': 1, 'b': [1, 2, 3], 'c': 'hello'}
    memory_size = calculate_memory_size({'test_dict': test_dict})

    shallow_size = sys.getsizeof(test_dict)
    assert memory_size > shallow_size, 'Recursive size should include nested structures'


def test_memory_calculation_mixed_types():
    '''Memory calculation with mixed types.'''
    df = pd.DataFrame({'a': [1, 2, 3]})
    arr = np.array([1, 2, 3])
    list_data = [1, 2, 3]

    total_size = calculate_memory_size({
        'df': df,
        'arr': arr,
        'list_data': list_data,
    })

    df_size = df.memory_usage(deep=True).sum()
    arr_size = arr.nbytes
    assert total_size > df_size + arr_size, 'Total size should account for all variables'


def test_memory_calculation_performance():
    '''Memory calculation is fast (< 100ms for large DataFrame).'''
    df = pd.DataFrame({
        'a': np.random.rand(100000),
        'b': np.random.rand(100000),
        'c': ['text'] * 100000,
    })

    start_time = time.time()
    memory_size = calculate_memory_size({'df': df})
    elapsed_time = time.time() - start_time

    assert elapsed_time < 0.1, f'Memory calculation took {elapsed_time*1000:.1f}ms, should be <100ms'

    expected_size = df.memory_usage(deep=True).sum()
    assert memory_size == expected_size, 'Memory calculation should be accurate'


def test_recursive_getsizeof_circular_reference():
    '''Recursive getsizeof handles circular references without infinite loop.'''
    list1 = [1, 2, 3]
    list2 = [4, 5, list1]
    list1.append(list2)

    size = _recursive_getsizeof(list1)
    assert size > 0, 'Should handle circular references'


def test_memory_calculation_fallback():
    '''Unpicklable objects fall back gracefully.'''
    func = lambda x: x * 2  # noqa: E731

    memory_size = calculate_memory_size({'func': func})
    assert memory_size > 0, 'Should handle unpicklable objects'
