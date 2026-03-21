import pytest
import time
import sys
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from cash.notebook.magics import CashMagics
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable


class MockShell(Configurable):
    '''Mock IPython shell for testing.'''
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []


@pytest.fixture
def perf_magics():
    '''Provide CashMagics instance for performance testing.'''
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    
    yield magics
    
    # Cleanup
    backend.clear()


def test_memory_calculation_dataframe(perf_magics):
    '''Test that DataFrame memory calculation uses memory_usage().'''
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    variables = {'df': df}
    memory_size = perf_magics._calculate_memory_size(variables)
    
    # Should be fast and accurate
    expected_size = df.memory_usage(deep=True).sum()
    assert memory_size == expected_size, 'DataFrame memory size should match memory_usage()'


def test_memory_calculation_numpy_array(perf_magics):
    '''Test that NumPy array memory calculation uses nbytes.'''
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    
    variables = {'arr': arr}
    memory_size = perf_magics._calculate_memory_size(variables)
    
    assert memory_size == arr.nbytes, 'NumPy array memory size should match nbytes'


def test_memory_calculation_series(perf_magics):
    '''Test that pandas Series memory calculation uses memory_usage().'''
    series = pd.Series([1, 2, 3, 4, 5])
    
    variables = {'series': series}
    memory_size = perf_magics._calculate_memory_size(variables)
    
    expected_size = series.memory_usage(deep=True)
    assert memory_size == expected_size, 'Series memory size should match memory_usage()'


def test_memory_calculation_list(perf_magics):
    '''Test that list memory calculation uses recursive getsizeof.'''
    test_list = [1, 2, 3, 'hello', 'world']
    
    variables = {'test_list': test_list}
    memory_size = perf_magics._calculate_memory_size(variables)
    
    # Should be greater than shallow size
    shallow_size = sys.getsizeof(test_list)
    assert memory_size > shallow_size, 'Recursive size should be greater than shallow size'


def test_memory_calculation_dict(perf_magics):
    '''Test that dict memory calculation uses recursive getsizeof.'''
    test_dict = {'a': 1, 'b': [1, 2, 3], 'c': 'hello'}
    
    variables = {'test_dict': test_dict}
    memory_size = perf_magics._calculate_memory_size(variables)
    
    shallow_size = sys.getsizeof(test_dict)
    assert memory_size > shallow_size, 'Recursive size should include nested structures'


def test_memory_calculation_mixed_types(perf_magics):
    '''Test memory calculation with mixed types.'''
    df = pd.DataFrame({'a': [1, 2, 3]})
    arr = np.array([1, 2, 3])
    list_data = [1, 2, 3]
    
    variables = {
        'df': df,
        'arr': arr,
        'list_data': list_data
    }
    
    total_size = perf_magics._calculate_memory_size(variables)
    
    # Should be sum of individual sizes
    df_size = df.memory_usage(deep=True).sum()
    arr_size = arr.nbytes
    
    assert total_size > df_size + arr_size, 'Total size should account for all variables'


def test_memory_calculation_performance(perf_magics):
    '''Test that memory calculation is fast (< 100ms for large DataFrame).'''
    # Create a moderately large DataFrame
    df = pd.DataFrame({
        'a': np.random.rand(100000),
        'b': np.random.rand(100000),
        'c': ['text'] * 100000
    })
    
    variables = {'df': df}
    
    start_time = time.time()
    memory_size = perf_magics._calculate_memory_size(variables)
    elapsed_time = time.time() - start_time
    
    # Should complete in well under 100ms (typically <10ms)
    assert elapsed_time < 0.1, f'Memory calculation took {elapsed_time*1000:.1f}ms, should be <100ms'
    
    # Verify it's accurate
    expected_size = df.memory_usage(deep=True).sum()
    assert memory_size == expected_size, 'Memory calculation should be accurate'


def test_recursive_getsizeof_circular_reference(perf_magics):
    '''Test that recursive getsizeof handles circular references.'''
    # Create circular reference
    list1 = [1, 2, 3]
    list2 = [4, 5, list1]
    list1.append(list2)
    
    # Should not crash or infinite loop
    size = perf_magics._recursive_getsizeof(list1)
    
    assert size > 0, 'Should handle circular references'


def test_memory_calculation_fallback(perf_magics):
    '''Test that unpicklable objects fall back gracefully.'''
    # Create a lambda (not picklable)
    func = lambda x: x * 2
    
    variables = {'func': func}
    
    # Should not crash
    memory_size = perf_magics._calculate_memory_size(variables)
    
    # Should at least return sys.getsizeof estimate
    assert memory_size > 0, 'Should handle unpicklable objects'
