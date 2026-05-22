"""Tests for `cash.notebook.object_hashing` — pure hashing/sizing utilities.

Migrated from `test_magics_coverage.py` when the functions moved out of
`CashMagics` into their own module. Tests no longer need an IPython shell
fixture; they exercise the module functions directly.
"""
import sys

import pytest

from cash.notebook.object_hashing import (
    _recursive_getsizeof,
    calculate_memory_size,
    compute_hash,
)


# ============================================================================
# compute_hash
# ============================================================================

class TestComputeHash:
    def test_hash_simple_objects(self):
        h1 = compute_hash(42)
        h2 = compute_hash(42)
        h3 = compute_hash(43)
        assert h1 == h2
        assert h1 != h3

    def test_hash_string(self):
        h = compute_hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_hash_list(self):
        h = compute_hash([1, 2, 3])
        assert isinstance(h, str)

    def test_hash_dict(self):
        h = compute_hash({"a": 1, "b": 2})
        assert isinstance(h, str)

    def test_hash_dataframe(self):
        """DataFrame should use fast shape+dtypes hashing."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        h = compute_hash(df)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_numpy_array(self):
        """ndarray should use shape+dtype hashing."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        arr = np.array([1, 2, 3, 4, 5])
        h = compute_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_series(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        s = pd.Series([1, 2, 3], name='test')
        h = compute_hash(s)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_unpicklable_object(self):
        """Unpicklable objects should fall back to id-based hash."""
        import threading
        lock = threading.Lock()
        h = compute_hash(lock)
        assert isinstance(h, str)


# ============================================================================
# calculate_memory_size
# ============================================================================

class TestCalculateMemorySize:
    def test_simple_types(self):
        size = calculate_memory_size({'x': 42, 'y': 'hello'})
        assert size > 0

    def test_dataframe_memory(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        df = pd.DataFrame({'a': range(1000), 'b': range(1000)})
        size = calculate_memory_size({'df': df})
        assert size > 1000

    def test_numpy_array_memory(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        arr = np.zeros(10000)
        size = calculate_memory_size({'arr': arr})
        assert size >= 80000  # 10000 * 8 bytes per float64

    def test_series_memory(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        s = pd.Series(range(1000))
        size = calculate_memory_size({'s': s})
        assert size > 0

    def test_empty_dict(self):
        size = calculate_memory_size({})
        assert size == 0

    def test_nested_containers(self):
        data = {'nested': {'a': [1, 2, 3], 'b': {'c': [4, 5]}}}
        size = calculate_memory_size(data)
        assert size > 0


# ============================================================================
# _recursive_getsizeof
# ============================================================================

class TestRecursiveGetsizeof:
    def test_simple_int(self):
        size = _recursive_getsizeof(42)
        assert size > 0

    def test_list_includes_elements(self):
        list_size = _recursive_getsizeof([1, 2, 3])
        int_size = _recursive_getsizeof(42)
        assert list_size > int_size

    def test_dict_includes_keys_and_values(self):
        size = _recursive_getsizeof({'key': 'value', 'key2': 'value2'})
        assert size > sys.getsizeof({})

    def test_handles_circular_reference(self):
        """Should handle circular references without infinite recursion."""
        lst = [1, 2, 3]
        lst.append(lst)
        size = _recursive_getsizeof(lst)
        assert size > 0
