"""Tests for `cash.notebook.object_hashing.compute_hash` — hash dispatch table.

Migrated from `test_magics_coverage.py` when the functions moved out of
`CashMagics` into their own module. The companion size-estimator
(``estimate_object_size``) is exercised by ``test_size_estimator.py``.
"""
import pytest

from cash.notebook.object_hashing import compute_hash


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

