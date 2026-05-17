"""
Unit tests for the decorator-notebook bridge features.

Tests the integration between @cash.cache decorator and notebook caching:
1. Decorator call logging and draining
2. Type hasher registry
3. Built-in type hashers for DataFrames, ndarrays, etc.
4. Badge rendering for decorator calls (condensed view)
5. Non-hashable parameter handling
"""

import pytest
import hashlib
import time
from unittest.mock import patch
from cash.core import Cash
from cash.notebook import badge_renderer as _badge


# ============================================================================
# A1: Decorator Call Logging
# ============================================================================


class TestDecoratorCallLogging:
    """Test that @cash.cache calls are logged for notebook integration."""

    def test_call_log_on_cache_miss(self):
        """First call (cache miss) should log the call."""
        c = Cash()

        @c.cache
        def compute(x):
            return x * 2

        compute(5)
        calls = c.drain_decorator_calls()

        assert len(calls) == 1
        assert calls[0]['func_name'] == Cash._get_func_key(compute)
        assert calls[0]['cache_hit'] is False
        assert calls[0]['execution_time'] > 0
        assert calls[0]['args_hash'] != ''
        assert calls[0]['cache_key'] != ''
        assert 'timestamp' in calls[0]

    def test_call_log_on_cache_hit(self):
        """Second call (cache hit) should log the call with cache_hit=True."""
        c = Cash()

        @c.cache
        def compute(x):
            return x * 2

        compute(5)
        c.drain_decorator_calls()  # Clear first call

        compute(5)  # Cache hit
        calls = c.drain_decorator_calls()

        assert len(calls) == 1
        assert calls[0]['cache_hit'] is True
        assert calls[0]['execution_time'] >= 0

    def test_drain_clears_log(self):
        """drain_decorator_calls should clear the log."""
        c = Cash()

        @c.cache
        def compute(x):
            return x * 2

        compute(5)
        calls1 = c.drain_decorator_calls()
        calls2 = c.drain_decorator_calls()

        assert len(calls1) == 1
        assert len(calls2) == 0

    def test_multiple_calls_logged(self):
        """Multiple calls should all be logged."""
        c = Cash()

        @c.cache
        def compute(x):
            return x * 2

        compute(1)
        compute(2)
        compute(3)
        compute(1)  # Cache hit

        calls = c.drain_decorator_calls()
        assert len(calls) == 4
        assert calls[0]['cache_hit'] is False  # miss
        assert calls[1]['cache_hit'] is False  # miss
        assert calls[2]['cache_hit'] is False  # miss
        assert calls[3]['cache_hit'] is True   # hit

    def test_different_functions_logged_separately(self):
        """Calls to different cached functions should be logged with correct func_name."""
        c = Cash()

        @c.cache
        def func_a(x):
            return x + 1

        @c.cache
        def func_b(x):
            return x * 2

        func_a(1)
        func_b(2)

        calls = c.drain_decorator_calls()
        assert len(calls) == 2
        assert 'func_a' in calls[0]['func_name']
        assert 'func_b' in calls[1]['func_name']

    def test_call_log_with_ttl(self):
        """TTL-decorated functions should also be logged."""
        c = Cash()

        @c.cache(ttl=3600)
        def compute(x):
            return x * 2

        compute(5)
        calls = c.drain_decorator_calls()

        assert len(calls) == 1
        assert calls[0]['cache_hit'] is False

    def test_call_log_when_serialization_fails(self):
        """When args can't be serialized, call should still be logged."""
        c = Cash()

        @c.cache
        def compute(x):
            return str(x)

        # Pass a generator (not picklable by default)
        gen = (i for i in range(5))
        compute(gen)

        calls = c.drain_decorator_calls()
        assert len(calls) == 1
        # Should show unhashable since generator can't be pickled
        assert calls[0]['cache_hit'] is False

    def test_call_log_preserves_return_value(self):
        """Call logging should not affect the return value."""
        c = Cash()

        @c.cache
        def compute(x):
            return x ** 2

        assert compute(5) == 25
        assert compute(5) == 25  # cache hit
        assert compute(3) == 9


# ============================================================================
# B1: Custom Type Hasher Registry
# ============================================================================


class TestTypeHasherRegistry:
    """Test the custom type hasher registry."""

    def test_register_hasher_basic(self):
        """Register a custom hasher and verify it's used."""
        c = Cash()

        class MyData:
            def __init__(self, value):
                self.value = value

        c.register_hasher(MyData, lambda d: hashlib.sha256(str(d.value).encode()).hexdigest())

        @c.cache
        def process(data):
            return data.value * 2

        d1 = MyData(5)
        result1 = process(d1)
        assert result1 == 10

        # Same value, should be cache hit
        d2 = MyData(5)
        result2 = process(d2)
        assert result2 == 10

        calls = c.drain_decorator_calls()
        assert len(calls) == 2
        assert calls[1]['cache_hit'] is True

    def test_register_hasher_different_values(self):
        """Different values of custom type should produce different hashes."""
        c = Cash()

        class MyData:
            def __init__(self, value):
                self.value = value

        c.register_hasher(MyData, lambda d: hashlib.sha256(str(d.value).encode()).hexdigest())

        @c.cache
        def process(data):
            return data.value * 2

        d1 = MyData(5)
        d2 = MyData(10)

        process(d1)
        process(d2)

        calls = c.drain_decorator_calls()
        assert len(calls) == 2
        assert calls[0]['cache_hit'] is False
        assert calls[1]['cache_hit'] is False
        # Different args hashes
        assert calls[0]['args_hash'] != calls[1]['args_hash']

    def test_register_hasher_via_global(self):
        """register_hasher should be accessible from the module level."""
        import cash
        assert hasattr(cash, 'register_hasher')
        assert callable(cash.register_hasher)

    def test_hasher_priority_cash_hash_first(self):
        """_cash_lineage_hash attribute should take priority over registered hasher."""
        c = Cash()

        class MyData:
            def __init__(self, value):
                self.value = value
                self._cash_lineage_hash = "fixed_hash"

        c.register_hasher(MyData, lambda d: hashlib.sha256(str(d.value).encode()).hexdigest())

        @c.cache
        def process(data):
            return data.value * 2

        # Two different values but same _cash_lineage_hash
        d1 = MyData(5)
        d2 = MyData(10)

        process(d1)
        process(d2)

        calls = c.drain_decorator_calls()
        assert len(calls) == 2
        # Second call should be a hit because _cash_lineage_hash is the same
        assert calls[1]['cache_hit'] is True


# ============================================================================
# B2: Built-in Type Hashers
# ============================================================================


class TestBuiltinTypeHashers:
    """Test built-in hashers for common data types."""

    def test_numpy_array_small(self):
        """Small numpy arrays should be hashed via tobytes()."""
        pytest.importorskip("numpy")
        import numpy as np

        c = Cash()

        @c.cache
        def process(arr):
            return arr.sum()

        arr1 = np.array([1, 2, 3])
        result1 = process(arr1)
        assert result1 == 6

        arr2 = np.array([1, 2, 3])  # Same content
        result2 = process(arr2)
        assert result2 == 6

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True

    def test_numpy_array_different(self):
        """Different numpy arrays should produce different hashes."""
        pytest.importorskip("numpy")
        import numpy as np

        c = Cash()

        @c.cache
        def process(arr):
            return arr.sum()

        process(np.array([1, 2, 3]))
        process(np.array([4, 5, 6]))

        calls = c.drain_decorator_calls()
        assert calls[0]['cache_hit'] is False
        assert calls[1]['cache_hit'] is False

    def test_pandas_dataframe(self):
        """pandas DataFrames should be hashed via hash_pandas_object."""
        pytest.importorskip("pandas")
        import pandas as pd

        c = Cash()

        @c.cache
        def process(df):
            return df.sum().to_dict()

        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result1 = process(df1)

        df2 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result2 = process(df2)

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True
        assert result1 == result2

    def test_pandas_series(self):
        """pandas Series should be hashed via hash_pandas_object."""
        pytest.importorskip("pandas")
        import pandas as pd

        c = Cash()

        @c.cache
        def process(s):
            return s.sum()

        s1 = pd.Series([1, 2, 3])
        result1 = process(s1)

        s2 = pd.Series([1, 2, 3])
        result2 = process(s2)

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True
        assert result1 == result2


# ============================================================================
# B3: Graceful Fallback for Non-Hashable Types
# ============================================================================


class TestNonHashableFallback:
    """Test graceful handling of non-hashable types."""

    def test_unhashable_type_still_executes(self):
        """Function with unhashable args should still execute (without caching)."""
        c = Cash()

        @c.cache
        def process(data):
            return list(data)

        # Generators are not picklable
        result = process(iter([1, 2, 3]))
        # The function should still work
        assert result == [1, 2, 3]

    def test_dict_args_work(self):
        """Dicts should be hashable via pickle."""
        c = Cash()

        @c.cache
        def process(d):
            return sum(d.values())

        result1 = process({'a': 1, 'b': 2})
        result2 = process({'a': 1, 'b': 2})

        calls = c.drain_decorator_calls()
        assert len(calls) == 2
        assert result1 == 3
        assert result2 == 3
        # Dicts are picklable so this should be a cache hit
        assert calls[1]['cache_hit'] is True

    def test_list_args_work(self):
        """Lists should be hashable via pickle."""
        c = Cash()

        @c.cache
        def process(lst):
            return sum(lst)

        result1 = process([1, 2, 3])
        result2 = process([1, 2, 3])

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True
        assert result1 == result2 == 6

    def test_mixed_args_with_non_hashable(self):
        """If one arg is hashable and one isn't, function should still execute."""
        c = Cash()

        @c.cache
        def process(x, gen):
            return x + next(gen)

        result = process(5, iter([10]))
        assert result == 15


# Decorator badge rendering is covered structurally by
# test_badge_html_renderer.test_decorator_section_renders_when_calls_present
# and end-to-end by test_badge_display. Phrasing assertions (e.g.
# "ALL N CACHED", "5 more calls") were tied to the deleted legacy
# render_decorator_calls implementation and have been retired with it.


# ============================================================================
# Lineage Attachment for Multiple DataFrame Types
# ============================================================================


class TestLineageAttachment:
    """Test lineage attachment for different DataFrame types."""

    def test_pandas_dataframe_lineage(self):
        """pandas DataFrame should get _cash_lineage_hash attached."""
        pytest.importorskip("pandas")
        import pandas as pd

        c = Cash()

        @c.cache
        def create_df():
            return pd.DataFrame({'a': [1, 2, 3]})

        df = create_df()
        assert hasattr(df, '_cash_lineage_hash')
        assert df._cash_lineage_hash != ''

    def test_pandas_series_lineage(self):
        """pandas Series should get _cash_lineage_hash attached."""
        pytest.importorskip("pandas")
        import pandas as pd

        c = Cash()

        @c.cache
        def create_series():
            return pd.Series([1, 2, 3])

        s = create_series()
        assert hasattr(s, '_cash_lineage_hash')

    def test_non_dataframe_no_lineage(self):
        """Non-DataFrame results should not have _cash_lineage_hash."""
        c = Cash()

        @c.cache
        def compute(x):
            return x * 2

        result = compute(5)
        assert not hasattr(result, '_cash_lineage_hash')


# ============================================================================
# C1: Source-Aware Cache Invalidation
# ============================================================================


class TestSourceAwareCacheInvalidation:
    """Test that @cash.cache detects function source changes."""

    def test_source_hash_changes_on_redefine(self):
        """Re-decorating a function with different body should produce different hash."""
        c = Cash()

        @c.cache
        def compute(x):
            return x * 2

        # Get the module-qualified key used internally
        func_key = Cash._get_func_key(compute)
        hash1 = c.source_hashes.get(func_key)
        assert hash1, f"Source hash should be non-empty, got: {hash1!r} (key: {func_key})"

        # Re-define with different body
        @c.cache
        def compute(x):
            return x * 3

        func_key2 = Cash._get_func_key(compute)
        hash2 = c.source_hashes.get(func_key2)
        assert hash2, f"Source hash should be non-empty, got: {hash2!r}"
        assert hash1 != hash2, "Different function bodies should produce different hashes"

    def test_cache_invalidates_on_redefine(self):
        """Re-defining a cached function should produce new results."""
        c = Cash()

        @c.cache
        def compute(x):
            return x * 2

        assert compute(5) == 10

        @c.cache
        def compute(x):
            return x * 3

        assert compute(5) == 15  # Should not return stale 10

    def test_bytecode_fallback_when_source_unavailable(self):
        """CodeAnalyzer.get_source_hash should use bytecode when inspect.getsource fails."""
        from cash.notebook.analysis import CodeAnalyzer

        def my_func(x):
            return x + 1

        # Normal source hash works
        h1 = CodeAnalyzer.get_source_hash(my_func)
        assert h1 != ""

        # Simulate getsource failure (e.g., IPython context with %cash_on)
        with patch('cash.notebook.analysis.inspect.getsource', side_effect=OSError):
            h2 = CodeAnalyzer.get_source_hash(my_func)
            assert h2 != "", "Should fall back to bytecode hash"

        # Different function body should produce different bytecode hash
        def my_func2(x):
            return x + 2

        with patch('cash.notebook.analysis.inspect.getsource', side_effect=OSError):
            h3 = CodeAnalyzer.get_source_hash(my_func2)
            assert h3 != ""
            assert h3 != h2, "Different bytecodes should produce different hashes"

    def test_function_tracker_bytecode_fallback(self):
        """FunctionTracker.get_function_source_hash should use bytecode fallback."""
        from cash.notebook.function_tracker import FunctionTracker

        ft = FunctionTracker()

        def my_func(x):
            return x * 10

        # Normal path
        h1 = ft.get_function_source_hash(my_func)
        assert h1 is not None

        # Clear cache and simulate getsource failure
        ft._source_cache.clear()
        with patch('cash.notebook.function_tracker.inspect.getsource', side_effect=OSError):
            h2 = ft.get_function_source_hash(my_func)
            assert h2 is not None, "Should fall back to bytecode hash"

    def test_bytecode_fallback_for_wrapped_function(self):
        """Bytecode fallback should follow __wrapped__ for functools.wraps wrappers."""
        import functools
        from cash.notebook.analysis import CodeAnalyzer

        def original(x):
            return x * 2

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return original(*args, **kwargs)

        with patch('cash.notebook.analysis.inspect.getsource', side_effect=OSError):
            h = CodeAnalyzer.get_source_hash(wrapper)
            assert h != "", "Should use __wrapped__.__code__ as fallback"


# ============================================================================
# C2: DataFrame Library Onboarding
# ============================================================================


class TestDataFrameLibraryOnboarding:
    """Test hashing and lineage attachment for multiple DataFrame libraries."""

    def test_polars_dataframe_hash(self):
        """polars DataFrame should be hashable via hash_rows."""
        pytest.importorskip("polars")
        import polars as pl

        c = Cash()

        @c.cache
        def process(df):
            return df.select(pl.col("a").sum()).item()

        df1 = pl.DataFrame({"a": [1, 2, 3]})
        result1 = process(df1)
        assert result1 == 6

        df2 = pl.DataFrame({"a": [1, 2, 3]})
        result2 = process(df2)
        assert result2 == 6

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True

    def test_polars_dataframe_different(self):
        """Different polars DataFrames should produce different hashes."""
        pytest.importorskip("polars")
        import polars as pl

        c = Cash()

        @c.cache
        def process(df):
            return df.select(pl.col("a").sum()).item()

        process(pl.DataFrame({"a": [1, 2, 3]}))
        process(pl.DataFrame({"a": [4, 5, 6]}))

        calls = c.drain_decorator_calls()
        assert calls[0]['cache_hit'] is False
        assert calls[1]['cache_hit'] is False

    def test_polars_series_hash(self):
        """polars Series should be hashable."""
        pytest.importorskip("polars")
        import polars as pl

        c = Cash()

        @c.cache
        def process(s):
            return s.sum()

        s1 = pl.Series("x", [10, 20, 30])
        result1 = process(s1)
        assert result1 == 60

        s2 = pl.Series("x", [10, 20, 30])
        process(s2)

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True

    def test_polars_lazyframe_hash(self):
        """polars LazyFrame should be hashable via explain()."""
        pytest.importorskip("polars")
        import polars as pl

        c = Cash()

        @c.cache
        def get_plan(lf):
            return lf.explain()

        lf1 = pl.DataFrame({"a": [1, 2]}).lazy().filter(pl.col("a") > 0)
        get_plan(lf1)

        lf2 = pl.DataFrame({"a": [1, 2]}).lazy().filter(pl.col("a") > 0)
        get_plan(lf2)

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True

    def test_polars_dataframe_lineage(self):
        """polars DataFrame result should get _cash_lineage_hash attached."""
        pytest.importorskip("polars")
        import polars as pl

        c = Cash()

        @c.cache
        def create_df():
            return pl.DataFrame({"a": [1, 2, 3]})

        df = create_df()
        assert hasattr(df, '_cash_lineage_hash')

    def test_pyarrow_table_hash(self):
        """PyArrow Table should be hashable."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        c = Cash()

        @c.cache
        def process(table):
            return table.num_rows

        t1 = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
        result1 = process(t1)
        assert result1 == 3

        t2 = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
        process(t2)

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True

    def test_pyarrow_different_tables(self):
        """Different PyArrow Tables should produce different hashes."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        c = Cash()

        @c.cache
        def process(table):
            return table.num_rows

        process(pa.table({"a": [1, 2]}))
        process(pa.table({"a": [3, 4, 5]}))

        calls = c.drain_decorator_calls()
        assert calls[0]['cache_hit'] is False
        assert calls[1]['cache_hit'] is False

    def test_builtin_type_hash_module_dispatch(self):
        """_try_builtin_type_hash should dispatch based on module, not just type name."""
        import pandas as pd

        # Ensure pandas DataFrame is detected via module
        df = pd.DataFrame({"a": [1]})
        h = Cash._try_builtin_type_hash(df)
        assert h is not None

    def test_builtin_type_hash_returns_none_for_plain_dict(self):
        """Plain dicts should fall through to pickle path."""
        h = Cash._try_builtin_type_hash({"key": "value"})
        assert h is None  # Dicts are handled by pickle

    def test_builtin_type_hash_returns_none_for_generator(self):
        """Generators should return None (cannot hash)."""
        gen = (x for x in range(10))
        h = Cash._try_builtin_type_hash(gen)
        assert h is None


# ============================================================================
# C3: File Dependency Tracking for Decorator
# ============================================================================


class TestDecoratorFileDependencies:
    """Test file_depends_on parameter for @cash.cache."""

    def test_file_depends_on_single(self, tmp_path):
        """file_depends_on with a single file path."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n3,4")

        c = Cash()

        @c.cache(file_depends_on=str(data_file))
        def load_data():
            return data_file.read_text()

        result1 = load_data()
        assert result1 == "a,b\n1,2\n3,4"

        # Same file, same content → cache hit
        load_data()
        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True

    def test_file_depends_on_invalidates_on_change(self, tmp_path):
        """Cache should invalidate when tracked file changes."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2")

        c = Cash()

        @c.cache(file_depends_on=str(data_file))
        def load_data():
            return data_file.read_text()

        result1 = load_data()
        assert "1,2" in result1

        # Modify the file
        import time
        time.sleep(0.05)  # Ensure mtime changes
        data_file.write_text("a,b\n5,6")

        result2 = load_data()
        # Should recompute because file changed
        assert "5,6" in result2

    def test_file_depends_on_multiple_files(self, tmp_path):
        """file_depends_on with multiple file paths."""
        f1 = tmp_path / "config.json"
        f2 = tmp_path / "data.csv"
        f1.write_text('{"key": "val"}')
        f2.write_text("a\n1")

        c = Cash()

        @c.cache(file_depends_on=[str(f1), str(f2)])
        def process():
            return f1.read_text() + f2.read_text()

        process()
        process()

        calls = c.drain_decorator_calls()
        assert calls[1]['cache_hit'] is True

    def test_file_depends_on_registers_datasource(self, tmp_path):
        """file_depends_on should register FileDataSource in the graph."""
        data_file = tmp_path / "test.txt"
        data_file.write_text("hello")

        c = Cash()

        @c.cache(file_depends_on=str(data_file))
        def read_file():
            return data_file.read_text()

        # Check that the file data source is registered
        assert len(c.data_sources) == 1
        ds_key = list(c.data_sources.keys())[0]
        assert ds_key.startswith("file:")

    def test_file_depends_on_with_ttl(self, tmp_path):
        """file_depends_on should work together with ttl."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("original")

        c = Cash()

        @c.cache(file_depends_on=str(data_file), ttl=3600)
        def read():
            return data_file.read_text()

        result = read()
        assert result == "original"

        calls = c.drain_decorator_calls()
        assert calls[0]['cache_hit'] is False


# ============================================================================
# C4: Thread Safety for Decorator Call Log
# ============================================================================


class TestDecoratorThreadSafety:
    """Test thread safety of decorator call logging."""

    def test_concurrent_call_logging(self):
        """Multiple threads calling decorated functions should not lose events."""
        import threading

        c = Cash()
        call_count = 0
        lock = threading.Lock()

        @c.cache
        def compute(x):
            nonlocal call_count
            with lock:
                call_count += 1
            return x * 2

        threads = []
        for i in range(20):
            t = threading.Thread(target=compute, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        calls = c.drain_decorator_calls()
        # All 20 calls should be logged (different args, so all cache misses)
        assert len(calls) == 20

    def test_drain_is_atomic(self):
        """drain_decorator_calls should clear the log atomically."""
        c = Cash()

        @c.cache
        def compute(x):
            return x

        for i in range(5):
            compute(i)

        calls = c.drain_decorator_calls()
        assert len(calls) == 5

        # Second drain should be empty
        calls2 = c.drain_decorator_calls()
        assert len(calls2) == 0


# ============================================================================
# C5: Decorator Cache Statistics & Introspection API
# ============================================================================


class TestDecoratorIntrospectionAPI:
    """Test cache_info() and cache_clear() on decorated functions."""

    def test_cache_info_basic(self):
        """cache_info() should return hit/miss stats."""
        c = Cash()

        @c.cache
        def compute(x):
            return x * 2

        compute(1)  # miss
        compute(1)  # hit
        compute(2)  # miss
        compute(2)  # hit
        compute(2)  # hit

        info = compute.cache_info()
        assert info['hits'] == 3
        assert info['misses'] == 2
        assert info['hit_rate'] == 3 / 5

    def test_cache_info_initial(self):
        """cache_info() should be zero before any calls."""
        c = Cash()

        @c.cache
        def compute(x):
            return x

        info = compute.cache_info()
        assert info['hits'] == 0
        assert info['misses'] == 0
        assert info['hit_rate'] == 0.0

    def test_cache_clear_resets_stats(self):
        """cache_clear() should reset hit/miss stats."""
        c = Cash()

        @c.cache
        def compute(x):
            return x

        compute(1)
        compute(1)

        info = compute.cache_info()
        assert info['hits'] == 1

        compute.cache_clear()

        info2 = compute.cache_info()
        assert info2['hits'] == 0
        assert info2['misses'] == 0

    def test_cache_info_has_total_time_saved(self):
        """cache_info() should track total time saved by cache hits."""
        c = Cash()

        @c.cache
        def compute(x):
            return x

        compute(1)
        compute(1)  # hit

        info = compute.cache_info()
        assert 'total_time_saved' in info
        assert info['total_time_saved'] >= 0

    def test_wrapped_attribute(self):
        """Decorated function should have __wrapped__ pointing to original."""
        c = Cash()

        @c.cache
        def my_func(x):
            return x

        assert hasattr(my_func, '__wrapped__')
        assert my_func.__wrapped__.__name__ == 'my_func'

    def test_functools_wraps_preserved(self):
        """Decorated function should preserve name and docstring."""
        c = Cash()

        @c.cache
        def my_documented_func(x):
            """This is the docstring."""
            return x

        assert my_documented_func.__name__ == 'my_documented_func'
        assert my_documented_func.__doc__ == "This is the docstring."
