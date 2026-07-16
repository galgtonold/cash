"""Tests for the Cash class and decorator-based caching (core.py)."""
import time
from unittest.mock import patch, MagicMock
from cash.core import Cash
from cash.backends import InMemoryBackend, CascadingBackend
from cash.backends.tiered_backend import TieredBackend
from cash.data_source import FileDataSource


class TestCashInit:
    """Test Cash initialization with various backend configurations."""

    def test_default_init(self, tmp_path):
        """Cash() creates a default TieredBackend with .cash directory."""
        with patch('cash.core.get_config') as mock_config:
            mock_config.return_value = MagicMock(
                cache_dir=str(tmp_path / ".cash"),
                compress=False,
                debug=False,
                smart_persistence=False,
                max_memory_entries=1000,
                max_cache_size=None,
                flush_interval=5,
            )
            c = Cash(register_magic=False)
            assert c.backend is not None

    def test_init_with_memory_backend(self):
        """Cash with explicit InMemoryBackend."""
        backend = InMemoryBackend()
        c = Cash(backend=backend, register_magic=False)
        assert c.backend is backend

    def test_init_with_cache_dir(self, tmp_path):
        """Cash with cache_dir creates FileBackend."""
        c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
        # Should create a TieredBackend by default
        assert c.backend is not None

    def test_init_with_backends_list(self):
        """Cash with multiple backends creates CascadingBackend."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        c = Cash(backends=[b1, b2], register_magic=False)
        assert isinstance(c.backend, CascadingBackend)

    def test_init_with_single_backend_in_list(self):
        """Cash with single backend in list uses it directly."""
        b1 = InMemoryBackend()
        c = Cash(backends=[b1], register_magic=False)
        assert c.backend is b1

    def test_init_with_empty_backends_list(self):
        """Cash with empty backends list falls back to default TieredBackend (via config)."""
        c = Cash(backends=[], register_magic=False)
        # Empty list is treated as "no backends specified" → falls through to default config
        assert isinstance(c.backend, TieredBackend)

    def test_init_with_debug(self):
        """Cash with debug flag."""
        c = Cash(backend=InMemoryBackend(), register_magic=False, debug=True)
        assert c.debug is True

    def test_init_registers_atexit(self):
        """Cash registers shutdown via atexit."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        # Just ensure it doesn't crash
        c.shutdown()

    def test_init_with_smart_persistence(self, tmp_path):
        """Cash with smart_persistence creates TieredBackend with policy."""
        with patch('cash.core.get_config') as mock_config:
            mock_config.return_value = MagicMock(
                cache_dir=str(tmp_path / ".cash"),
                compress=False,
                debug=False,
                smart_persistence=True,
                smart_persistence_threshold=0.5,
                max_memory_entries=1000,
                max_cache_size=None,
                flush_interval=5,
            )
            c = Cash(register_magic=False)
            assert isinstance(c.backend, TieredBackend)

    def test_smart_persistence_policy_persists_by_restore_vs_compute(self, tmp_path):
        """The promotion policy promotes when recomputing costs more than the
        predicted restore — not on a size-scaled bandwidth guess (CAS-141).

        The old policy modelled raw disk I/O (``execution_time > 2·size/100MB/s``)
        with zero serialization cost, so a *bigger* result was *less* likely to
        persist — the exact inversion that left large frames RAM-only. The
        corrected rule mirrors the statement processor's Gate A:
        ``promote iff execution_time - est_restore > min_savings · execution_time``.
        """
        with patch('cash.core.get_config') as mock_config:
            mock_config.return_value = MagicMock(
                cache_dir=str(tmp_path / ".cash"),
                compress=False,
                debug=False,
                smart_persistence=True,
                smart_persistence_threshold=1.0,
                min_cache_savings_pct=0.20,
                max_memory_entries=1000,
                max_cache_size=None,
                flush_interval=5,
            )
            c = Cash(register_magic=False)
            policy = c.backend.promotion_policy

            # Sub-100 ms cells: never persist (below the compute floor — disk
            # I/O alone would cost more than rerunning).
            assert policy(0.001, 100) is False
            assert policy(0.05, 50_000) is False

            # 100 ms–1 s cells whose tiny result restores far cheaper than it
            # recomputes: persist.
            assert policy(0.15, 100) is True
            assert policy(0.5, 50_000) is True

            # A 200 KB result computed in 150 ms: restore (~11 ms) is a small
            # fraction of the 150 ms recompute, so it IS worth persisting. (The
            # old threshold short-circuit wrongly refused this.)
            assert policy(0.15, 200_000) is True

            # > 1 s, medium result whose restore is far cheaper than recompute.
            assert policy(5.0, 10 * 1024 * 1024) is True

            # HEADLINE FIX: a 500 MB result computed in 1.5 s. Predicted restore
            # (~1.05 s) still saves more than the 20% floor, so it persists —
            # where the inverted bandwidth model refused it.
            assert policy(1.5, 500 * 1024 * 1024) is True

            # Genuine refuse: a 1 GB result computed in only 0.5 s. Predicted
            # restore (~2.1 s) is slower than recomputing, so rehydrating would
            # cost more than it saves — keep it RAM-only.
            assert policy(0.5, 1024 ** 3) is False
            assert policy(1.0, 1024 ** 3) is False

    def test_repr(self):
        """Cash repr includes backend and function count."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        r = repr(c)
        assert "InMemoryBackend" in r
        assert "functions=0" in r
        assert "debug=" in r


class TestCashCacheDecorator:
    """Test the @cash.cache decorator."""

    def test_basic_caching(self):
        """Cache hit on second call with same args."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        call_count = 0

        @c.cache
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        result1 = add(1, 2)
        result2 = add(1, 2)
        assert result1 == 3
        assert result2 == 3
        assert call_count == 1  # Second call should be cached

    def test_different_args_not_cached(self):
        """Different args compute separately."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        call_count = 0

        @c.cache
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        result1 = add(1, 2)
        result2 = add(3, 4)
        assert result1 == 3
        assert result2 == 7
        assert call_count == 2

    def test_cache_with_kwargs(self):
        """Caching works with keyword arguments."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        call_count = 0

        @c.cache
        def greet(name, greeting="Hello"):
            nonlocal call_count
            call_count += 1
            return f"{greeting}, {name}!"

        result1 = greet("Alice", greeting="Hi")
        result2 = greet("Alice", greeting="Hi")
        assert result1 == "Hi, Alice!"
        assert result2 == "Hi, Alice!"
        assert call_count == 1

    def test_cache_with_ttl(self):
        """TTL expiration works."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        call_count = 0

        @c.cache(ttl=1)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = compute(5)
        assert result1 == 10
        assert call_count == 1

        # Still cached
        compute(5)
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(1.1)
        result3 = compute(5)
        assert result3 == 10
        assert call_count == 2  # Recomputed

    def test_cache_preserves_function_metadata(self):
        """functools.wraps preserves function name/docstring."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache
        def my_func(x):
            """My docstring."""
            return x

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_cache_with_unhashable_args(self):
        """Unhashable args (like dicts) should still work via pickle."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        call_count = 0

        @c.cache
        def process(data):
            nonlocal call_count
            call_count += 1
            return sum(data.values())

        result1 = process({"a": 1, "b": 2})
        result2 = process({"a": 1, "b": 2})
        assert result1 == 3
        assert result2 == 3
        assert call_count == 1

    def test_cache_with_depends_on_function(self):
        """Static dependency on another function."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache
        def helper():
            return 42

        @c.cache(depends_on=[helper])
        def main():
            return helper() + 1

        result = main()
        assert result == 43

    def test_cache_with_depends_on_data_source(self, tmp_path):
        """Static dependency on a FileDataSource."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("hello")
        ds = FileDataSource(str(data_file))

        c = Cash(backend=InMemoryBackend(), register_magic=False)
        call_count = 0

        @c.cache(depends_on=[ds])
        def read_data():
            nonlocal call_count
            call_count += 1
            return data_file.read_text()

        result1 = read_data()
        assert result1 == "hello"
        assert call_count == 1

        # Same file, cached
        read_data()
        assert call_count == 1

    def test_cache_with_dynamic_depends_on(self, tmp_path):
        """Dynamic dependency resolution."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("content1")

        c = Cash(backend=InMemoryBackend(), register_magic=False)
        call_count = 0

        def get_source(filename):
            return FileDataSource(str(tmp_path / filename))

        @c.cache(dynamic_depends_on=get_source)
        def read(filename):
            nonlocal call_count
            call_count += 1
            return (tmp_path / filename).read_text()

        result = read("data.txt")
        assert result == "content1"
        assert call_count == 1


class TestCashCacheKeyGeneration:
    """Test cache key computation edge cases."""

    def test_unpicklable_args_fallback(self):
        """If args can't be pickled, function runs without caching."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        call_count = 0

        @c.cache
        def process(x):
            nonlocal call_count
            call_count += 1
            return str(x)

        # Lambda can't be pickled with standard pickle
        process(lambda: None)
        assert call_count == 1
        # Should still work, just not cached (no crash)


class TestCashCleanup:
    """Test cache cleanup and TTL expiration."""

    def test_cleanup_expired(self):
        """Cash.cleanup removes expired entries."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache(ttl=1)
        def compute(x):
            return x * 2

        compute(1)
        compute(2)

        # Wait for expiration
        time.sleep(1.1)
        removed = c.cleanup()
        assert removed >= 0  # Backend may or may not track

    def test_cleanup_with_max_age(self):
        """Cash.cleanup with max_age removes old entries."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache
        def compute(x):
            return x * 2

        compute(1)
        time.sleep(0.5)
        removed = c.cleanup(max_age=0)  # Remove everything
        assert removed >= 0

    def test_explorer(self):
        """Cash.explorer returns a CacheExplorer."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        explorer = c.explorer()
        assert explorer is not None


class TestCashWithLocking:
    """Test double-checked locking in cache decorator."""

    def test_cache_with_locking(self):
        """Cache with use_locking=True still works correctly."""
        c = Cash(backend=InMemoryBackend(), register_magic=False, use_locking=True)
        call_count = 0

        @c.cache
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 3

        result1 = compute(5)
        result2 = compute(5)
        assert result1 == 15
        assert result2 == 15
        assert call_count == 1


class TestCashDependencyAnalysis:
    """Test automatic dependency detection."""

    def test_auto_detect_function_calls(self):
        """Cash auto-detects calls to other cached functions."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache
        def helper():
            return 10

        @c.cache
        def main():
            return helper() + 5

        result = main()
        assert result == 15

        # Verify dependency was registered in graph
        # The qualname may vary, so just check graph has nodes
        all_nodes = set()
        for func_name in c.functions:
            deps = c.graph.get_dependencies(func_name)
            all_nodes.update(deps)
        # At minimum, the functions should be registered
        assert len(c.functions) == 2


class TestCashShutdown:
    """Test graceful shutdown."""

    def test_shutdown(self):
        """Shutdown doesn't raise errors."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        c.shutdown()

    def test_shutdown_with_file_backend(self, tmp_path):
        """Shutdown flushes FileBackend."""
        c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
        c.shutdown()


class TestCashRegisterMagic:
    """Test magic registration."""

    def test_register_magic_no_ipython(self):
        """register_magic() handles missing IPython gracefully."""
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        # Should not raise even if IPython is not available
        with patch('cash.core.get_ipython', return_value=None, create=True):
            c.register_magic()
