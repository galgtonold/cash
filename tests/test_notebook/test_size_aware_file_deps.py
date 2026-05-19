"""
Test that size-aware caching correctly handles different backend types and
uses a savings-based heuristic to decide when caching is worthwhile.

The core principle: caching should be skipped only when the expected time
savings (execution_time - restore_time) is less than a configurable
percentage of the execution time (default 20%).

The cost model is backend-aware:
  - RAM (InMemoryBackend): uses deepcopy (~2 GB/s for pandas, ~500 MB/s generic)
  - Disk (FileBackend): uses pickle serialization (~200 MB/s)

Special cases that always allow caching:
  - File dependencies (pd.read_csv, etc.) — I/O-bound, caching saves re-reading
  - force_persist (@cash:persist annotation) — user explicitly wants caching
"""
import pytest
import pandas as pd
import numpy as np

from cash.notebook.statement_processor import StatementProcessor
from cash.core import Cash
from cash.backends.backend import InMemoryBackend, FileBackend
from traitlets.config.configurable import Configurable


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


class TestSizeAwareCachingWithFileDeps:
    """Test that file-dependent statements bypass size-aware caching skip.

    Uses FileBackend so the cost-model predictions reflect the disk path
    (where serialize/deserialize cost is high enough for the policy to
    actually fire on the test sizes). On RAM, the fitted DataFrame copy
    cost is well under the fixed budget for 80 MB inputs.
    """

    @pytest.fixture
    def processor(self, tmp_path):
        """Create a StatementProcessor with a disk backend for testing."""
        backend = FileBackend(cache_dir=str(tmp_path))
        cash_instance = Cash(backend=backend, register_magic=False)
        shell = MockShell()
        processor = StatementProcessor(shell, cash_instance)
        processor.debug = False
        yield processor

    def test_skip_large_object_without_file_deps(self, processor):
        """Large object without file deps and fast compute → should skip cache."""
        large_data = {'df': pd.DataFrame(np.random.randn(100000, 100))}
        # 80MB DataFrame at 200MB/s → est_serial ≈ 0.4s
        # With exec_time=0.01, threshold = 2.0 * 0.01 = 0.02 → 0.4 > 0.02 → skip
        result, reason, _ = processor._should_skip_large_object_caching(
            large_data, execution_time=0.01, force_persist=False, has_file_dependencies=False
        )
        assert result is True
        assert reason is not None

    def test_no_skip_large_object_with_file_deps(self, processor):
        """Large object WITH file deps → should NOT skip cache (file I/O is expensive)."""
        large_data = {'df': pd.DataFrame(np.random.randn(100000, 100))}
        # Even with fast execution and large object, file deps mean we should cache
        result, reason, _ = processor._should_skip_large_object_caching(
            large_data, execution_time=1.0, force_persist=False, has_file_dependencies=True
        )
        assert result is False
        assert reason is None

    def test_no_skip_large_object_with_force_persist(self, processor):
        """Large object with force_persist → should NOT skip cache."""
        large_data = {'df': pd.DataFrame(np.random.randn(100000, 100))}
        result, reason, _ = processor._should_skip_large_object_caching(
            large_data, execution_time=1.0, force_persist=True, has_file_dependencies=False
        )
        assert result is False
        assert reason is None

    def test_no_skip_small_object(self, processor):
        """Small object → should NOT skip cache regardless of file deps."""
        small_data = {'x': pd.DataFrame({'a': [1, 2, 3]})}
        result, reason, _ = processor._should_skip_large_object_caching(
            small_data, execution_time=0.1, force_persist=False, has_file_dependencies=False
        )
        assert result is False
        assert reason is None

    def test_no_skip_large_object_slow_compute(self, processor):
        """Large object with slow compute → should NOT skip cache."""
        large_data = {'df': pd.DataFrame(np.random.randn(100000, 100))}
        # Execution time > 10s → expensive computation, ratio check passes
        result, reason, _ = processor._should_skip_large_object_caching(
            large_data, execution_time=10.0, force_persist=False, has_file_dependencies=False
        )
        assert result is False
        assert reason is None


class TestBackendAwareCaching:
    """Test that the caching heuristic adapts to backend type."""

    @pytest.fixture
    def ram_processor(self):
        """Processor with InMemoryBackend (RAM)."""
        backend = InMemoryBackend()
        cash_instance = Cash(backend=backend, register_magic=False)
        shell = MockShell()
        processor = StatementProcessor(shell, cash_instance)
        processor.debug = False
        yield processor
        backend.clear()

    @pytest.fixture
    def disk_processor(self):
        """Processor with FileBackend (disk)."""
        from cash.backends.backend import FileBackend
        import tempfile
        tmp = tempfile.mkdtemp()
        backend = FileBackend(cache_dir=tmp)
        cash_instance = Cash(backend=backend, register_magic=False)
        shell = MockShell()
        processor = StatementProcessor(shell, cash_instance)
        processor.debug = False
        yield processor
        backend.clear()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_ram_caches_large_dataframe_easily(self, ram_processor):
        """RAM backend uses fast deepcopy — 80MB DataFrame with 0.1s compute is cached."""
        # 80MB DataFrame. RAM deepcopy at ~2 GB/s ≈ 0.04s restore time.
        # 0.1s compute → savings = 0.1 - 0.04 = 0.06 → 60% savings → well above 20% threshold
        large_data = {'df': pd.DataFrame(np.random.randn(100000, 100))}
        result, reason, _ = ram_processor._should_skip_large_object_caching(
            large_data, execution_time=0.1, force_persist=False, has_file_dependencies=False
        )
        assert result is False, f"RAM should cache 80MB DataFrame with 0.1s compute, but got skip: {reason}"

    def test_ram_skips_giant_object_with_modest_compute(self, ram_processor):
        """RAM backend skips caching if deepcopy takes longer than the computation.
        Uses execution_time=0.05 (50 ms) to stay above the 10 ms min-execution-time
        floor while remaining far below the ~1 s predicted deepcopy cost for 500 MB."""
        # Create a very large generic object (not pandas — slower deepcopy at ~500 MB/s)
        # 500 MB → est_restore ≈ 1.0s at 500MB/s. exec_time=0.05 → threshold=0.04
        # 1.0 > 0.04 → skip
        large_data = {'big': 'x' * (500 * 1024 * 1024)}  # ~500 MB string
        result, reason, _ = ram_processor._should_skip_large_object_caching(
            large_data, execution_time=0.05, force_persist=False, has_file_dependencies=False
        )
        assert result is True
        assert "copying" in reason.lower() or "@cash:persist" in reason

    def test_disk_skips_when_serialization_too_slow(self, disk_processor):
        """Disk backend with slow serialization relative to compute should skip."""
        # 80MB DataFrame at 200MB/s → est_restore ≈ 0.4s. 
        # exec_time=0.01 → threshold=0.008. 0.4 > 0.008 → skip
        large_data = {'df': pd.DataFrame(np.random.randn(100000, 100))}
        result, reason, _ = disk_processor._should_skip_large_object_caching(
            large_data, execution_time=0.01, force_persist=False, has_file_dependencies=False
        )
        assert result is True
        assert "serializing" in reason.lower() or "@cash:persist" in reason

    def test_disk_caches_when_compute_is_expensive(self, disk_processor):
        """Disk backend caches when computation is much more expensive than serialization."""
        # 80MB DataFrame at 200MB/s → est_restore ≈ 0.4s.
        # exec_time=10 → threshold=8.0. 0.4 < 8.0 → cache
        large_data = {'df': pd.DataFrame(np.random.randn(100000, 100))}
        result, reason, _ = disk_processor._should_skip_large_object_caching(
            large_data, execution_time=10.0, force_persist=False, has_file_dependencies=False
        )
        assert result is False

    def test_reason_always_provided_on_skip(self, ram_processor):
        """When caching is skipped, a human-readable reason is always returned.
        Uses execution_time=0.05 (50 ms) to stay above the 10 ms min-execution-time
        floor so the object-size-based skip reason (not the floor reason) fires."""
        large_data = {'big': 'x' * (500 * 1024 * 1024)}
        result, reason, _ = ram_processor._should_skip_large_object_caching(
            large_data, execution_time=0.05, force_persist=False, has_file_dependencies=False
        )
        assert result is True
        assert reason is not None
        assert len(reason) > 10  # Not just an empty string
        assert "MB" in reason or "GB" in reason
        assert "@cash:persist" in reason

    def test_reason_none_when_caching(self, ram_processor):
        """When caching proceeds, reason is None."""
        small_data = {'x': 42}
        result, reason, _ = ram_processor._should_skip_large_object_caching(
            small_data, execution_time=1.0
        )
        assert result is False
        assert reason is None


class TestReadCsvCachingUnit:
    """Unit test that pd.read_csv caches correctly even for large files."""

    @pytest.fixture
    def magics_fixture(self):
        """Provide CashMagics + mock shell for testing."""
        from cash.notebook.magics import CashMagics
        from unittest.mock import MagicMock

        backend = InMemoryBackend()
        cash_instance = Cash(backend=backend, register_magic=False)
        shell = MockShell()
        shell.run_cell = MagicMock()
        shell.events = MagicMock()

        magics = CashMagics(shell, cash_instance)
        magics._auto_cache_enabled = True
        magics._debug = False

        yield magics, shell, backend

        backend.clear()
        shell.user_ns.clear()

    def test_read_csv_large_file_is_cached(self, magics_fixture, tmp_path):
        """
        A large CSV file read via pd.read_csv should be cached,
        even if the resulting DataFrame exceeds 50MB.
        """
        magics, shell, backend = magics_fixture
        magics._debug = True

        # Create a large-ish CSV (not 100MB but enough to trigger size check)
        csv_path = tmp_path / "large_data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        
        # Create DataFrame > 50MB in memory
        n_rows = 500000
        large_df = pd.DataFrame({
            'a': np.random.randn(n_rows),
            'b': np.random.randn(n_rows),
            'c': np.random.randn(n_rows),
            'd': np.random.randn(n_rows),
            'e': np.random.randn(n_rows),
            'f': np.random.randn(n_rows),
            'g': np.random.randn(n_rows),
            'h': np.random.randn(n_rows),
        })
        large_df.to_csv(csv_path, index=False)

        # Setup namespace
        import pandas
        shell.user_ns['pd'] = pandas
        # Execute to give pd a lineage
        magics.cash("", "import pandas as pd")

        # Set data_path 
        magics.cash("", f"data_path = '{csv_path_str}'")

        # Now read the CSV - this should be cached despite large size
        magics.cash("", "df = pd.read_csv(data_path)")
        
        assert 'df' in shell.user_ns
        assert len(shell.user_ns['df']) == n_rows

        # Check that a cache entry was stored (not skipped)
        sp = magics._statement_processor
        assert 'df' in sp.variable_lineage, "df should have a lineage after caching"
        
        # The key test: the backend should have a cache entry for read_csv.
        # data_path = '...' is a trivial string assignment — it may be skipped by the
        # min-execution-time floor (< 10 ms) — so we only require at least 1 entry.
        stored_keys = list(backend._store.keys())
        assert len(stored_keys) >= 1, f"Expected at least 1 cache entry, got {len(stored_keys)}"
        
        # Verify the read_csv cache entry exists by checking metadata
        found_read_csv = False
        for key in stored_keys:
            meta, _ = backend.get(key)
            if meta and meta.get('code', '').startswith('df = pd.read_csv'):
                found_read_csv = True
                break
        assert found_read_csv, f"No cache entry found for pd.read_csv. Keys: {stored_keys}"
