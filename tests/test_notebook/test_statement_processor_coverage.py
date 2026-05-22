from cash.notebook.cache_status import CacheStatus
"""
Tests for StatementProcessor methods that need additional coverage.

Targets: _check_cache (stale format, file deps, TTL), _create_error_result,
         _update_mutation_lineages, _should_skip_variable, _handle_execution_error,
         file dep propagation, module lineage, forbidden function scan error
"""
import pytest
import os
import time
from unittest.mock import MagicMock, patch

from cash.notebook.magics import CashMagics
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def processor_fixture():
    """Provide StatementProcessor instance for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    processor = magics._statement_processor
    yield processor, shell, backend
    backend.clear()
    shell.user_ns.clear()


# ============================================================================
# _check_cache - stale format, TTL, file dependencies
# ============================================================================

class TestCheckCache:
    """Test _check_cache method edge cases."""

    def test_cache_miss_returns_none(self, processor_fixture):
        processor, _, _ = processor_fixture
        metadata, cached_data, time_taken = processor._freshness.check_cache("nonexistent_key", None)
        assert cached_data is None

    def test_stale_format_invalidation(self, processor_fixture):
        """Cache entries without output_lineages should be invalidated."""
        processor, _, backend = processor_fixture
        # Manually store a cache entry without output_lineages
        cache_key = "test_stale_key"
        metadata = {
            'timestamp': time.time(),
            # Missing 'output_lineages' - stale format
        }
        cached_data = {'variables': {'x': 42}}
        backend.set(cache_key, cached_data, metadata)
        
        result_meta, result_data, _ = processor._freshness.check_cache(cache_key, None)
        # Should be invalidated due to missing output_lineages
        assert result_data is None

    def test_ttl_expiration(self, processor_fixture):
        """Cache entries past TTL should be invalidated."""
        processor, _, backend = processor_fixture
        cache_key = "test_ttl_key"
        metadata = {
            'timestamp': time.time() - 100,  # 100 seconds ago
            'output_lineages': {'x': 'abc123'},
        }
        cached_data = {'variables': {'x': 42}}
        backend.set(cache_key, cached_data, metadata)
        
        # TTL of 10 seconds - entry should be expired
        result_meta, result_data, _ = processor._freshness.check_cache(cache_key, 10)
        assert result_data is None

    def test_ttl_not_expired(self, processor_fixture):
        """Cache entries within TTL should be valid."""
        processor, _, backend = processor_fixture
        cache_key = "test_ttl_valid"
        metadata = {
            'timestamp': time.time(),
            'output_lineages': {'x': 'abc123'},
        }
        cached_data = {'variables': {'x': 42}}
        backend.set(cache_key, cached_data, metadata)
        
        result_meta, result_data, _ = processor._freshness.check_cache(cache_key, 3600)
        assert result_data is not None

    def test_file_dep_missing_file(self, processor_fixture):
        """Cache with file dep pointing to missing file should be invalidated."""
        processor, _, backend = processor_fixture
        cache_key = "test_file_dep_missing"
        metadata = {
            'timestamp': time.time(),
            'output_lineages': {'x': 'abc123'},
            'file_dependencies': {'/nonexistent/file.csv': time.time()},
        }
        cached_data = {'variables': {'x': 42}}
        backend.set(cache_key, cached_data, metadata)
        
        result_meta, result_data, _ = processor._freshness.check_cache(cache_key, None)
        assert result_data is None

    def test_file_dep_changed_mtime(self, processor_fixture, tmp_path):
        """Cache with changed file mtime should be invalidated."""
        processor, _, backend = processor_fixture
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n")
        
        cache_key = "test_file_dep_changed"
        metadata = {
            'timestamp': time.time(),
            'output_lineages': {'x': 'abc123'},
            'file_dependencies': {str(test_file): time.time() - 100},  # Old mtime
        }
        cached_data = {'variables': {'x': 42}}
        backend.set(cache_key, cached_data, metadata)
        
        result_meta, result_data, _ = processor._freshness.check_cache(cache_key, None)
        assert result_data is None

    def test_file_dep_unchanged(self, processor_fixture, tmp_path):
        """Cache with unchanged file should be valid."""
        processor, _, backend = processor_fixture
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n")
        current_mtime = os.path.getmtime(str(test_file))
        
        cache_key = "test_file_dep_ok"
        metadata = {
            'timestamp': time.time(),
            'output_lineages': {'x': 'abc123'},
            'file_dependencies': {str(test_file): current_mtime},
        }
        cached_data = {'variables': {'x': 42}}
        backend.set(cache_key, cached_data, metadata)
        
        result_meta, result_data, _ = processor._freshness.check_cache(cache_key, None)
        assert result_data is not None

    def test_input_file_dep_invalidation(self, processor_fixture, tmp_path):
        """Cache should invalidate when an input variable's file dep changed."""
        processor, shell, backend = processor_fixture
        test_file = tmp_path / "source.csv"
        test_file.write_text("a,b\n1,2\n")
        current_mtime = os.path.getmtime(str(test_file))
        
        # Set up input var's file dependencies — mutate the shared dict so the
        # freshness checker (which holds its own ref via set_tracking_state)
        # sees the update too.
        processor.executed_file_deps['df'] = {str(test_file)}

        # Store source cache entry for the input variable with OLD mtime
        source_key = "source_cache_key"
        source_meta = {
            'timestamp': time.time(),
            'output_lineages': {'df': 'def456'},
            'file_dependencies': {str(test_file): current_mtime - 100},  # Old mtime
        }
        backend.set(source_key, {'variables': {'df': 'data'}}, source_meta)
        processor.variable_sources['df'] = source_key
        
        # Now store the dependent cache entry (no direct file deps)
        cache_key = "test_input_dep"
        metadata = {
            'timestamp': time.time(),
            'output_lineages': {'result': 'ghi789'},
        }
        cached_data = {'variables': {'result': 100}}
        backend.set(cache_key, cached_data, metadata)
        
        result_meta, result_data, _ = processor._freshness.check_cache(cache_key, None, inputs={'df'})
        assert result_data is None


# ============================================================================
# _should_skip_variable
# ============================================================================

class TestShouldSkipVariable:
    """Test _should_skip_variable method."""

    def test_skip_module(self, processor_fixture):
        processor, _, _ = processor_fixture
        import os as os_mod
        assert processor._should_skip_variable('os', os_mod) is True

    def test_skip_get_ipython(self, processor_fixture):
        processor, _, _ = processor_fixture
        assert processor._should_skip_variable('get_ipython', lambda: None) is True

    def test_skip_private_callable(self, processor_fixture):
        processor, _, _ = processor_fixture
        func = MagicMock()
        func.__self__ = MagicMock()
        assert processor._should_skip_variable('_private', func) is True

    def test_dont_skip_regular_variable(self, processor_fixture):
        processor, _, _ = processor_fixture
        assert processor._should_skip_variable('x', 42) is False

    def test_dont_skip_user_function(self, processor_fixture):
        processor, _, _ = processor_fixture
        def my_func():
            pass
        assert processor._should_skip_variable('my_func', my_func) is False

    def test_dont_skip_list(self, processor_fixture):
        processor, _, _ = processor_fixture
        assert processor._should_skip_variable('data', [1, 2, 3]) is False


# ============================================================================
# _create_error_result
# ============================================================================

class TestCreateErrorResult:
    """Test _create_error_result method."""

    def test_basic_error_result(self, processor_fixture):
        processor, shell, _ = processor_fixture
        # Test _create_error_result directly
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = processor._create_error_result(e)
            assert result.success is False
            assert isinstance(result.error, ValueError)
            assert "test error" in str(result.error)
            assert isinstance(result.tb_string, str)

    def test_error_result_with_nested_frames(self, processor_fixture):
        processor, _, _ = processor_fixture
        def inner():
            raise RuntimeError("inner error")
        try:
            inner()
        except RuntimeError as e:
            result = processor._create_error_result(e)
            assert result.success is False
            assert "inner error" in str(result.error)


# ============================================================================
# _handle_execution_error  
# ============================================================================

class TestHandleExecutionError:
    """Test _handle_execution_error method."""

    def test_non_silent_raises(self, processor_fixture):
        processor, _, _ = processor_fixture
        result = MagicMock()
        result.error = ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            processor._handle_execution_error(result, silent=False)

    def test_silent_returns_false(self, processor_fixture):
        processor, _, _ = processor_fixture
        result = MagicMock()
        result.error = ValueError("boom")
        ret = processor._handle_execution_error(result, silent=True)
        assert ret is False

    def test_silent_debug_output(self, processor_fixture, capsys):
        processor, _, _ = processor_fixture
        processor.debug = True
        result = MagicMock()
        result.error = ValueError("debug error")
        ret = processor._handle_execution_error(result, silent=True)
        assert ret is False


# ============================================================================
# Forbidden function scan error handling
# ============================================================================

class TestForbiddenFunctionScan:
    """Test forbidden function scan error handling."""

    def test_forbidden_function_disables_cache(self, processor_fixture):
        """time.time() should be detected as forbidden."""
        processor, shell, _ = processor_fixture
        import time as time_mod
        shell.user_ns['time'] = time_mod
        processor.process_statement("t = time.time()")
        # Should execute but mark as uncacheable
        assert shell.user_ns.get('t') is not None

    def test_scan_error_handled_gracefully(self, processor_fixture):
        """If forbidden scan raises, execution should still proceed."""
        processor, shell, _ = processor_fixture
        # Patch the scan to raise
        with patch('cash.notebook.analysis.CodeAnalyzer.scan_for_forbidden_functions', side_effect=TypeError("scan error")):
            processor.process_statement("x = 42")
        assert shell.user_ns.get('x') == 42


# ============================================================================
# File dependency propagation (scalar vs non-scalar)
# ============================================================================

class TestFileDependencyPropagation:
    """Test file dep propagation from inputs to outputs."""

    def test_scalar_output_no_file_dep_propagation(self, processor_fixture, tmp_path):
        """Scalar outputs should NOT inherit file deps from inputs."""
        processor, shell, _ = processor_fixture
        
        # First, create a variable with file deps
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n3,4\n")
        
        # Simulate that 'df' has file deps — mutate the shared dicts so
        # sibling sub-components (StatementFileDeps) see the update too.
        processor.executed_file_deps['df'] = {str(test_file)}
        processor.executed_file_mtimes['df'] = {str(test_file): os.path.getmtime(str(test_file))}
        
        # Set up 'df' in namespace (as a list to avoid pandas dependency)
        shell.user_ns['df'] = [1, 2, 3]
        processor.variable_lineage['df'] = 'df_lineage'
        
        # Now compute a scalar from df
        processor.process_statement("n = len(df)")
        
        # 'n' is an int (scalar) - should NOT inherit file deps
        assert shell.user_ns.get('n') == 3
        file_deps = processor.executed_file_deps.get('n', set())
        assert len(file_deps) == 0

    def test_non_scalar_output_inherits_file_deps(self, processor_fixture, tmp_path):
        """Non-scalar outputs SHOULD inherit file deps from inputs."""
        processor, shell, _ = processor_fixture
        
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n")
        
        # Mutate the shared dicts so StatementFileDeps sees the update too.
        processor.executed_file_deps['data'] = {str(test_file)}
        processor.executed_file_mtimes['data'] = {str(test_file): os.path.getmtime(str(test_file))}
        
        shell.user_ns['data'] = [1, 2, 3]
        processor.variable_lineage['data'] = 'data_lineage'
        
        # Create a non-scalar output from data
        processor.process_statement("result = list(data)")
        
        assert shell.user_ns.get('result') == [1, 2, 3]
        file_deps = processor.executed_file_deps.get('result', set())
        assert str(test_file) in file_deps


# ============================================================================
# Module lineage component
# ============================================================================

class TestModuleLineage:
    """Test module lineage component in _capture_variables."""

    def test_module_import_creates_lineage(self, processor_fixture):
        """Importing a module should create a lineage entry."""
        processor, shell, _ = processor_fixture
        processor.process_statement("import json")
        # json should be in user_ns but not cached (modules are skipped)
        assert 'json' in shell.user_ns
        # Module gets lineage tracking
        assert 'json' in processor.variable_lineage


# ============================================================================
# Purity check paths
# ============================================================================

class TestPurityChecks:
    """Test purity check branches in process()."""

    def test_stateful_function_skips_cache(self, processor_fixture):
        """@stateful functions should skip cache."""
        processor, shell, _ = processor_fixture
        from cash.notebook.purity import stateful
        
        @stateful
        def get_data():
            return [1, 2, 3]
        
        shell.user_ns['get_data'] = get_data
        metrics = processor.process_statement("result = get_data()")
        assert shell.user_ns.get('result') == [1, 2, 3]
        # Should be COMPUTED (not cacheable)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert any('stateful' in r.lower() for r in metrics.get('uncacheable_reasons', []))

    def test_pure_function_is_cacheable(self, processor_fixture):
        """@pure functions should be cacheable."""
        processor, shell, _ = processor_fixture
        from cash.notebook.purity import pure
        
        @pure
        def add(a, b):
            return a + b
        
        shell.user_ns['add'] = add
        shell.user_ns['x'] = 5
        shell.user_ns['y'] = 3
        processor.variable_lineage['x'] = 'x_lin'
        processor.variable_lineage['y'] = 'y_lin'
        
        metrics = processor.process_statement("result = add(x, y)")
        assert shell.user_ns.get('result') == 8
        # Should NOT have stateful uncacheable reason
        assert not any('stateful' in r.lower() for r in metrics.get('uncacheable_reasons', []))


# ============================================================================
# _render_status_badge
# ============================================================================

class TestRenderStatusBadge:
    """Test _render_status_badge (standalone usage)."""

    def test_render_restored_badge(self, processor_fixture):
        """Should not crash when rendering RESTORED badge."""
        processor, _, _ = processor_fixture
        # Should not raise
        processor._render_status_badge(
            'RESTORED', 
            execution_time=0.1, 
            time_saved=2.5, 
            source='memory',
            storage=['InMemory']
        )

    def test_render_computed_badge(self, processor_fixture):
        processor, _, _ = processor_fixture
        processor._render_status_badge(
            'COMPUTED', 
            execution_time=1.5,
            storage=['InMemory', 'File']
        )

    def test_render_with_no_extras(self, processor_fixture):
        processor, _, _ = processor_fixture
        processor._render_status_badge('SKIPPED')
