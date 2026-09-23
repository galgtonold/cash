from cash.notebook.cache_status import CacheStatus

"""
Tests for StatementProcessor methods that need additional coverage.

Targets: _check_cache (stale format, file deps, TTL), error_result,
         _update_mutation_lineages, _handle_execution_error,
         file dep propagation, module lineage, forbidden function scan error,
         lineage-exemption predicate (via cacheability_decision.is_lineage_exempt).
"""
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from cash.notebook.statement.run import error_result

# ============================================================================
# _check_cache - stale format, TTL, file dependencies
# ============================================================================


class TestCheckCache:
    """Test _check_cache method edge cases."""

    def test_cache_miss_returns_none(self, statement_processor):
        metadata, cached_data, time_taken = statement_processor._freshness.check_cache(
            statement_processor.tracking_state, "nonexistent_key", None
        )
        assert cached_data is None

    def test_ttl_expiration(self, statement_processor, clean_backend):
        """Cache entries past TTL should be invalidated."""
        cache_key = "test_ttl_key"
        metadata = {
            "timestamp": time.time() - 100,  # 100 seconds ago
            "output_lineages": {"x": "abc123"},
        }
        cached_data = {"variables": {"x": 42}}
        clean_backend.set(cache_key, cached_data, metadata)

        # TTL of 10 seconds - entry should be expired
        result_meta, result_data, _ = statement_processor._freshness.check_cache(
            statement_processor.tracking_state, cache_key, 10
        )
        assert result_data is None

    def test_ttl_not_expired(self, statement_processor, clean_backend):
        """Cache entries within TTL should be valid."""
        cache_key = "test_ttl_valid"
        metadata = {
            "timestamp": time.time(),
            "output_lineages": {"x": "abc123"},
        }
        cached_data = {"variables": {"x": 42}}
        clean_backend.set(cache_key, cached_data, metadata)

        result_meta, result_data, _ = statement_processor._freshness.check_cache(
            statement_processor.tracking_state, cache_key, 3600
        )
        assert result_data is not None

    def test_file_dep_missing_file(self, statement_processor, clean_backend):
        """Cache with file dep pointing to missing file should be invalidated."""
        cache_key = "test_file_dep_missing"
        metadata = {
            "timestamp": time.time(),
            "output_lineages": {"x": "abc123"},
            "file_dependencies": {"/nonexistent/file.csv": {"mtime": time.time()}},
        }
        cached_data = {"variables": {"x": 42}}
        clean_backend.set(cache_key, cached_data, metadata)

        result_meta, result_data, _ = statement_processor._freshness.check_cache(
            statement_processor.tracking_state, cache_key, None
        )
        assert result_data is None

    def test_file_dep_changed_mtime(self, statement_processor, clean_backend, tmp_path):
        """Cache with changed file mtime should be invalidated."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n", encoding="utf-8")
        from cash.tracking.file_dep_snapshot import snapshot_file_deps

        snapshot = snapshot_file_deps({str(test_file)})
        test_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")  # changed since the snapshot

        cache_key = "test_file_dep_changed"
        metadata = {
            "timestamp": time.time(),
            "output_lineages": {"x": "abc123"},
            "file_dependencies": snapshot,
        }
        cached_data = {"variables": {"x": 42}}
        clean_backend.set(cache_key, cached_data, metadata)

        result_meta, result_data, _ = statement_processor._freshness.check_cache(
            statement_processor.tracking_state, cache_key, None
        )
        assert result_data is None

    def test_file_dep_unchanged(self, statement_processor, clean_backend, tmp_path):
        """Cache with unchanged file should be valid."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n", encoding="utf-8")
        from cash.tracking.file_dep_snapshot import snapshot_file_deps

        cache_key = "test_file_dep_ok"
        metadata = {
            "timestamp": time.time(),
            "output_lineages": {"x": "abc123"},
            "file_dependencies": snapshot_file_deps({str(test_file)}),
        }
        cached_data = {"variables": {"x": 42}}
        clean_backend.set(cache_key, cached_data, metadata)

        result_meta, result_data, _ = statement_processor._freshness.check_cache(
            statement_processor.tracking_state, cache_key, None
        )
        assert result_data is not None

    def test_input_file_dep_invalidation(self, statement_processor, clean_backend, tmp_path):
        """Cache should invalidate when an input variable's file dep changed."""
        test_file = tmp_path / "source.csv"
        test_file.write_text("a,b\n1,2\n", encoding="utf-8")
        current_mtime = os.path.getmtime(str(test_file))

        # Set up input var's file dependencies — mutate the shared TrackingState
        # dict so the freshness checker (which reads it per-call) sees the update.
        statement_processor.tracking_state.executed_file_deps["df"] = {str(test_file)}

        # Store source cache entry for the input variable with OLD mtime
        source_key = "source_cache_key"
        source_meta = {
            "timestamp": time.time(),
            "output_lineages": {"df": "def456"},
            "file_dependencies": {str(test_file): {"mtime": current_mtime - 100}},  # Old mtime
        }
        clean_backend.set(source_key, {"variables": {"df": "data"}}, source_meta)
        statement_processor.tracking_state.variable_sources["df"] = source_key

        # Now store the dependent cache entry (no direct file deps)
        cache_key = "test_input_dep"
        metadata = {
            "timestamp": time.time(),
            "output_lineages": {"result": "ghi789"},
        }
        cached_data = {"variables": {"result": 100}}
        clean_backend.set(cache_key, cached_data, metadata)

        result_meta, result_data, _ = statement_processor._freshness.check_cache(
            statement_processor.tracking_state, cache_key, None, inputs={"df"}
        )
        assert result_data is None


# ============================================================================
# is_lineage_exempt
# ============================================================================

from cash.analysis.cacheability_decision import is_lineage_exempt


class TestIsLineageExempt:
    """Test the lineage-exemption predicate."""

    def test_skip_module(self):
        import os as os_mod

        assert is_lineage_exempt("os", os_mod) is True

    def test_skip_get_ipython(self):
        assert is_lineage_exempt("get_ipython", lambda: None) is True

    def test_skip_private_callable(self):
        func = MagicMock()
        func.__self__ = MagicMock()
        assert is_lineage_exempt("_private", func) is True

    def test_dont_skip_regular_variable(self):
        assert is_lineage_exempt("x", 42) is False

    def test_dont_skip_user_function(self):
        def my_func():
            pass

        assert is_lineage_exempt("my_func", my_func) is False

    def test_dont_skip_list(self):
        assert is_lineage_exempt("data", [1, 2, 3]) is False


# ============================================================================
# error_result
# ============================================================================


class TestCreateErrorResult:
    """Test the error_result helper."""

    def test_basic_error_result(self):
        # Test error_result directly
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = error_result(e)
            assert result.success is False
            assert isinstance(result.error, ValueError)
            assert "test error" in str(result.error)
            assert isinstance(result.tb_string, str)

    def test_error_result_with_nested_frames(self):

        def inner():
            raise RuntimeError("inner error")

        try:
            inner()
        except RuntimeError as e:
            result = error_result(e)
            assert result.success is False
            assert "inner error" in str(result.error)


# ============================================================================
# _handle_execution_error
# ============================================================================


class TestHandleExecutionError:
    """Test _handle_execution_error method."""

    def test_non_silent_raises(self, statement_processor):
        result = MagicMock()
        result.error = ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            statement_processor._handle_execution_error(result, silent=False)

    def test_silent_returns_false(self, statement_processor):
        result = MagicMock()
        result.error = ValueError("boom")
        ret = statement_processor._handle_execution_error(result, silent=True)
        assert ret is False

    def test_silent_debug_output(self, statement_processor, caplog):
        caplog.set_level("DEBUG", logger="cash")
        result = MagicMock()
        result.error = ValueError("debug error")
        ret = statement_processor._handle_execution_error(result, silent=True)
        assert ret is False
        assert "debug error" in caplog.text


# ============================================================================
# Forbidden function scan error handling
# ============================================================================


class TestForbiddenFunctionScan:
    """Test forbidden function scan error handling."""

    def test_forbidden_function_disables_cache(self, statement_processor, mock_shell):
        """time.time() should be detected as forbidden."""
        import time as time_mod

        mock_shell.user_ns["time"] = time_mod
        statement_processor.process_statement("t = time.time()")
        # Should execute but mark as uncacheable
        assert mock_shell.user_ns.get("t") is not None

    def test_scan_error_handled_gracefully(self, statement_processor, mock_shell):
        """If forbidden scan raises, execution should still proceed."""
        # Patch the scan to raise
        with patch(
            "cash.analysis.code_analyzer.CodeAnalyzer.scan_for_forbidden_functions", side_effect=TypeError("scan error")
        ):
            statement_processor.process_statement("x = 42")
        assert mock_shell.user_ns.get("x") == 42


# ============================================================================
# File dependency propagation (scalar vs non-scalar)
# ============================================================================


class TestFileDependencyPropagation:
    """Test file dep propagation from inputs to outputs."""

    def test_scalar_output_no_file_dep_propagation(self, statement_processor, mock_shell, tmp_path):
        """Scalar outputs should NOT inherit file deps from inputs."""

        # First, create a variable with file deps
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

        # Simulate that 'df' has file deps — mutate the shared dicts so
        # sibling sub-components (StatementFileDeps) see the update too.
        statement_processor.tracking_state.executed_file_deps["df"] = {str(test_file)}

        # Set up 'df' in namespace (as a list to avoid pandas dependency)
        mock_shell.user_ns["df"] = [1, 2, 3]
        statement_processor.tracking_state.lineage.record("df", "df_lineage")

        # Now compute a scalar from df
        statement_processor.process_statement("n = len(df)")

        # 'n' is an int (scalar) - should NOT inherit file deps
        assert mock_shell.user_ns.get("n") == 3
        file_deps = statement_processor.tracking_state.executed_file_deps.get("n", set())
        assert len(file_deps) == 0

    def test_non_scalar_output_inherits_file_deps(self, statement_processor, mock_shell, tmp_path):
        """Non-scalar outputs SHOULD inherit file deps from inputs."""

        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n", encoding="utf-8")

        # Mutate the shared dicts so StatementFileDeps sees the update too.
        statement_processor.tracking_state.executed_file_deps["data"] = {str(test_file)}

        mock_shell.user_ns["data"] = [1, 2, 3]
        statement_processor.tracking_state.lineage.record("data", "data_lineage")

        # Create a non-scalar output from data
        statement_processor.process_statement("result = list(data)")

        assert mock_shell.user_ns.get("result") == [1, 2, 3]
        file_deps = statement_processor.tracking_state.executed_file_deps.get("result", set())
        assert str(test_file) in file_deps


# ============================================================================
# Module lineage component
# ============================================================================


class TestModuleLineage:
    """Test module lineage component in _capture_variables."""

    def test_module_import_creates_lineage(self, statement_processor, mock_shell):
        """Importing a module should create a lineage entry."""
        statement_processor.process_statement("import json")
        # json should be in user_ns but not cached (modules are skipped)
        assert "json" in mock_shell.user_ns
        # Module gets lineage tracking
        assert "json" in statement_processor.tracking_state.variable_lineage


# ============================================================================
# Purity check paths
# ============================================================================


class TestPurityChecks:
    """Test purity check branches in process()."""

    def test_stateful_function_skips_cache(self, statement_processor, mock_shell):
        """@stateful functions should skip cache."""
        from cash.purity import stateful

        @stateful
        def get_data():
            return [1, 2, 3]

        mock_shell.user_ns["get_data"] = get_data
        metrics = statement_processor.process_statement("result = get_data()")
        assert mock_shell.user_ns.get("result") == [1, 2, 3]
        # Should be COMPUTED (not cacheable)
        assert metrics["status"] == CacheStatus.COMPUTED
        assert any("stateful" in r.lower() for r in metrics.get("uncacheable_reasons", []))

    def test_pure_function_is_cacheable(self, statement_processor, mock_shell):
        """@pure functions should be cacheable."""
        from cash.purity import pure

        @pure
        def add(a, b):
            return a + b

        mock_shell.user_ns["add"] = add
        mock_shell.user_ns["x"] = 5
        mock_shell.user_ns["y"] = 3
        statement_processor.tracking_state.lineage.record("x", "x_lin")
        statement_processor.tracking_state.lineage.record("y", "y_lin")

        metrics = statement_processor.process_statement("result = add(x, y)")
        assert mock_shell.user_ns.get("result") == 8
        # Should NOT have stateful uncacheable reason
        assert not any("stateful" in r.lower() for r in metrics.get("uncacheable_reasons", []))
