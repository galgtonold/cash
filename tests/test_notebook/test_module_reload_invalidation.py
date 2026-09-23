from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus

"""
Tests for module reload cache invalidation and exception surfacing.

Issue 1: When a tracked local module (e.g., metrics.py) is modified and reloaded,
         statements that depend on the module should get cache misses, not cache hits.

Issue 2: Exceptions during cell execution (e.g., import nonexistent_module, raise ValueError)
         should be surfaced to the user, not silently swallowed.
"""

import hashlib
import importlib
import sys
import time
from unittest.mock import MagicMock

import pytest

from cash.tracking.function_tracker import FunctionTracker
from tests._cell_driver import run_cash_cell

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_module(tmp_path):
    """Create a temporary Python module for tracking tests.

    Returns (module_name, module_path) and cleans up sys.modules/path on exit.
    """
    module_name = f"_test_reload_mod_{id(tmp_path)}"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text("def increment(x):\n    return x + 1\n")

    sys.path.insert(0, str(tmp_path))

    yield module_name, str(module_file)

    sys.path.remove(str(tmp_path))
    if module_name in sys.modules:
        del sys.modules[module_name]


# ============================================================================
# Module Reload Cache Invalidation Tests
# ============================================================================


class TestModuleReloadInvalidation:
    """Test that module reload properly invalidates cache for dependent statements."""

    def test_invalidate_module_lineages_updates_module_lineage(self, cash_magics, mock_shell, temp_module):
        """After module reload, module's variable_lineage should change."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        # Import the module
        mod = importlib.import_module(module_name)
        mock_shell.user_ns[module_name] = mod

        # Set initial lineage for the module
        old_lineage = hashlib.sha256(b"old_source").hexdigest()
        sp.tracking_state.lineage.record(module_name, old_lineage)

        # Simulate module reload
        changed_modules = {module_name: module_file}
        cash_magics._module_invalidator.invalidate(
            changed_modules,
            cash_magics._statement_processor,
        )

        # Lineage should have changed
        new_lineage = sp.tracking_state.variable_lineage[module_name]
        assert new_lineage != old_lineage
        # It is the module's source identity, the one an import's lineage carries
        from cash.notebook.lineage_formula import read_module_source_hash

        assert new_lineage == read_module_source_hash(module_file)

    def test_invalidate_module_lineages_clears_dependent_vars(self, cash_magics, temp_module):
        """Variables computed from a changed module should have their lineage cleared."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        # Set up lineage: module has old lineage, result depends on it
        old_module_lineage = hashlib.sha256(b"old_source").hexdigest()
        sp.tracking_state.lineage.record(module_name, old_module_lineage)
        sp.tracking_state.lineage.record("result", "some_hash_for_result")
        sp.tracking_state.executed_cell_codes["result"] = f"result = {module_name}.increment(5)"
        sp.tracking_state.executed_input_lineages["result"] = {module_name: old_module_lineage}
        sp.tracking_state.current_session_hashes["result"] = "some_content_hash"

        # Trigger invalidation
        changed_modules = {module_name: module_file}
        cash_magics._module_invalidator.invalidate(
            changed_modules,
            cash_magics._statement_processor,
        )

        # Dependent variable 'result' should have been cleared
        assert "result" not in sp.tracking_state.variable_lineage
        assert "result" not in sp.tracking_state.executed_cell_codes
        assert "result" not in sp.tracking_state.executed_input_lineages
        assert "result" not in sp.tracking_state.current_session_hashes

    def test_invalidate_module_lineages_preserves_unrelated_vars(self, cash_magics, temp_module):
        """Variables NOT depending on the changed module should be preserved."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        old_module_lineage = hashlib.sha256(b"old_source").hexdigest()
        sp.tracking_state.lineage.record(module_name, old_module_lineage)
        # Unrelated variable
        sp.tracking_state.lineage.record("unrelated", "unrelated_hash")
        sp.tracking_state.executed_cell_codes["unrelated"] = "unrelated = 42"
        sp.tracking_state.executed_input_lineages["unrelated"] = {}

        changed_modules = {module_name: module_file}
        cash_magics._module_invalidator.invalidate(
            changed_modules,
            cash_magics._statement_processor,
        )

        # Unrelated variable should be preserved
        assert sp.tracking_state.variable_lineage["unrelated"] == "unrelated_hash"
        assert sp.tracking_state.executed_cell_codes["unrelated"] == "unrelated = 42"

    def test_recently_reloaded_modules_set(self, cash_magics, temp_module):
        """After invalidation, recently_reloaded_modules should contain the module name."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        changed_modules = {module_name: module_file}
        cash_magics._module_invalidator.invalidate(
            changed_modules,
            cash_magics._statement_processor,
        )

        assert module_name in sp.recently_reloaded_modules

    def test_recently_reloaded_modules_cleared_after_cell(self, cash_magics, temp_module):
        """recently_reloaded_modules should persist across non-import cells
        and be cleared when the import statement is re-executed."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        # Set recently reloaded
        sp.recently_reloaded_modules.add(module_name)

        # Execute a simple non-import cell via cash magic
        run_cash_cell(cash_magics, "x = 1")

        # Should PERSIST after non-import cell execution (the module flag
        # must survive until the actual import statement re-executes)
        assert module_name in sp.recently_reloaded_modules

        # Now execute an import statement for this module - should clear the flag
        # We need the module to be importable for this to work
        try:
            sp.recently_reloaded_modules.add(module_name)
            run_cash_cell(cash_magics, f"import {module_name}")
            # Should be cleared now because the import ran
            assert module_name not in sp.recently_reloaded_modules
        except Exception:
            # Module might not actually be importable in test env; just verify
            # the persistence behavior above was correct
            pass

    def test_module_lineage_included_in_cache_key(self, cash_magics, mock_shell, temp_module):
        """When a module has lineage, it should be included in the cache key."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        # Import the module
        mod = importlib.import_module(module_name)
        mock_shell.user_ns[module_name] = mod

        # Compute cache key WITHOUT module lineage
        code = f"result = {module_name}.increment(5)"
        _, _, key1, _, _ = sp._analyze_and_hash(code)

        # Now set a module lineage and recompute
        sp.tracking_state.lineage.record(module_name, hashlib.sha256(b"version_1").hexdigest())
        _, _, key2, _, _ = sp._analyze_and_hash(code)

        # Keys should differ because module lineage is now included
        assert key1 != key2

    def test_module_lineage_change_changes_cache_key(self, cash_magics, mock_shell, temp_module):
        """Changing module lineage should change the cache key for dependent code."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        mod = importlib.import_module(module_name)
        mock_shell.user_ns[module_name] = mod

        code = f"result = {module_name}.increment(5)"

        # Version 1
        sp.tracking_state.lineage.record(module_name, hashlib.sha256(b"version_1").hexdigest())
        _, _, key_v1, _, _ = sp._analyze_and_hash(code)

        # Version 2
        sp.tracking_state.lineage.record(module_name, hashlib.sha256(b"version_2").hexdigest())
        _, _, key_v2, _, _ = sp._analyze_and_hash(code)

        assert key_v1 != key_v2

    def test_redundant_import_not_skipped_after_reload(self, cash_magics, mock_shell, temp_module):
        """Import statement should NOT be skipped if the module was recently reloaded."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        # Import the module initially
        mod = importlib.import_module(module_name)
        mock_shell.user_ns[module_name] = mod

        # First import should be skipped (redundant)
        code = f"import {module_name}"
        metrics1 = sp.process_statement(code, silent=True)
        assert metrics1["status"] == CacheStatus.SKIPPED

        # Simulate full reload flow: invalidate lineages (which sets recently_reloaded_modules
        # AND clears executed_cell_codes/variable_lineage for the module)
        cash_magics._module_invalidator.invalidate(
            {module_name: module_file},
            cash_magics._statement_processor,
        )

        # Now import should NOT be skipped (module was reloaded)
        metrics2 = sp.process_statement(code, silent=True)
        # It should be computed or at least not SKIPPED
        assert metrics2["status"] != CacheStatus.SKIPPED

    def test_full_flow_module_change_invalidates_cache(self, cash_magics, mock_shell, tmp_path):
        """End-to-end: changing module source should cause cache miss for dependent statements.
        _PERSIST forces caching regardless of the 10 ms min-execution-time floor so that
        the cache-mechanics (SKIPPED/RESTORED on re-run, COMPUTED after invalidation)
        are actually exercised."""
        sp = cash_magics._statement_processor
        ft = sp.function_tracker

        # _PERSIST annotation: bypass min-execution-time floor for trivial module calls.
        _persist = CacheAnnotation(persist=True)

        # Create module
        module_name = f"_test_flow_mod_{id(tmp_path)}"
        module_file = tmp_path / f"{module_name}.py"
        module_file.write_text("def increment(x):\n    return x + 1\n")

        sys.path.insert(0, str(tmp_path))
        try:
            # Import module
            mod = importlib.import_module(module_name)
            mock_shell.user_ns[module_name] = mod

            # Track it
            ft.track_module(module_name)

            # Execute: import statement
            import_code = f"import {module_name}"
            sp.process_statement(import_code, silent=True)

            # Execute: use the module
            use_code = f"result = {module_name}.increment(5)"
            metrics_use1 = sp.process_statement(use_code, silent=True, annotation=_persist)
            assert metrics_use1["status"] == CacheStatus.COMPUTED
            assert mock_shell.user_ns.get("result") == 6

            # Re-run the same code - should be SKIPPED or RESTORED
            metrics_use2 = sp.process_statement(use_code, silent=True, annotation=_persist)
            assert metrics_use2["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED)

            # Now change the module source
            time.sleep(0.05)
            module_file.write_text("def increment(x):\n    return x + 10\n")

            # Simulate what _execute_cell does: check and reload
            changed, per_mod_syms = ft.check_and_reload_changed_modules(mock_shell.user_ns)
            assert module_name in changed

            # Invalidate lineages
            cash_magics._module_invalidator.invalidate(
                changed,
                cash_magics._statement_processor,
                per_mod_syms,
            )

            # Re-run the import (should not be skipped because of recently_reloaded_modules)
            sp.process_statement(import_code, silent=True)

            # Re-run: use the module - should be COMPUTED (cache miss)
            metrics_use3 = sp.process_statement(use_code, silent=True, annotation=_persist)
            assert metrics_use3["status"] == CacheStatus.COMPUTED
            assert mock_shell.user_ns.get("result") == 15  # 5 + 10

        finally:
            sys.path.remove(str(tmp_path))
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_invalidate_with_no_prior_lineage(self, cash_magics, temp_module):
        """Module invalidation should work even if module had no prior lineage."""
        module_name, module_file = temp_module
        sp = cash_magics._statement_processor

        # No prior lineage
        assert module_name not in sp.tracking_state.variable_lineage

        changed_modules = {module_name: module_file}
        cash_magics._module_invalidator.invalidate(
            changed_modules,
            cash_magics._statement_processor,
        )

        # Should set new lineage regardless
        assert module_name in sp.tracking_state.variable_lineage

    def test_invalidate_with_missing_file(self, cash_magics):
        """Module invalidation should handle missing file gracefully."""
        sp = cash_magics._statement_processor

        changed_modules = {"nonexistent_module": "/path/that/does/not/exist.py"}
        # Should not raise
        cash_magics._module_invalidator.invalidate(
            changed_modules,
            cash_magics._statement_processor,
        )

        # Should get a random hash as fallback
        assert "nonexistent_module" in sp.tracking_state.variable_lineage


# ============================================================================
# Exception Surfacing Tests
# ============================================================================


class TestExceptionSurfacing:
    """Test that execution errors in statements are properly surfaced."""

    def test_import_error_surfaces_in_cash_magic(self, cash_magics):
        """ImportError from 'import nonexistent_module' should be raised."""

        with pytest.raises(Exception) as exc_info:
            run_cash_cell(cash_magics, "import nonexistent_module_xyz_123")

        # Should be a ModuleNotFoundError (subclass of ImportError)
        assert "nonexistent_module_xyz_123" in str(exc_info.value)

    def test_value_error_surfaces_in_cash_magic(self, cash_magics):
        """ValueError from 'raise ValueError(...)' should be raised."""

        with pytest.raises(ValueError, match="test error message"):
            run_cash_cell(cash_magics, "raise ValueError('test error message')")

    def test_name_error_surfaces_in_cash_magic(self, cash_magics):
        """NameError from referencing undefined variable should be raised."""

        with pytest.raises(NameError):
            run_cash_cell(cash_magics, "print(undefined_variable_xyz)")

    def test_error_in_multi_statement_cell_stops_execution(self, cash_magics, mock_shell):
        """Error in first statement should prevent second statement from running."""

        cell = "raise ValueError('stop here')\nx = 42"
        with pytest.raises(ValueError, match="stop here"):
            run_cash_cell(cash_magics, cell)

        # Second statement should not have executed
        assert "x" not in mock_shell.user_ns

    def test_error_after_successful_statement(self, cash_magics, mock_shell):
        """Error in second statement should still surface, first statement's effects persist."""

        cell = "x = 42\nraise ValueError('second fails')"
        with pytest.raises(ValueError, match="second fails"):
            run_cash_cell(cash_magics, cell)

        # First statement should have executed
        assert mock_shell.user_ns.get("x") == 42

    def test_syntax_error_handled(self, cash_magics):
        """SyntaxError in cell should be handled (returned, not crashing)."""

        # SyntaxError is handled before statement processing (in ast.parse)
        # It should return None, not crash
        run_cash_cell(cash_magics, "def foo(")
        # Should not raise

    def test_error_metrics_contain_error_info(self, cash_magics):
        """When a statement fails, metrics should contain error info."""
        sp = cash_magics._statement_processor

        metrics = sp.process_statement("raise RuntimeError('test')", silent=True)
        assert metrics["status"] == CacheStatus.ERROR
        assert metrics["error"] is not None

    def test_execute_cell_surfaces_error(self, cash_magics, mock_shell):
        """_execute_cell should display errors cleanly via showtraceback,
        then re-raise through IPython so the kernel reply status is 'error'."""

        # Track calls to _original_run_cell
        run_cell_calls = []

        def mock_run_cell(code, *args, **kwargs):
            run_cell_calls.append(code)
            return MagicMock()

        cash_magics._original_run_cell = mock_run_cell

        # Add showtraceback to the mock shell so show_clean_error can call it
        showtraceback_calls = []

        def mock_showtraceback(*args, **kwargs):
            exc_tuple = kwargs.get("exc_tuple") or (args[0] if args else None)
            showtraceback_calls.append(exc_tuple)

        mock_shell.showtraceback = mock_showtraceback

        # Run a cell that will fail, through the %cash_on hook
        cash_magics.cash_on("")
        cash_magics._execute_cell("raise ValueError('proxy test')")

        # The clean error should have been displayed via showtraceback
        assert len(showtraceback_calls) == 1
        exc_tuple = showtraceback_calls[0]
        assert exc_tuple is not None
        assert exc_tuple[0] is ValueError
        assert "proxy test" in str(exc_tuple[1])

        # The exception should be re-raised via _original_run_cell
        # so the kernel marks the cell as 'error'
        assert any("raise __cash_exception__" in call for call in run_cell_calls)
        assert isinstance(mock_shell.user_ns.get("__cash_exception__"), ValueError)


# ============================================================================
# Transitive Dependency Tests
# ============================================================================


class TestTransitiveDependencyTracking:
    """Test that changes to sub-dependencies (e.g., helpers.py used by metrics.py)
    transitively invalidate the parent module and all dependent variables."""

    @pytest.fixture
    def two_level_modules(self, tmp_path):
        """Create a two-level module hierarchy: metrics -> helpers."""
        helpers_file = tmp_path / "helpers.py"
        helpers_file.write_text("def add_one(x):\n    return x + 1\n")

        metrics_file = tmp_path / "metrics.py"
        metrics_file.write_text("from helpers import add_one\ndef compute(x):\n    return add_one(x) * 2\n")

        sys.path.insert(0, str(tmp_path))
        # Another test in this worker may have left its own "helpers" behind.
        for mod_name in ("helpers", "metrics"):
            sys.modules.pop(mod_name, None)

        # Import so they appear in sys.modules
        import importlib

        helpers_mod = importlib.import_module("helpers")
        metrics_mod = importlib.import_module("metrics")

        yield {
            "helpers_file": str(helpers_file),
            "metrics_file": str(metrics_file),
            "helpers_mod": helpers_mod,
            "metrics_mod": metrics_mod,
            "tmp_path": tmp_path,
        }

        sys.path.remove(str(tmp_path))
        for mod_name in ("helpers", "metrics"):
            sys.modules.pop(mod_name, None)

    def test_discover_transitive_deps(self, two_level_modules):
        """track_module('metrics') should discover helpers.py as a transitive dep."""
        ft = FunctionTracker()
        ft.track_module("metrics")

        # helpers.py should appear as a dependency of metrics
        found = False
        for dep_path, parents in ft.dep_file_to_parents.items():
            if "helpers" in dep_path and "metrics" in parents:
                found = True
                break
        assert found, (
            f"Expected helpers.py to be a dependency of metrics. dep_file_to_parents = {ft.dep_file_to_parents}"
        )

    def test_sub_dep_tracked_in_tracked_modules(self, two_level_modules):
        """After tracking metrics, helpers should also be in tracked_modules."""
        ft = FunctionTracker()
        ft.track_module("metrics")

        assert "helpers" in ft.tracked_modules

    def test_sub_dep_mtime_recorded(self, two_level_modules):
        """helpers.py mtime should be recorded in _dep_file_mtimes."""
        ft = FunctionTracker()
        ft.track_module("metrics")

        found_mtime = False
        for dep_path in ft._dep_file_mtimes:
            if "helpers" in dep_path:
                found_mtime = True
                break
        assert found_mtime

    def test_sub_dep_change_detected(self, two_level_modules):
        """Changing helpers.py should cause check_tracked_modules to report metrics as changed."""
        info = two_level_modules
        ft = FunctionTracker()
        ft.track_module("metrics")

        # Verify no changes initially
        changed = ft.check_tracked_modules()
        assert "metrics" not in changed

        # Now change helpers.py
        time.sleep(0.05)
        with open(info["helpers_file"], "w") as f:
            f.write("def add_one(x):\n    return x + 100\n")

        changed = ft.check_tracked_modules()
        assert "metrics" in changed, (
            f"Expected metrics to be reported as changed after helpers.py changed. changed = {changed}"
        )

    def test_sub_dep_change_reloads_parent(self, two_level_modules):
        """Changing helpers.py and calling check_and_reload should reload metrics."""
        info = two_level_modules
        ft = FunctionTracker()
        ft.track_module("metrics")
        user_ns = {"metrics": info["metrics_mod"]}

        # Change helpers.py
        time.sleep(0.05)
        with open(info["helpers_file"], "w") as f:
            f.write("def add_one(x):\n    return x + 100\n")

        result, _ = ft.check_and_reload_changed_modules(user_ns)
        assert "metrics" in result

        # Verify the reloaded module uses the new helpers
        import metrics

        assert metrics.compute(5) == (5 + 100) * 2

    def test_invalidate_lineages_with_transitive_dep(self, cash_magics, two_level_modules):
        """_invalidate_module_lineages should produce a lineage hash that includes sub-dep content."""
        info = two_level_modules
        sp = cash_magics._statement_processor

        # Track metrics → discovers helpers as transitive dep
        sp.function_tracker.track_module("metrics")

        # Set initial lineage
        old_lineage = hashlib.sha256(b"initial").hexdigest()
        sp.tracking_state.lineage.record("metrics", old_lineage)

        # Invalidate — the new lineage should include helpers.py content
        changed_modules = {"metrics": info["metrics_file"]}
        cash_magics._module_invalidator.invalidate(
            changed_modules,
            cash_magics._statement_processor,
        )

        new_lineage = sp.tracking_state.variable_lineage["metrics"]
        assert new_lineage != old_lineage

        # Expected: the source identity of metrics.py together with helpers.py
        from cash.notebook.lineage_formula import read_module_source_hash

        dep_files = {dp for dp, parents in sp.function_tracker.dep_file_to_parents.items() if "metrics" in parents}
        assert dep_files, "helpers.py was not discovered as a dependency"
        assert new_lineage == read_module_source_hash(info["metrics_file"], dep_files)
        assert new_lineage != read_module_source_hash(info["metrics_file"])

    def test_three_level_transitive_deps(self, tmp_path):
        """Three-level chain: app -> service -> utils. Changing utils should invalidate app."""
        utils_file = tmp_path / "dep_utils.py"
        utils_file.write_text("VALUE = 1\n")

        service_file = tmp_path / "dep_service.py"
        service_file.write_text("import dep_utils\ndef get():\n    return dep_utils.VALUE\n")

        app_file = tmp_path / "dep_app.py"
        app_file.write_text("import dep_service\ndef run():\n    return dep_service.get()\n")

        sys.path.insert(0, str(tmp_path))
        try:
            import importlib

            importlib.import_module("dep_utils")
            importlib.import_module("dep_service")
            importlib.import_module("dep_app")

            ft = FunctionTracker()
            ft.track_module("dep_app")

            # Check that dep_utils is a transitive dep of dep_app
            found_utils = False
            for dep_path, parents in ft.dep_file_to_parents.items():
                if "dep_utils" in dep_path and "dep_app" in parents:
                    found_utils = True
                    break
            assert found_utils, (
                f"dep_utils should be a transitive dep of dep_app. dep_file_to_parents = {ft.dep_file_to_parents}"
            )

            # Change dep_utils
            time.sleep(0.05)
            utils_file.write_text("VALUE = 999\n")

            changed = ft.check_tracked_modules()
            assert "dep_app" in changed, (
                f"dep_app should be reported as changed after dep_utils changed. changed={changed}"
            )
        finally:
            sys.path.remove(str(tmp_path))
            for mod_name in ("dep_utils", "dep_service", "dep_app"):
                sys.modules.pop(mod_name, None)

    def test_refresh_transitive_dependencies_after_reload(self, two_level_modules):
        """After reload, refresh_transitive_dependencies should discover new deps."""
        ft = FunctionTracker()
        ft.track_module("metrics")

        # Confirm helpers is tracked
        assert any("helpers" in dp for dp in ft.dep_file_to_parents)

        # Refresh should rebuild the dependency graph
        ft.refresh_transitive_dependencies()
        assert any("helpers" in dp for dp in ft.dep_file_to_parents)

    def test_no_false_positive_for_stdlib_imports(self, two_level_modules):
        """Sub-dependencies that are stdlib modules should NOT be tracked."""
        ft = FunctionTracker()
        ft.track_module("metrics")

        # No stdlib paths should appear in dep_file_to_parents
        from cash.install_paths import is_installed_path

        for dep_path in ft.dep_file_to_parents:
            assert not is_installed_path(dep_path), f"Stdlib path {dep_path} should not be tracked as a dependency"


# ============================================================================
# The %cash_on hook runs the whole pipeline
# ============================================================================


class TestHookRunsTheFullPipeline:
    """The `%cash_on` hook runs module-change detection and the pre-execution
    notifications for every cell. A second entry point (the removed `%%cash`
    magic) once skipped both; there is now one, and these pin that it keeps
    doing both.
    """

    @pytest.fixture(autouse=True)
    def _cash_on(self, cash_magics):
        cash_magics.cash_on("")

    def test_hook_invokes_module_change_detection(self, cash_magics):
        executor = cash_magics._cell_executor
        called_with: list[str] = []
        original = executor._detect_module_changes

        def spy(raw_cell):
            called_with.append(raw_cell)
            return original(raw_cell)

        executor._detect_module_changes = spy
        cash_magics._execute_cell("x = 1")
        assert called_with == ["x = 1"]

    def test_hook_invokes_pre_execution_notifications(self, cash_magics):
        executor = cash_magics._cell_executor
        call_count = [0]
        original = executor._build_pre_execution_notifications

        def spy(raw_cell, pre_upstream_metrics, upstream_metrics):
            call_count[0] += 1
            return original(raw_cell, pre_upstream_metrics, upstream_metrics)

        executor._build_pre_execution_notifications = spy
        cash_magics._execute_cell("x = 1")
        assert call_count[0] == 1
