"""
Tests for module file tracking.

Tests the FunctionTracker module tracking features:
- track_module: Register a module for file change detection
- check_tracked_modules: Detect file modifications
- reload_module: Force reload of changed modules
"""

import sys
import time

import pytest

from cash.tracking.function_tracker import FunctionTracker

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tracker():
    """Provide a fresh FunctionTracker."""
    ft = FunctionTracker()
    yield ft
    ft.clear()


@pytest.fixture
def temp_module(tmp_path):
    """Create a temporary Python module file for tracking tests.

    Returns (module_name, module_path) and cleans up sys.modules/path on exit.
    """
    module_name = f"_test_tracking_mod_{id(tmp_path)}"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text("def helper(x):\n    return x * 2\n", encoding="utf-8")

    # Add to sys.path so it can be imported
    sys.path.insert(0, str(tmp_path))

    yield module_name, str(module_file)

    # Cleanup
    sys.path.remove(str(tmp_path))
    if module_name in sys.modules:
        del sys.modules[module_name]


# ============================================================================
# FunctionTracker.track_module tests
# ============================================================================


class TestTrackModule:
    """Tests for tracking modules by file path."""

    def test_track_imported_module(self, tracker, temp_module):
        """Track a module that's already imported."""
        module_name, module_file = temp_module
        import importlib

        importlib.import_module(module_name)

        result = tracker.track_module(module_name)
        assert result is not None
        assert module_name in tracker.tracked_modules
        assert module_name in tracker.module_mtimes

    def test_track_unimported_module(self, tracker):
        """track_module returns None for module not in sys.modules."""
        result = tracker.track_module("nonexistent_xyz_module")
        # Module name is added to tracked set even if not yet imported
        assert "nonexistent_xyz_module" in tracker.tracked_modules
        assert result is None

    def test_track_builtin_module(self, tracker):
        """Track a built-in module (no __file__)."""
        result = tracker.track_module("builtins")
        # builtins has no __file__, should return None
        assert result is None
        assert "builtins" in tracker.tracked_modules

    def test_track_module_records_mtime(self, tracker, temp_module):
        """Module mtime is recorded on tracking."""
        module_name, module_file = temp_module
        import importlib

        importlib.import_module(module_name)

        tracker.track_module(module_name)
        mtime = tracker.module_mtimes.get(module_name)
        assert mtime is not None
        assert isinstance(mtime, float)


# ============================================================================
# FunctionTracker.check_tracked_modules tests
# ============================================================================


class TestCheckTrackedModules:
    """Tests for detecting module file changes."""

    def test_no_changes(self, tracker, temp_module):
        """No changes detected when file hasn't been modified."""
        module_name, module_file = temp_module
        import importlib

        importlib.import_module(module_name)

        tracker.track_module(module_name)
        changed = tracker.check_tracked_modules()
        assert len(changed) == 0

    def test_detect_file_change(self, tracker, temp_module):
        """Detect when a tracked module's file is modified."""
        module_name, module_file = temp_module
        import importlib

        importlib.import_module(module_name)

        tracker.track_module(module_name)

        # Modify the file (need to ensure mtime changes)
        time.sleep(0.1)
        with open(module_file, "w", encoding="utf-8") as f:
            f.write("def helper(x):\n    return x * 3\n")

        changed = tracker.check_tracked_modules()
        assert module_name in changed

    def test_no_false_positives_after_check(self, tracker, temp_module):
        """After detecting a change, subsequent check with no new changes returns empty."""
        module_name, module_file = temp_module
        import importlib

        importlib.import_module(module_name)

        tracker.track_module(module_name)

        # Modify and detect
        time.sleep(0.1)
        with open(module_file, "w", encoding="utf-8") as f:
            f.write("def helper(x):\n    return x * 3\n")

        changed = tracker.check_tracked_modules()
        assert module_name in changed

        # Second check without further changes
        changed2 = tracker.check_tracked_modules()
        assert len(changed2) == 0

    def test_unimported_module_not_in_changes(self, tracker):
        """Module in tracked set but not imported doesn't appear in changes."""
        tracker.tracked_modules.add("nonexistent_module_xyz")
        changed = tracker.check_tracked_modules()
        assert len(changed) == 0


# ============================================================================
# FunctionTracker.reload_module tests
# ============================================================================


class TestReloadModule:
    """Tests for module reload functionality."""

    def test_reload_imported_module(self, tracker, temp_module):
        """Reload updates the module in sys.modules."""
        module_name, module_file = temp_module
        import importlib

        mod = importlib.import_module(module_name)

        tracker.track_module(module_name)

        # Verify original behavior
        assert mod.helper(5) == 10

        # Modify the module
        time.sleep(0.1)
        with open(module_file, "w", encoding="utf-8") as f:
            f.write("def helper(x):\n    return x * 3\n")

        # Reload
        success = tracker.reload_module(module_name)
        assert success is True

        # After reload, the module object in sys.modules is updated
        # Access the new function through sys.modules
        mod_reloaded = sys.modules[module_name]
        # The reloaded module's helper function should use the new code
        assert mod_reloaded.helper(5) == 15

    def test_reload_nonexistent_module(self, tracker):
        """Reload returns False for module not in sys.modules."""
        result = tracker.reload_module("nonexistent_xyz")
        assert result is False

    def test_reload_clears_source_cache(self, tracker, temp_module):
        """Reloading a module invalidates cached source hashes."""
        module_name, module_file = temp_module
        import importlib

        mod = importlib.import_module(module_name)

        # Cache the function hash
        hash1 = tracker.get_function_source_hash(mod.helper)
        assert hash1 is not None

        # Modify the module
        time.sleep(0.1)
        with open(module_file, "w", encoding="utf-8") as f:
            f.write("def helper(x):\n    return x * 3\n")

        # Reload (should clear cache)
        tracker.reload_module(module_name)

        # Get new hash - should be different
        mod_reloaded = sys.modules[module_name]
        hash2 = tracker.get_function_source_hash(mod_reloaded.helper)
        assert hash2 is not None
        assert hash1 != hash2


# ============================================================================
# FunctionTracker.clear tests
# ============================================================================


class TestClear:
    """Tests for clear() including module tracking state."""

    def test_clear_resets_module_tracking(self, tracker, temp_module):
        """clear() removes all module tracking state."""
        module_name, module_file = temp_module
        import importlib

        importlib.import_module(module_name)

        tracker.track_module(module_name)
        assert len(tracker.tracked_modules) > 0
        assert len(tracker.module_mtimes) > 0

        tracker.clear()
        assert len(tracker.tracked_modules) == 0
        assert len(tracker.module_mtimes) == 0
        assert len(tracker._source_cache) == 0
        assert len(tracker._function_hashes) == 0
