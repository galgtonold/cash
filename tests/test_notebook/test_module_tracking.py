"""
Tests for module file tracking and %cash_track magic.

Tests the FunctionTracker module tracking features:
- track_module: Register a module for file change detection
- check_tracked_modules: Detect file modifications
- reload_module: Force reload of changed modules
- %cash_track magic: IPython line magic interface
"""

import sys
import time
import pytest
from unittest.mock import MagicMock

from cash.notebook.function_tracker import FunctionTracker
from cash.notebook.magics import CashMagics
from cash.core import Cash
from cash.backends import InMemoryBackend
from traitlets.config.configurable import Configurable


# ============================================================================
# Fixtures
# ============================================================================


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
def tracker():
    """Provide a fresh FunctionTracker."""
    ft = FunctionTracker()
    yield ft
    ft.clear()


@pytest.fixture
def magics_fixture():
    """Provide CashMagics instance for testing %cash_track."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


@pytest.fixture
def temp_module(tmp_path):
    """Create a temporary Python module file for tracking tests.

    Returns (module_name, module_path) and cleans up sys.modules/path on exit.
    """
    module_name = f"_test_tracking_mod_{id(tmp_path)}"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text("def helper(x):\n    return x * 2\n")

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
        assert module_name in tracker._tracked_modules
        assert module_name in tracker._module_mtimes

    def test_track_unimported_module(self, tracker):
        """track_module returns None for module not in sys.modules."""
        result = tracker.track_module("nonexistent_xyz_module")
        # Module name is added to tracked set even if not yet imported
        assert "nonexistent_xyz_module" in tracker._tracked_modules
        assert result is None

    def test_track_builtin_module(self, tracker):
        """Track a built-in module (no __file__)."""
        result = tracker.track_module("builtins")
        # builtins has no __file__, should return None
        assert result is None
        assert "builtins" in tracker._tracked_modules

    def test_track_module_records_mtime(self, tracker, temp_module):
        """Module mtime is recorded on tracking."""
        module_name, module_file = temp_module
        import importlib
        importlib.import_module(module_name)

        tracker.track_module(module_name)
        mtime = tracker._module_mtimes.get(module_name)
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
        with open(module_file, 'w') as f:
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
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x * 3\n")

        changed = tracker.check_tracked_modules()
        assert module_name in changed

        # Second check without further changes
        changed2 = tracker.check_tracked_modules()
        assert len(changed2) == 0

    def test_unimported_module_not_in_changes(self, tracker):
        """Module in tracked set but not imported doesn't appear in changes."""
        tracker._tracked_modules.add("nonexistent_module_xyz")
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
        with open(module_file, 'w') as f:
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
        with open(module_file, 'w') as f:
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
        assert len(tracker._tracked_modules) > 0
        assert len(tracker._module_mtimes) > 0

        tracker.clear()
        assert len(tracker._tracked_modules) == 0
        assert len(tracker._module_mtimes) == 0
        assert len(tracker._source_cache) == 0
        assert len(tracker._function_hashes) == 0


# ============================================================================
# %cash_track magic tests
# ============================================================================


class TestCashTrackMagic:
    """Tests for the %cash_track IPython line magic."""

    def test_list_no_modules(self, magics_fixture, capsys):
        """--list with no tracked modules shows help message."""
        magics, shell, backend = magics_fixture
        magics.cash_track("--list")
        captured = capsys.readouterr()
        assert "No modules tracked" in captured.out

    def test_list_empty_args(self, magics_fixture, capsys):
        """No arguments defaults to --list behavior."""
        magics, shell, backend = magics_fixture
        magics.cash_track("")
        captured = capsys.readouterr()
        assert "No modules tracked" in captured.out

    def test_track_module(self, magics_fixture, temp_module, capsys):
        """Track a real module via magic command."""
        magics, shell, backend = magics_fixture
        module_name, module_file = temp_module

        # Import it first
        import importlib
        importlib.import_module(module_name)

        magics.cash_track(module_name)
        captured = capsys.readouterr()
        assert f"Tracking module '{module_name}'" in captured.out

    def test_track_nonexistent_module(self, magics_fixture, capsys):
        """Track a module that doesn't exist shows error."""
        magics, shell, backend = magics_fixture
        magics.cash_track("totally_nonexistent_module_xyz_999")
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_track_and_list(self, magics_fixture, temp_module, capsys):
        """Track a module then list shows it."""
        magics, shell, backend = magics_fixture
        module_name, module_file = temp_module

        import importlib
        importlib.import_module(module_name)

        magics.cash_track(module_name)
        capsys.readouterr()  # clear capture

        magics.cash_track("--list")
        captured = capsys.readouterr()
        assert module_name in captured.out
        assert "Tracked modules:" in captured.out

    def test_check_no_changes(self, magics_fixture, temp_module, capsys):
        """--check with no changes reports clean."""
        magics, shell, backend = magics_fixture
        module_name, module_file = temp_module

        import importlib
        importlib.import_module(module_name)

        magics.cash_track(module_name)
        capsys.readouterr()  # clear

        magics.cash_track("--check")
        captured = capsys.readouterr()
        assert "No tracked modules have changed" in captured.out

    def test_check_detects_change(self, magics_fixture, temp_module, capsys):
        """--check detects file modification and auto-reloads."""
        magics, shell, backend = magics_fixture
        module_name, module_file = temp_module

        import importlib
        importlib.import_module(module_name)

        magics.cash_track(module_name)
        capsys.readouterr()  # clear

        # Modify the module file
        time.sleep(0.1)
        with open(module_file, 'w') as f:
            f.write("def helper(x):\n    return x * 100\n")

        magics.cash_track("--check")
        captured = capsys.readouterr()
        assert "Changed modules:" in captured.out
        assert module_name in captured.out
        assert "Reloaded:" in captured.out

    def test_reload_flag(self, magics_fixture, temp_module, capsys):
        """--reload forces module reload."""
        magics, shell, backend = magics_fixture
        module_name, module_file = temp_module

        import importlib
        mod = importlib.import_module(module_name)
        assert mod.helper(5) == 10

        magics.cash_track(f"{module_name} --reload")
        captured = capsys.readouterr()
        assert "Tracking module" in captured.out
        assert "Reloaded:" in captured.out

    def test_auto_import_and_track(self, magics_fixture, temp_module, capsys):
        """Module not yet imported gets auto-imported and tracked."""
        magics, shell, backend = magics_fixture
        module_name, module_file = temp_module

        # Don't import it - let the magic do it
        assert module_name not in sys.modules

        magics.cash_track(module_name)
        captured = capsys.readouterr()
        assert f"Tracking module '{module_name}'" in captured.out
        # Module should now be imported
        assert module_name in sys.modules
