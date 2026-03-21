"""Unit tests for the FileAccessTracker and FileDependencyRegistry.

Directly imports and tests ``cash.notebook.file_tracker`` to verify file
dependency interception, registry handlers, and patch/unpatch lifecycle.
"""

import os
from pathlib import Path

from cash.notebook.file_tracker import (
    FileAccessTracker,
    FileDependencyRegistry,
)


# ---------------------------------------------------------------------------
# FileDependencyRegistry
# ---------------------------------------------------------------------------

class TestFileDependencyRegistry:
    """Test the singleton registry and handler registration."""

    def test_singleton_pattern(self):
        r1 = FileDependencyRegistry()
        r2 = FileDependencyRegistry()
        assert r1 is r2

    def test_has_default_handlers(self):
        registry = FileDependencyRegistry()
        assert "builtins" in registry.handlers
        assert "pandas" in registry.handlers

    def test_get_handlers_for_module(self):
        registry = FileDependencyRegistry()
        builtins_handlers = registry.get_handlers_for_module("builtins")
        assert len(builtins_handlers) >= 1
        func_names = [name for name, _ in builtins_handlers]
        assert "open" in func_names

    def test_get_handlers_for_unknown_module(self):
        registry = FileDependencyRegistry()
        handlers = registry.get_handlers_for_module("nonexistent_module_xyz")
        assert handlers == []


# ---------------------------------------------------------------------------
# FileAccessTracker — builtin open() interception
# ---------------------------------------------------------------------------

class TestFileAccessTrackerOpen:
    """Test that open() calls are intercepted and file paths tracked."""

    def test_tracks_read_open(self, tmp_path: Path):
        test_file = tmp_path / "data.txt"
        test_file.write_text("hello")

        tracker = FileAccessTracker()
        with tracker, open(str(test_file), "r") as f:
            _ = f.read()

        accessed = tracker.get_accessed_files()
        # The canonical path should be tracked (forward slashes)
        normalised = os.path.realpath(str(test_file)).replace(os.sep, "/")
        assert normalised in accessed

    def test_does_not_track_write_only_open(self, tmp_path: Path):
        test_file = tmp_path / "output.txt"

        tracker = FileAccessTracker()
        with tracker, open(str(test_file), "w") as f:
            f.write("written")

        accessed = tracker.get_accessed_files()
        normalised = os.path.realpath(str(test_file)).replace(os.sep, "/")
        assert normalised not in accessed

    def test_unpatches_on_exit(self):
        import builtins

        original_open = builtins.open
        tracker = FileAccessTracker()
        with tracker:
            assert builtins.open is not original_open  # patched
        assert builtins.open is original_open  # restored


# ---------------------------------------------------------------------------
# FileAccessTracker — path normalisation
# ---------------------------------------------------------------------------

class TestPathNormalisation:
    """Paths are stored as canonical, forward-slash paths."""

    def test_path_normalised(self, tmp_path: Path):
        test_file = tmp_path / "sub" / "test.csv"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("a,b\n1,2")

        tracker = FileAccessTracker()
        with tracker, open(str(test_file), "r") as f:
            _ = f.read()

        accessed = tracker.get_accessed_files()
        for path in accessed:
            # All paths should use forward slashes
            assert "\\" not in path


# ---------------------------------------------------------------------------
# FileAccessTracker — user namespace patching
# ---------------------------------------------------------------------------

class TestUserNamespacePatching:
    """open() in user_ns is also intercepted."""

    def test_patches_open_in_user_ns(self, tmp_path: Path):
        test_file = tmp_path / "ns_test.txt"
        test_file.write_text("ns content")

        user_ns = {"open": open}
        tracker = FileAccessTracker(user_ns=user_ns)
        with tracker:  # noqa: SIM117
            # Use the user_ns version of open
            with user_ns["open"](str(test_file), "r") as f:
                _ = f.read()

        normalised = os.path.realpath(str(test_file)).replace(os.sep, "/")
        assert normalised in tracker.get_accessed_files()

    def test_restores_user_ns_open(self):
        user_ns = {"open": open}
        original = user_ns["open"]
        tracker = FileAccessTracker(user_ns=user_ns)
        with tracker:
            pass
        assert user_ns["open"] is original
