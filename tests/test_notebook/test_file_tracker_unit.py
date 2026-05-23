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

    def test_tracker_inactive_outside_with_block(self, tmp_path: Path):
        """After the ContextVar refactor, ``builtins.open`` stays patched
        for the lifetime of the process, but the wrapper is a no-op when
        no tracker is active. This test replaces the old
        ``test_unpatches_on_exit`` which checked that ``builtins.open``
        was literally restored to the original on exit — that contract no
        longer holds. The functional guarantee — file reads outside the
        ``with`` block are not recorded — is still enforced and is what
        this test now verifies.
        """
        test_file = tmp_path / "outside.txt"
        test_file.write_text("data")

        tracker = FileAccessTracker()
        with tracker:
            pass
        # After exit: a read should NOT be recorded by the tracker.
        with open(str(test_file), "r") as f:
            _ = f.read()
        assert test_file.name not in {p.rsplit("/", 1)[-1] for p in tracker.get_accessed_files()}


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


# ---------------------------------------------------------------------------
# FileAccessTracker — self-healing of leaked wrappers
# ---------------------------------------------------------------------------

class TestPermanentInstallSemantics:
    """After the ContextVar refactor the per-tracker
    install/unpatch dance is replaced by an install-once-permanently
    dispatcher that consults ``_active_tracker``. This eliminates the
    leaked-wrapper class of bug entirely — once installed, the wrapper
    stays in place but is a no-op when no tracker is active. These
    tests pin the new contract.
    """

    def test_wrapper_is_noop_when_no_tracker_active(self, tmp_path: Path):
        """With patches installed but no active tracker, reads must not
        be recorded anywhere — there's no tracker to record into."""
        import builtins

        # Trigger the install-once if some other test in this file
        # hasn't already.
        with FileAccessTracker():
            pass

        # Patches are permanent — the wrapper is on builtins.open.
        assert getattr(builtins.open, "_is_file_tracker_patch", False), (
            "Expected the dispatcher wrapper to be permanently installed."
        )

        # Now do a read with no active tracker: nothing tracks it.
        test_file = tmp_path / "stray.txt"
        test_file.write_text("x")
        # The dispatcher should pass through cleanly.
        with open(str(test_file), "r") as f:
            assert f.read() == "x"

    def test_sequential_trackers_are_isolated(self, tmp_path: Path):
        """Two trackers used back-to-back must each see only their own
        block's reads — no cross-contamination."""
        p_a = tmp_path / "a.txt"; p_a.write_text("a")
        p_b = tmp_path / "b.txt"; p_b.write_text("b")

        t1 = FileAccessTracker()
        with t1, open(str(p_a)) as f:
            f.read()

        t2 = FileAccessTracker()
        with t2, open(str(p_b)) as f:
            f.read()

        assert any(r.endswith("a.txt") for r in t1.get_accessed_files())
        assert not any(r.endswith("b.txt") for r in t1.get_accessed_files())
        assert any(r.endswith("b.txt") for r in t2.get_accessed_files())
        assert not any(r.endswith("a.txt") for r in t2.get_accessed_files())
