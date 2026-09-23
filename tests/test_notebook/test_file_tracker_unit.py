"""Unit tests for the FileAccessTracker and FileDependencyRegistry.

Directly imports and tests ``cash.tracking.file_tracker`` to verify file
dependency interception, registry handlers, and patch/unpatch lifecycle.
"""

import os
from pathlib import Path

import pytest

from cash.tracking.file_tracker import (
    FileAccessTracker,
    file_registry,
)

# ---------------------------------------------------------------------------
# FileDependencyRegistry
# ---------------------------------------------------------------------------


class TestFileDependencyRegistry:
    """Test the process-wide registry and handler registration."""

    def test_one_registry_per_process(self):
        assert file_registry() is file_registry()

    def test_has_default_handlers(self):
        registry = file_registry()
        assert "pandas" in registry.handlers
        assert "sqlite3" in registry.handlers
        # `open` arrives as an audit event, so nothing wraps it.
        assert "builtins" not in registry.handlers

    def test_get_handlers_for_module(self):
        registry = file_registry()
        sqlite_handlers = registry.get_handlers_for_module("sqlite3")
        assert len(sqlite_handlers) >= 1
        func_names = [name for name, _ in sqlite_handlers]
        assert "connect" in func_names

    def test_get_handlers_for_unknown_module(self):
        registry = file_registry()
        handlers = registry.get_handlers_for_module("nonexistent_module_xyz")
        assert handlers == []


# ---------------------------------------------------------------------------
# FileAccessTracker — builtin open() interception
# ---------------------------------------------------------------------------


class TestFileAccessTrackerOpen:
    """Test that open() calls are intercepted and file paths tracked."""

    def test_tracks_read_open(self, tmp_path: Path):
        test_file = tmp_path / "data.txt"
        test_file.write_text("hello", encoding="utf-8")

        tracker = FileAccessTracker()
        with tracker, open(str(test_file), "r", encoding="utf-8") as f:
            _ = f.read()

        accessed = tracker.get_accessed_files()
        # The canonical path should be tracked (forward slashes)
        normalised = os.path.realpath(str(test_file)).replace(os.sep, "/")
        assert normalised in accessed

    def test_does_not_track_write_only_open(self, tmp_path: Path):
        test_file = tmp_path / "output.txt"

        tracker = FileAccessTracker()
        with tracker, open(str(test_file), "w", encoding="utf-8") as f:
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
        test_file.write_text("data", encoding="utf-8")

        tracker = FileAccessTracker()
        with tracker:
            pass
        # After exit: a read should NOT be recorded by the tracker.
        with open(str(test_file), "r", encoding="utf-8") as f:
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
        test_file.write_text("a,b\n1,2", encoding="utf-8")

        tracker = FileAccessTracker()
        with tracker, open(str(test_file), "r", encoding="utf-8") as f:
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
        test_file.write_text("ns content", encoding="utf-8")

        user_ns = {"open": open}
        tracker = FileAccessTracker(user_ns=user_ns)
        with tracker:  # noqa: SIM117
            # Use the user_ns version of open
            with user_ns["open"](str(test_file), "r") as f:
                _ = f.read()

        normalised = os.path.realpath(str(test_file)).replace(os.sep, "/")
        assert normalised in tracker.get_accessed_files()

    def test_user_ns_open_inactive_outside_with_block(self, tmp_path: Path):
        """Outside a `with tracker:` block, user_ns['open'] does not record reads.

        The wrapper installed on user_ns['open'] is permanent under the
        ContextVar design (patches are install-once-per-process), but the
        wrapper is a no-op when no tracker is active.
        """
        path = tmp_path / "data.txt"
        path.write_text("hello", encoding="utf-8")
        user_ns: dict = {"open": open}
        tracker = FileAccessTracker(user_ns=user_ns)
        with tracker:
            # Inside the block, reads should be tracked.
            with user_ns["open"](path) as f:
                f.read()
        assert any(r.endswith("data.txt") for r in tracker.get_accessed_files())

        # Outside the block, a read against the (still-patched) user_ns['open']
        # must NOT be added to this tracker.
        before = set(tracker.get_accessed_files())
        other_path = tmp_path / "after.txt"
        other_path.write_text("nope", encoding="utf-8")
        with user_ns["open"](other_path) as f:
            f.read()
        after = set(tracker.get_accessed_files())
        assert before == after, f"reads outside `with` block leaked into tracker: {after - before}"


# ---------------------------------------------------------------------------
# FileAccessTracker — wrappers installed only while a tracker is open
# ---------------------------------------------------------------------------


class TestInstalledWhileInUse:
    """The wrappers consult ``active_tracker`` at call time, so one install
    serves every tracker; the first tracker to open installs them and the
    last one to close puts the originals back."""

    def test_wrapper_is_noop_when_no_tracker_active(self, tmp_path: Path):
        """Read through a wrapper bound while a tracker was open, after it
        closed: nothing is recorded, and the read passes through."""
        import sqlite3

        db = tmp_path / "stray.db"
        tracker = FileAccessTracker()
        with tracker:
            connect = sqlite3.connect
            assert getattr(connect, "_is_file_tracker_patch", False)

        connect(str(db)).close()
        assert tracker.get_accessed_files() == set()

    def test_sequential_trackers_are_isolated(self, tmp_path: Path):
        """Two trackers used back-to-back must each see only their own
        block's reads — no cross-contamination."""
        p_a = tmp_path / "a.txt"
        p_a.write_text("a", encoding="utf-8")
        p_b = tmp_path / "b.txt"
        p_b.write_text("b", encoding="utf-8")

        t1 = FileAccessTracker()
        with t1, open(str(p_a), encoding="utf-8") as f:
            f.read()

        t2 = FileAccessTracker()
        with t2, open(str(p_b), encoding="utf-8") as f:
            f.read()

        assert any(r.endswith("a.txt") for r in t1.get_accessed_files())
        assert not any(r.endswith("b.txt") for r in t1.get_accessed_files())
        assert any(r.endswith("b.txt") for r in t2.get_accessed_files())
        assert not any(r.endswith("a.txt") for r in t2.get_accessed_files())


class TestRemoteUrlChannel:
    """A remote read is a real dependency; it just isn't a stat-able one.

    ``pd.read_parquet("s3://bucket/key")`` hands the tracker the URL as given.
    ``os.path.realpath`` mangles it into a bogus local path, nothing resolves,
    and ``snapshot_file_deps`` drops it — so the entry would be stored with no
    dependency at all and hit forever even after the object changed. URLs are
    therefore kept on their own channel and tracked by the store's validator.
    """

    def test_a_url_lands_on_the_remote_channel_not_the_file_one(self):
        tracker = FileAccessTracker()
        tracker._track_path("s3://bucket/events.parquet")

        assert tracker.get_accessed_remote_urls() == {"s3://bucket/events.parquet"}
        assert tracker.get_accessed_files() == set(), (
            "a URL must never enter the file set - every consumer of it stats and hashes its members"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "s3://bucket/key.parquet",
            "gs://bucket/key.parquet",
            "az://container/key.parquet",
            "https://example.com/data.csv",
            "http://example.com/data.csv",
        ],
    )
    def test_recognised_schemes(self, url):
        tracker = FileAccessTracker()
        tracker._track_path(url)
        assert tracker.get_accessed_remote_urls() == {url}

    def test_tracking_a_url_does_not_warn(self):
        """The old 'cannot track this' warning for URLs is obsolete: it can now."""
        import warnings

        from cash.exceptions import CashCacheIneffectiveWarning

        tracker = FileAccessTracker()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker._track_path("s3://bucket/events.parquet")

        hits = [w for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
        assert hits == [], f"a tracked read must not warn, got {[str(h.message) for h in hits]}"

    def test_local_paths_stay_on_the_file_channel(self, tmp_path):
        """A Windows drive letter is not a URL scheme, and ``file://`` names a
        path that can genuinely be stat'ed."""
        real = tmp_path / "data.csv"
        real.write_text("a,b\n1,2\n", encoding="utf-8")
        tracker = FileAccessTracker()
        tracker._track_path(str(real))
        tracker._track_path(r"C:\Users\someone\data.csv")

        assert tracker.get_accessed_remote_urls() == set()
        assert tracker.get_accessed_files(), "local paths must still be tracked"

    def test_propagation_reaches_the_enclosing_tracker(self):
        """An outer cached function must inherit an inner one's remote deps,
        exactly as it inherits its file deps."""
        outer = FileAccessTracker(propagate_to_parent=True)
        with outer:
            inner = FileAccessTracker(propagate_to_parent=True)
            with inner:
                inner._track_path("s3://bucket/events.parquet")

        assert outer.get_accessed_remote_urls() == {"s3://bucket/events.parquet"}
