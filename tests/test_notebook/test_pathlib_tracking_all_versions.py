"""A pathlib read must register a file dependency on every supported Python.

``Path.open`` reaches the filesystem differently across versions:

    3.11+   Path.open -> io.open(self, ...)          — the io.open patch sees it
    3.10    Path.open -> Path._accessor.open(...)     — and _NormalAccessor.open
                                                        was bound to the ORIGINAL
                                                        io.open when pathlib was
                                                        first imported

So on 3.10 the io.open patch never reached pathlib, and a cell reading a file
through ``Path.read_text()`` recorded no dependency at all — it kept restoring
from cache after the file changed. Nothing failed loudly; the cache was just
wrong.

These tests assert the OUTCOME (the path is tracked) rather than which
attribute got patched, so they stay meaningful when pathlib is restructured
again.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

from cash.notebook.file_tracker import FileAccessTracker


@pytest.fixture
def data_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n")
    return f


def _tracked(tracker) -> set[str]:
    return {p.replace("\\", "/").lower() for p in tracker.get_accessed_files()}


class TestPathlibReadsAreTracked:
    def test_read_text_registers_a_dependency(self, data_file):
        tracker = FileAccessTracker()
        with tracker:
            pathlib.Path(data_file).read_text()
        assert str(data_file).replace("\\", "/").lower() in _tracked(tracker), (
            "Path.read_text() left no file dependency — a cell reading through "
            "pathlib would never be invalidated when the file changes"
        )

    def test_read_bytes_registers_a_dependency(self, data_file):
        tracker = FileAccessTracker()
        with tracker:
            pathlib.Path(data_file).read_bytes()
        assert str(data_file).replace("\\", "/").lower() in _tracked(tracker)

    def test_explicit_open_registers_a_dependency(self, data_file):
        tracker = FileAccessTracker()
        with tracker:
            with pathlib.Path(data_file).open() as fh:
                fh.read()
        assert str(data_file).replace("\\", "/").lower() in _tracked(tracker)

    def test_builtin_open_still_tracked(self, data_file):
        """Guard: the pathlib patch must not disturb the ordinary path."""
        tracker = FileAccessTracker()
        with tracker:
            with open(data_file) as fh:
                fh.read()
        assert str(data_file).replace("\\", "/").lower() in _tracked(tracker)


class TestPathlibStillWorks:
    """The patch must not change pathlib's behaviour, only observe it."""

    def test_contents_are_unchanged(self, data_file):
        tracker = FileAccessTracker()
        with tracker:
            text = pathlib.Path(data_file).read_text()
        assert text == "a,b\n1,2\n"

    def test_write_and_reread_roundtrip(self, tmp_path):
        target = tmp_path / "out.txt"
        tracker = FileAccessTracker()
        with tracker:
            pathlib.Path(target).write_text("written")
            assert pathlib.Path(target).read_text() == "written"

    def test_missing_file_still_raises(self, tmp_path):
        tracker = FileAccessTracker()
        with tracker:
            with pytest.raises(FileNotFoundError):
                pathlib.Path(tmp_path / "nope.txt").read_text()

    @pytest.mark.skipif(
        not hasattr(pathlib, "_NormalAccessor"),
        reason="accessor indirection only exists on Python 3.10 and earlier",
    )
    def test_accessor_patch_is_a_staticmethod(self):
        """The 3.10 accessor holds a builtin, which does not bind.

        Installing a plain function would make ``acc.open`` a bound method and
        shift every argument by one, so pathlib would pass the accessor where
        it means to pass the path.
        """
        FileAccessTracker()._apply_patches()
        raw = pathlib._NormalAccessor.__dict__.get("open")
        assert isinstance(raw, staticmethod), (
            f"accessor.open must be a staticmethod, got {type(raw).__name__}; "
            "a bound method would swallow the path argument"
        )

    def test_patch_is_idempotent(self, data_file):
        """Repeated trackers must not stack wrappers on the accessor."""
        for _ in range(3):
            tracker = FileAccessTracker()
            with tracker:
                pathlib.Path(data_file).read_text()
            assert str(data_file).replace("\\", "/").lower() in _tracked(tracker)


def test_version_assumption_holds():
    """Document the branch: only 3.10 has the accessor indirection."""
    has_accessor = hasattr(pathlib, "_NormalAccessor")
    if sys.version_info >= (3, 11):
        assert not has_accessor, (
            "pathlib grew an accessor again — _patch_pathlib_accessor's "
            "'3.11+ needs nothing' assumption needs rechecking"
        )
