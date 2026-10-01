"""A pathlib read must register a file dependency on every supported Python.

``Path.open`` reaches the filesystem differently across versions:

    3.11+   Path.open -> io.open(self, ...)
    3.10    Path.open -> Path._accessor.open(...)     — and _NormalAccessor.open
                                                        was bound to the ORIGINAL
                                                        io.open when pathlib was
                                                        first imported

So on 3.10 a patch on ``io.open`` never reached pathlib, and a cell reading a
file through ``Path.read_text()`` recorded no dependency at all — it kept
restoring from cache after the file changed. Nothing failed loudly; the cache
was just wrong. Reads now arrive as the ``open`` audit event, which the C
``io.open`` raises whoever holds a reference to it.

These tests assert the OUTCOME (the path is tracked) rather than how the read
was seen, so they stay meaningful when pathlib is restructured again.
"""

from __future__ import annotations

import pathlib

import pytest

from cash.tracking.file_tracker import FileAccessTracker


@pytest.fixture
def data_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    return f


def _tracked(tracker) -> set[str]:
    return {p.replace("\\", "/").lower() for p in tracker.get_accessed_files()}


class TestPathlibReadsAreTracked:
    def test_read_text_registers_a_dependency(self, data_file):
        tracker = FileAccessTracker()
        with tracker:
            pathlib.Path(data_file).read_text(encoding="utf-8")
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
            with pathlib.Path(data_file).open(encoding="utf-8") as fh:
                fh.read()
        assert str(data_file).replace("\\", "/").lower() in _tracked(tracker)

    def test_builtin_open_still_tracked(self, data_file):
        """Guard: the pathlib patch must not disturb the ordinary path."""
        tracker = FileAccessTracker()
        with tracker:
            with open(data_file, encoding="utf-8") as fh:
                fh.read()
        assert str(data_file).replace("\\", "/").lower() in _tracked(tracker)


class TestPathlibStillWorks:
    """The patch must not change pathlib's behaviour, only observe it."""

    def test_contents_are_unchanged(self, data_file):
        tracker = FileAccessTracker()
        with tracker:
            text = pathlib.Path(data_file).read_text(encoding="utf-8")
        assert text == "a,b\n1,2\n"

    def test_write_and_reread_roundtrip(self, tmp_path):
        target = tmp_path / "out.txt"
        tracker = FileAccessTracker()
        with tracker:
            pathlib.Path(target).write_text("written", encoding="utf-8")
            assert pathlib.Path(target).read_text(encoding="utf-8") == "written"

    def test_missing_file_still_raises(self, tmp_path):
        tracker = FileAccessTracker()
        with tracker:
            with pytest.raises(FileNotFoundError):
                pathlib.Path(tmp_path / "nope.txt").read_text(encoding="utf-8")

    def test_repeated_trackers_each_see_the_read(self, data_file):
        """Repeated trackers each record the read, however it was seen."""
        for _ in range(3):
            tracker = FileAccessTracker()
            with tracker:
                pathlib.Path(data_file).read_text(encoding="utf-8")
            assert str(data_file).replace("\\", "/").lower() in _tracked(tracker)
