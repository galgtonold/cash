"""A reader given a string too long to be a file name reads no file.

``pd.read_html(page)`` and ``pd.read_json(text)`` take the string itself as
content. Treating a long HTML page as a path pattern sent it through
``glob.glob``, which recursed once per ``/`` and raised RecursionError, so the
user's call failed under cash with an error plain Python never shows.
"""

from __future__ import annotations

from cash.tracking import reader_patches
from cash.tracking.reader_patches import track_dataset


class _Recorder:
    def __init__(self):
        self.paths = []

    def track_path(self, path):
        self.paths.append(path)


def test_a_long_html_string_is_not_tracked_as_a_path():
    page = "<table>" + "<tr><td>a/b/c</td><td>[1]</td></tr>" * 400 + "</table>"
    tracker = _Recorder()
    track_dataset(tracker, page)
    assert tracker.paths == []


def test_a_deep_pattern_within_the_limit_is_recorded_as_named(monkeypatch):
    monkeypatch.setattr(reader_patches, "_PATH_MAX", 10**6)
    pattern = "[a]/" * 3000 + "x"
    tracker = _Recorder()
    track_dataset(tracker, pattern)
    assert tracker.paths == [pattern]


def test_a_short_path_is_still_tracked(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a\n1\n")
    tracker = _Recorder()
    track_dataset(tracker, str(f))
    assert tracker.paths == [str(f)]
