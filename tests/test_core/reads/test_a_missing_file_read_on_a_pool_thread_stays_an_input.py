"""A file a pool thread finds missing stays an input while other threads read.

The open event skips its own stat when resolving the path just found a regular
file there. That answer was kept on the tracker, which a ``ThreadPoolExecutor``'s
workers share: another worker's found file answered for this one's missing file,
its absence went unrecorded, and a cell doing ``try: open(p) except
FileNotFoundError:`` on a pool kept serving its default after the file appeared.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from cash.tracking import file_dep_snapshot
from cash.tracking.file_tracker import FileAccessTracker


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "default"


def test_a_missing_file_read_among_found_ones_is_recorded_absent(tmp_path):
    found = []
    for i in range(100):
        path = tmp_path / f"f{i}.txt"
        path.write_text("x", encoding="utf-8")
        found.append(str(path))
    missing = str(tmp_path / "missing.txt")
    for trial in range(20):
        paths = found[:]
        paths.insert(trial * 5, missing)
        file_dep_snapshot.begin_file_state_epoch()
        try:
            tracker = FileAccessTracker()
            with tracker, ThreadPoolExecutor(8) as pool:
                assert list(pool.map(_read, paths)).count("default") == 1
        finally:
            file_dep_snapshot.end_file_state_epoch()
        assert any(p.endswith("missing.txt") for p in tracker.absent_files), trial
