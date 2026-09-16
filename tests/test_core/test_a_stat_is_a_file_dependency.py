"""What ``Path.stat()`` reports about a file makes that file a dependency.

The notebook arm, and the round-24 story, is
``test_notebook_integration/test_a_file_read_by_its_size_is_a_dependency.py``.
The tracker is shared, so a cached function sees it the same way.
"""
from __future__ import annotations

import os
from pathlib import Path

from cash import Cash


def _settle(path: Path) -> None:
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def test_a_size_read_through_stat_recomputes_after_the_file_changes(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    target = tmp_path / "a.csv"
    target.write_bytes(b"x\n1\n")
    _settle(target)
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def size_of(p):
        runs.append(1)
        return Path(p).stat().st_size

    assert size_of(str(target)) == 4
    assert size_of(str(target)) == 4
    assert len(runs) == 1

    target.write_bytes(b"x\n1\n2\n3\n")
    assert size_of(str(target)) == 8
    assert len(runs) == 2


def test_a_directory_stat_is_not_a_file_dependency(tmp_path):
    """A directory has no content to hash; recording one would leave the entry
    unable ever to prove itself fresh."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    folder = tmp_path / "d"
    folder.mkdir()
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def is_folder(p):
        runs.append(1)
        return Path(p).stat().st_mode != 0

    assert is_folder(str(folder)) is True
    assert is_folder(str(folder)) is True
    assert len(runs) == 1
