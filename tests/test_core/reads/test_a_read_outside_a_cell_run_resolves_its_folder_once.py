"""A file read outside a cell run (a cached function called from a script)
resolved its path in full each time: an ``lstat`` of every component on
POSIX, final-path lookups on Windows, then a stat of the file again for its
digest and for whether it was there. The block now resolves each folder
once while it runs, and the ``lstat`` that shows the file is a regular file
answers the rest, as within a cell run. A folder link re-pointed between two
blocks is resolved anew.
"""

from __future__ import annotations

import os

import pytest

from cash._paths import normalize_path
from cash.tracking import file_dep_snapshot
from cash.tracking.file_tracker import FileAccessTracker


def _read_all(folder, names):
    for name in names:
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            fh.read()


def test_reads_in_one_folder_resolve_it_once(tmp_path, monkeypatch):
    assert file_dep_snapshot.HASH_EPOCH is None  # not in a cell run
    folder = tmp_path / "a" / "b" / "c"
    folder.mkdir(parents=True)
    names = [f"part{i}.txt" for i in range(20)]
    for name in names:
        (folder / name).write_text(name, encoding="utf-8")
    asked = []
    real_stat, real_lstat = os.stat, os.lstat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: asked.append(os.fspath(p)) or real_stat(p, *a, **k))
    monkeypatch.setattr(os, "lstat", lambda p, *a, **k: asked.append(os.fspath(p)) or real_lstat(p, *a, **k))
    tracker = FileAccessTracker(hash_on_read=True)
    with tracker:
        _read_all(str(folder), names)
    monkeypatch.undo()
    root = os.path.normcase(str(tmp_path))
    mine = [p for p in asked if os.path.normcase(os.path.abspath(p)).startswith(root)]
    # One look at each file, and the folders above it once for all twenty.
    assert len(mine) <= len(names) + 5, len(mine)
    for name in names:
        resolved = normalize_path(os.path.realpath(folder / name))
        assert tracker.read_digests[resolved] == file_dep_snapshot.file_content_hash(resolved)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_a_folder_link_re_pointed_between_two_blocks_is_resolved_anew(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for folder in (first, second):
        folder.mkdir()
        (folder / "data.txt").write_text(folder.name, encoding="utf-8")
    link = tmp_path / "current"
    try:
        os.symlink(first, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow a symlink here")
    seen = []
    for target in (first, second):
        if target is second:
            link.unlink()
            os.symlink(second, link, target_is_directory=True)
        tracker = FileAccessTracker()
        with tracker:
            _read_all(str(link), ["data.txt"])
        seen.append(set(tracker.accessed_files))
    assert normalize_path(os.path.realpath(first / "data.txt")) in seen[0]
    assert normalize_path(os.path.realpath(second / "data.txt")) in seen[1]


@pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason="a POSIX open climbs from a link's target")
def test_a_read_through_a_folder_link_and_dotdot_is_recorded_where_the_open_goes(tmp_path):
    target = tmp_path / "real" / "inner"
    target.mkdir(parents=True)
    (tmp_path / "real" / "data.txt").write_text("real", encoding="utf-8")
    (tmp_path / "data.txt").write_text("decoy", encoding="utf-8")
    os.symlink(target, tmp_path / "link", target_is_directory=True)
    tracker = FileAccessTracker()
    with tracker:
        with open(os.path.join(str(tmp_path), "link", "..", "data.txt"), encoding="utf-8") as fh:
            assert fh.read() == "real"
    assert normalize_path(os.path.realpath(tmp_path / "real" / "data.txt")) in tracker.accessed_files
