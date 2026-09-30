"""What ``os.lstat`` or an ``os.scandir`` entry's ``stat()`` reports about a
file makes that file a dependency, as ``os.stat`` and ``Path.stat`` do.

``os.lstat('a.txt').st_size`` and ``[e.stat().st_size for e in
os.scandir('sub')]`` were served with the old sizes after the files were
rewritten in place: neither call was watched, and an in-place rewrite does
not change the directory's listing.
"""

from __future__ import annotations

import os
from pathlib import Path

from cash import Cash


def _settle(path: Path) -> None:
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def _cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def test_a_size_read_through_lstat_recomputes_after_the_file_changes(tmp_path):
    c = _cash(tmp_path)
    target = tmp_path / "a.txt"
    target.write_bytes(b"1")
    _settle(target)

    @c.cache(assume_safe=True)
    def size_of(p):
        return os.lstat(p).st_size

    assert size_of(str(target)) == 1
    assert size_of(str(target)) == 1
    target.write_bytes(b"333")
    assert size_of(str(target)) == 3


def test_a_size_read_through_a_scandir_entry_recomputes_after_the_file_changes(tmp_path):
    c = _cash(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    target = sub / "a.txt"
    target.write_bytes(b"1")
    _settle(target)
    _settle(sub)

    @c.cache(assume_safe=True)
    def sizes(d):
        with os.scandir(d) as it:
            return sorted((e.name, e.stat().st_size) for e in it if e.is_file())

    assert sizes(str(sub)) == [("a.txt", 1)]
    assert sizes(str(sub)) == [("a.txt", 1)]
    target.write_bytes(b"333")
    assert sizes(str(sub)) == [("a.txt", 3)]


def test_a_scandir_entry_still_behaves_as_one(tmp_path):
    """Control: the entry the user gets answers like an ``os.DirEntry``."""
    c = _cash(tmp_path)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    (tmp_path / "d").mkdir()

    @c.cache(assume_safe=True)
    def listing(d):
        return sorted((e.name, e.is_dir(), os.fspath(e) == e.path, Path(e).name) for e in os.scandir(d))

    names = [row for row in listing(str(tmp_path)) if row[0] in ("f.txt", "d")]
    assert names == [("d", True, True, "d"), ("f.txt", False, True, "f.txt")]
