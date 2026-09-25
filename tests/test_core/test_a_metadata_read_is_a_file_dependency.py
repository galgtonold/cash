"""What ``os.stat``, ``getsize``, ``getmtime`` and ``getctime`` report about a
file makes that file a dependency, as ``Path.stat`` already did.

Found while stress-testing the decorator: ``os.path.getsize(p)`` and the
"newest export" idiom ``max(glob.glob("dd/*.csv"), key=os.path.getmtime)``
recorded nothing but the directory listing, and an in-place rewrite does not
move a directory's mtime -- so after one export was rewritten, the old size
and the old "newest" file were served. ``Path(p).stat()`` beside them
recomputed.
"""

from __future__ import annotations

import glob
import os
import shutil
import warnings

import pytest

from cash import Cash


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _write(path, text, age_hours):
    path.write_text(text, encoding="utf-8")
    t = os.stat(path).st_mtime - age_hours * 3600
    os.utime(path, (t, t))


READERS = {
    "os.path.getsize": lambda p: os.path.getsize(p),
    "os.stat": lambda p: os.stat(p).st_size,
    "os.path.getmtime": lambda p: os.path.getmtime(p),
    "os.path.getctime": lambda p: os.path.getctime(p),
}


@pytest.mark.parametrize("name", sorted(READERS))
def test_a_rewrite_recomputes(c, tmp_path, name):
    read = READERS[name]
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def size(p):
        runs.append(1)
        return read(p)

    target = tmp_path / "a.csv"
    _write(target, "1", 2)
    first = size(str(target))
    assert size(str(target)) == first
    assert len(runs) == 1
    _write(target, "333", 1)
    second = size(str(target))
    assert len(runs) == 2
    # Windows' ctime is the creation time, which a rewrite keeps.
    if not (name == "os.path.getctime" and os.name == "nt"):
        assert second != first


def test_the_newest_file_is_found_again_after_an_edit(c, tmp_path):
    dd = tmp_path / "dd"
    dd.mkdir()
    _write(dd / "a.csv", "1", 2)
    _write(dd / "b.csv", "2", 1)

    @c.cache(assume_safe=True)
    def newest(d):
        p = max(glob.glob(os.path.join(d, "*.csv")), key=os.path.getmtime)
        with open(p, encoding="utf-8") as fh:
            return os.path.basename(p) + ":" + fh.read()

    assert newest(str(dd)) == "b.csv:2"
    _write(dd / "a.csv", "333", 0)
    assert newest(str(dd)) == "a.csv:333"


def test_a_file_the_call_copies_over_is_not_an_input(c, tmp_path):
    """``shutil.copy`` stats the destination it overwrites; that stat is the
    library's, not the user's, so the call still caches."""
    src, dst = tmp_path / "src.txt", tmp_path / "dst.txt"
    _write(src, "x", 1)
    _write(dst, "old", 1)
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def publish():
        runs.append(1)
        shutil.copy(src, dst)
        return os.path.getsize(src)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        publish()
    publish()
    assert len(runs) == 1
