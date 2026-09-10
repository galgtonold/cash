"""A file that changes while the call runs does not become a stale entry.

CAS-109, round-17 tester r17s1 (#3, #4). The entry's file fingerprints were
taken when it was STORED -- after the body finished -- so a file rewritten
after the body read it was fingerprinted in its new state. The entry matched
the new file and held the old result, and every later call was a hit with the
old answer:

    read the file  ->  a sync job patches it  ->  the call returns and stores
    next call: HIT, the pre-patch answer, for as long as the entry lives

The nested form was worse: an outer cached aggregate re-fingerprinted a file
its inner call had already read, and ``explain()`` then contradicted itself --
outer HIT, inner MISS file_changed. The documented mitigation (write via a temp
file and rename) did not help: the rename lands before the store.

Now each file's stat is taken when it is first read and compared before the
store; if it moved, the result is returned but not cached, with
STORE-INPUT-CHANGED.

The concurrent writer is a thread sequenced with Events, not a sleep race: the
body reads, signals, and waits until the write has landed. Deterministic on a
loaded box.
"""
from __future__ import annotations

import threading
import time
import warnings

import pytest

from cash import Cash

pytestmark = pytest.mark.core


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _write(path, text):
    path.write_text(text, encoding="utf-8")


class _Writer:
    """Rewrites *path* once the body says it has read it."""

    def __init__(self, path, text):
        self.path, self.text = path, text
        self.read_done = threading.Event()
        self.written = threading.Event()
        self.armed = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.read_done.wait(10)
        _write(self.path, self.text)
        self.written.set()

    def after_read(self):
        """Called by the body: let the writer land, then carry on."""
        if self.armed:
            self.armed = False
            self.read_done.set()
            self.written.wait(10)


def _stored_warnings(rec):
    return [w for w in rec if "STORE-INPUT-CHANGED" in str(w.message)]


def test_a_file_rewritten_during_the_call_is_not_cached(c, tmp_path):
    """THE BUG: the patch was fingerprinted as if it were what was read."""
    data = tmp_path / "data.txt"
    _write(data, "1\n2\n3\n")
    runs = []
    writer = _Writer(data, "1\n2\n9\n")           # sync job lands mid-call

    @c.cache(assume_safe=True)
    def total(path):
        runs.append(1)
        with open(path, encoding="utf-8") as fh:
            value = sum(int(x) for x in fh.read().split())
        writer.after_read()
        time.sleep(0.2)                           # over the persistence floor
        return value

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        first = total(str(data))

    assert first == 6, "the body computed from what it read"
    assert _stored_warnings(rec), "nothing said the result was not cached"

    second = total(str(data))
    assert second == 12, "the pre-patch answer was served after the patch"
    assert len(runs) == 2


def test_the_nested_form_is_not_cached_either(c, tmp_path):
    """An outer aggregate must not re-fingerprint a file its inner call read."""
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    _write(a, "1\n")
    _write(b, "2\n")
    writer = _Writer(a, "5\n")                    # lands while inner(b) runs
    outer_runs = []

    @c.cache(assume_safe=True)
    def inner(path):
        with open(path, encoding="utf-8") as fh:
            value = int(fh.read())
        if path.endswith("b.txt"):
            writer.after_read()
        time.sleep(0.2)
        return value

    @c.cache(assume_safe=True)
    def outer(folder):
        outer_runs.append(1)
        return inner(str(tmp_path / "a.txt")) + inner(str(tmp_path / "b.txt"))

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        first = outer(str(tmp_path))

    assert first == 3
    assert _stored_warnings(rec)

    second = outer(str(tmp_path))
    assert second == 7, "the outer entry served the pre-change aggregate"
    assert len(outer_runs) == 2


def test_an_unchanged_file_still_stores_and_hits(c, tmp_path):
    """The control that matters: the check must not cost ordinary caching."""
    data = tmp_path / "data.txt"
    _write(data, "1\n2\n3\n")
    runs = []

    @c.cache(assume_safe=True)
    def total(path):
        runs.append(1)
        with open(path, encoding="utf-8") as fh:
            value = sum(int(x) for x in fh.read().split())
        time.sleep(0.2)
        return value

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert total(str(data)) == 6
        assert total(str(data)) == 6

    assert len(runs) == 1
    assert not _stored_warnings(rec)


def test_writing_into_a_listed_directory_still_caches(c, tmp_path):
    """Directories are left out on purpose.

    A function that lists a folder and writes its output into it moves the
    folder's mtime itself; refusing to cache every such function would be a
    regression for a smaller hole than the one this closes.
    """
    import glob
    import os

    folder = tmp_path / "data"
    folder.mkdir()
    _write(folder / "x.csv", "1\n")
    runs = []

    @c.cache(assume_safe=True)
    def summarise(path):
        runs.append(1)
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(path, "*.csv")))
        time.sleep(0.2)
        (folder / "summary.parquet").write_bytes(b"x")    # output into the same folder
        return names

    summarise(str(folder))
    summarise(str(folder))
    assert len(runs) == 1, "listing a folder and writing into it stopped caching"
