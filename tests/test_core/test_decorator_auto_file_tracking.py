"""Auto-tracked file reads must invalidate the decorator cache.

If a cached function reads a file via pandas/numpy/joblib/builtins.open,
the file path is recorded as an implicit dependency. A subsequent
modification of that file invalidates the cache, so the next call
re-computes.

The notebook path has had this behaviour from the start
(``FileAccessTracker`` is opened around every statement). The decorator
path was missing it — the validation report at
``docs/superpowers/scratch/decorator_validation/report.html`` surfaced
it across every backend.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from cash import Cash
from cash.backends.memory_backend import InMemoryBackend


def _touch_future(path: Path) -> None:
    """Bump mtime well past now so any cached-mtime check misses."""
    future = time.time() + 60
    os.utime(path, (future, future))


class TestAutoFileTracking:
    def test_open_invalidates_on_change(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("v1", encoding="utf-8")
        calls = [0]

        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache
        def f():
            time.sleep(0.12)
            calls[0] += 1
            with open(path) as fh:
                return fh.read()

        assert f() == "v1"
        assert f() == "v1"
        assert calls[0] == 1

        path.write_text("v2", encoding="utf-8")
        _touch_future(path)

        assert f() == "v2"
        assert calls[0] == 2, f"expected invalidation; calls={calls[0]}"

    def test_pandas_read_csv_invalidates_on_change(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        calls = [0]

        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache
        def load():
            time.sleep(0.12)
            calls[0] += 1
            return pd.read_csv(path)

        df1 = load()
        df2 = load()
        assert calls[0] == 1
        assert df1.equals(df2)

        # Mutate the file
        path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        _touch_future(path)

        df3 = load()
        assert calls[0] == 2, f"expected invalidation; calls={calls[0]}"
        assert len(df3) == 3

    def test_unchanged_file_still_hits(self, tmp_path):
        """Reading a file that hasn't changed → cache hit."""
        path = tmp_path / "data.txt"
        path.write_text("static", encoding="utf-8")
        calls = [0]

        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache
        def f():
            time.sleep(0.12)
            calls[0] += 1
            with open(path) as fh:
                return fh.read()

        assert f() == "static"
        assert f() == "static"
        assert f() == "static"
        assert calls[0] == 1, "file unchanged — must hit cache every subsequent call"

    def test_explicit_file_depends_on_still_works(self, tmp_path):
        """The explicit ``file_depends_on`` API stays functional alongside
        auto-tracking — they don't conflict."""
        path = tmp_path / "data.txt"
        path.write_text("v1", encoding="utf-8")
        calls = [0]

        c = Cash(backend=InMemoryBackend(), register_magic=False)

        @c.cache(file_depends_on=str(path))
        def f():
            time.sleep(0.12)
            calls[0] += 1
            return path.read_text()

        f(); f()
        assert calls[0] == 1
        path.write_text("v2", encoding="utf-8")
        _touch_future(path)
        assert f() == "v2"
        assert calls[0] == 2


# --------------------------------------------------------------------------- #
# Kernel pseudo-filesystems must never become cache dependencies              #
# --------------------------------------------------------------------------- #


def test_pseudo_fs_reads_are_not_tracked_as_dependencies():
    """A ``/proc`` read inside a cached call must not become a file dep.

    ``/proc/meminfo`` reports live memory, so its content changes on every
    read. An entry that records it as a dependency is found on lookup and then
    thrown away as stale, every single time — the cache silently never hits,
    with no warning.

    cash reaches it through its OWN machinery, not the user's:
    ``InMemoryBackend._check_and_evict`` calls ``psutil.virtual_memory()``
    every ``check_interval`` writes and psutil reads ``/proc/meminfo`` on
    Linux, while the tracker is active for the user's call.
    """
    from cash.notebook.file_tracker import FileAccessTracker, _is_pseudo_fs

    assert _is_pseudo_fs("/proc/meminfo")
    assert _is_pseudo_fs("/sys/fs/cgroup/memory.max")
    assert _is_pseudo_fs("/dev/urandom")
    assert not _is_pseudo_fs("/home/me/data.csv")
    assert not _is_pseudo_fs("/tmp/procession.csv")  # prefix, not a path component

    tracker = FileAccessTracker({})
    tracker._track_path("/proc/meminfo")
    tracker._track_path("/sys/kernel/mm/transparent_hugepage/enabled")
    assert tracker.accessed_files == set(), (
        f"pseudo-fs paths leaked into deps: {tracker.accessed_files}"
    )


def test_a_chunked_iterator_still_hits_after_an_eviction_check(tmp_path):
    """End-to-end guard for the shape that exposed this.

    A chunked iterator writes one entry per chunk plus a manifest, so a
    10-chunk result crosses ``InMemoryBackend``'s eviction-check threshold
    (``check_interval``, 10 writes) *inside a single cached call* — which is
    what pulled ``/proc/meminfo`` into the entry and made the second identical
    call recompute. Verified failing before the fix on Linux (``hits=0
    misses=2``); it is the second call hitting that matters here.
    """
    import cash

    c = cash.Cash(cache_dir=str(tmp_path / "c"), register_magic=False)

    @c.cache(chunk_max_items=10)
    def stream():
        yield from range(100)

    list(stream())
    before = stream.cache_info()["hits"]
    list(stream())
    after = stream.cache_info()["hits"]

    assert after == before + 1, (
        "the second identical call did not hit; a dependency captured during "
        f"the first call is defeating the cache. cache_info={stream.cache_info()}"
    )
