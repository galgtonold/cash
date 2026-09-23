"""A lookup over many files in one directory lists it instead of stat-ing each.

On Windows a stat opens the file, ~90 us, and re-running statements derived
from 3,000 files made one per file per lookup (round 23). A directory listing
reports every entry's size and times at once. It is taken only for files
hashed in full, where content -- not the listing -- decides freshness.
"""

from __future__ import annotations

import os
import types

import pytest

from cash.notebook import file_dep_snapshot
from cash.notebook.file_dep_snapshot import snapshot_file_deps, stats_from_listings
from cash.notebook.statement.freshness import CacheFreshnessChecker

N = 40
windows_only = pytest.mark.skipif(os.name != "nt", reason="listings replace stats on Windows only")


@pytest.fixture(autouse=True)
def _no_digest_reuse(monkeypatch):
    """Every check hashes afresh: what is under test is the stat, not the memo."""
    monkeypatch.setattr(file_dep_snapshot, "_HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_HASH_MEMO_TTL_SECONDS", 0.0)
    file_dep_snapshot._HASH_MEMO.clear()


def _inputs(tmp_path, n=N):
    paths = []
    for i in range(n):
        p = tmp_path / f"part{i:03d}.csv"
        p.write_text("g,v\n" + "\n".join(f"{i},{j}" for j in range(20)) + "\n")
        st = os.stat(p)
        os.utime(p, (st.st_atime - 3600, st.st_mtime - 3600))
        paths.append(str(p))
    return paths


def _check(paths):
    checker = CacheFreshnessChecker(backend=None)
    checker._checked, checker._listed = {}, {}
    metadata = types.SimpleNamespace(file_dependencies=snapshot_file_deps(set(paths)))
    return checker, metadata


@pytest.fixture
def per_file_stats(monkeypatch, tmp_path):
    counted: list[str] = []
    real = os.stat

    def counting(path, *a, **k):
        if str(tmp_path) in os.fspath(path):
            counted.append(os.fspath(path))
        return real(path, *a, **k)

    monkeypatch.setattr(os, "stat", counting)
    return counted


@windows_only
def test_a_crowded_directory_is_listed_not_stat_ed(tmp_path, per_file_stats):
    paths = _inputs(tmp_path)
    checker, metadata = _check(paths)
    per_file_stats.clear()
    assert checker._invalidate_if_direct_file_changed(metadata, "payload") == "payload"
    assert per_file_stats == [], f"{len(per_file_stats)} per-file stats beside the listing"


@pytest.mark.xfail(
    os.name == "nt",
    strict=True,
    reason="Windows: an edit that keeps the size and puts the mtime back is not seen once the file had settled -- a documented limitation (known-limitations: an edit that keeps size and timestamps); Linux and macOS catch it through the inode change time",
)
def test_an_edit_that_keeps_size_and_time_is_still_caught(tmp_path):
    paths = _inputs(tmp_path)
    checker, metadata = _check(paths)
    target = paths[23]
    before = os.stat(target)
    with open(target, "r+b") as fh:
        fh.seek(5)
        fh.write(b"Z")
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert checker._invalidate_if_direct_file_changed(metadata, "payload") is None
    assert "part023" in (checker.last_miss_reason or "")


def test_a_size_change_is_caught(tmp_path):
    paths = _inputs(tmp_path)
    checker, metadata = _check(paths)
    with open(paths[7], "a") as fh:
        fh.write("9,9\n")
    assert checker._invalidate_if_direct_file_changed(metadata, "payload") is None


@windows_only
def test_a_sampled_file_gets_a_stat_of_its_own(tmp_path, monkeypatch, per_file_stats):
    """Above the full-hash cap the timestamps back up a sampled hash, and a
    listing's timestamps are the one thing that can lag."""
    monkeypatch.setattr(file_dep_snapshot, "_full_hash_max_bytes", lambda: 16)
    import cash.notebook.statement.freshness as freshness

    monkeypatch.setattr(freshness, "_full_hash_max_bytes", lambda: 16)
    paths = _inputs(tmp_path)
    checker, metadata = _check(paths)
    per_file_stats.clear()
    assert checker._invalidate_if_direct_file_changed(metadata, "payload") == "payload"
    assert len(set(per_file_stats)) == N


def test_cash_lists_through_the_unpatched_scandir(tmp_path, monkeypatch):
    """The file tracker records a directory listed while it is active; a
    listing cash takes for itself must not become the user's dependency."""
    paths = _inputs(tmp_path, n=20)
    real = os.scandir

    def tracked(*_a, **_k):
        raise AssertionError("listed through the tracked os.scandir")

    tracked._original_func = real
    monkeypatch.setattr(os, "scandir", tracked)
    listed = stats_from_listings(paths)
    if os.name == "nt":
        assert set(listed) == set(paths)
        assert all(listed[p].st_size == os.stat(p).st_size for p in paths)
    else:
        assert listed == {}


def test_a_sparse_directory_is_not_listed(tmp_path):
    """A lookup needing a handful of files from a big directory stats them."""
    assert stats_from_listings(_inputs(tmp_path, n=3)) == {}
