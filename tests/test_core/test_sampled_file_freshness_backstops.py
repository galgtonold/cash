"""What a SAMPLED dependency's freshness check can and cannot see.

Two round-16 testers, independently, reproduced the same wrong answer (5/5 and
3/3): a CSV above the full-hash threshold, one field rewritten IN PLACE so the
size is unchanged, and the mtime restored afterwards -- `cp -p`, `rsync -a`,
`tar -x` and any script that saves and restores timestamps all do that -- was
served from cache with the old total. The sample covers head/middle/tail, so it
misses an interior edit; the mtime backstop was then put back by hand.

Three things came out of it:

* the snapshot records ``st_ctime`` for sampled files. On POSIX that is the
  inode CHANGE time: it moves on any write and no ordinary tool restores it, so
  the edit is caught. On Windows it is the CREATION time and does not move --
  measured on this machine, which is why the next two items exist.
* ``file_hash_full_max_bytes`` makes the threshold configurable.
* **the default is 64 MiB**, not the 8 MiB this shipped with, so the ordinary
  CSV or parquet is hashed in full and the hole does not reach it on any
  platform. A full hash costs about 0.72 ms per MiB, but the digest is
  memoized per process, so only the first check of a file pays it.

Above 64 MiB the sampled regime is still the sampled regime, and the timestamp
comparison is what stands in for the bytes the hash never reads. It compares
the integer nanoseconds exactly now, which splits the mtime-restoring family
in two: a tool that puts back whole seconds (``tar``, ``rsync -a``) cannot
reproduce the original nanoseconds and is caught, while one that puts back the
exact nanoseconds (``cp -p``, ``shutil.copystat``) still is not -- on Windows,
where there is no inode change time. Both arms are characterised here, at a
lowered threshold rather than with a 130 MiB fixture.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

from cash import Cash
from cash.notebook import file_dep_snapshot
from cash.notebook.file_dep_snapshot import (
    _HASH_FULL_MAX_BYTES_DEFAULT,
    snapshot_file_deps,
)

# A 9 MiB fixture is BELOW the 64 MiB default (full-hashed) and above this
# lowered one (sampled). Both regimes are then reachable from one file.
_SAMPLE_ABOVE = 1024 * 1024


@pytest.fixture
def sampled_regime(monkeypatch):
    """Put the fixture files into the sampled regime without growing them."""
    monkeypatch.setattr(
        file_dep_snapshot, "_full_hash_max_bytes", lambda: _SAMPLE_ABOVE
    )


def _big_csv(path, mib=9):
    """A file with a known interior field, full-hashed under the default."""
    row = b"1,alpha,10.00\n"
    rows_per_mib = (1024 * 1024) // len(row)
    with open(path, "wb") as fh:
        for _ in range(rows_per_mib * mib):
            fh.write(row)
    size = os.path.getsize(path)
    assert _SAMPLE_ABOVE < size <= _HASH_FULL_MAX_BYTES_DEFAULT
    return path


_EDITS = iter(range(11, 99))


def _edit_in_place_preserving_mtime(path, exact_ns=True):
    """Rewrite one interior row and put the timestamps back.

    *exact_ns* is the difference between the two families of restoring tool,
    and since the timestamp comparison became exact it is also the difference
    between missed and caught:

    * ``True`` puts back the integer nanoseconds -- ``cp -p`` (``utimensat``
      with the source's full precision), ``shutil.copystat``,
      ``robocopy /COPY:T``. Indistinguishable from no edit at all.
    * ``False`` puts back the float seconds -- plain ``tar`` (ustar headers
      carry whole seconds), ``rsync -a`` (whole seconds), and any script that
      round-trips the value through ``st_mtime``. Measured, that lands 200 ns
      off the original, which the exact comparison sees.

    A DIFFERENT value every call: two arms editing the same offset with the
    same bytes leave the file unchanged, and the second arm then "misses" an
    edit that never happened. (It did, in the first draft of this file.)
    """
    before = os.stat(path)
    size = before.st_size
    row = f"9,omega,{next(_EDITS)}.00\n".encode()
    assert len(row) == 14
    with open(path, "r+b") as fh:
        fh.seek(size // 4)
        fh.seek(-(size // 4) % 14, 1)      # land on a row boundary
        fh.write(row)
    if exact_ns:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        os.utime(path, (before.st_atime, before.st_mtime))
    return before


def test_the_default_hashes_an_ordinary_csv_in_full(tmp_path):
    """The knob's value is the fix: no ctime backstop means no sampling."""
    path = _big_csv(str(tmp_path / "big.csv"))
    assert "ctime" not in snapshot_file_deps({path})[path]


def test_a_sampled_file_records_its_ctime(sampled_regime, tmp_path):
    """The extra signal is captured; whether it MOVES is the platform's call."""
    path = _big_csv(str(tmp_path / "big.csv"))
    snap = snapshot_file_deps({path})
    assert "ctime" in snap[path]


def test_a_small_file_does_not_bother(tmp_path):
    """Below the threshold the content hash is authoritative on its own."""
    small = tmp_path / "small.csv"
    small.write_text("1,alpha,10.00\n" * 10, encoding="utf-8")
    snap = snapshot_file_deps({str(small)})
    assert "ctime" not in snap[str(small)]


@pytest.mark.skipif(sys.platform == "win32",
                    reason="st_ctime is the creation time on Windows and does "
                           "not move on an in-place edit -- measured")
def test_the_edit_is_caught_on_posix(sampled_regime, tmp_path):
    """POSIX: the inode change time gives it away even with mtime restored."""
    path = _big_csv(str(tmp_path / "big.csv"))
    snap = snapshot_file_deps({path})
    time.sleep(1.1)
    _edit_in_place_preserving_mtime(path)

    from cash.notebook.file_dep_snapshot import file_dep_is_fresh
    fresh, reason = file_dep_is_fresh(path, snap[path])
    assert not fresh
    assert reason == "ctime-sampled"


def _reader(cache_dir):
    c = Cash(cache_dir=cache_dir, register_magic=False)
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def read_total(p):
        runs.append(1)
        time.sleep(0.25)              # clear the persistence floor
        with open(p, "rb") as fh:
            return sum(float(line.split(b",")[2]) for line in fh if line.strip())

    return read_total, runs


def test_the_default_catches_the_stealth_edit(tmp_path):
    """End to end, on every platform, with nothing configured.

    This is what raising the default bought: the edit both round-16 testers
    reported now invalidates, because a 9 MiB CSV is hashed in full.
    """
    path = _big_csv(str(tmp_path / "big.csv"))
    read_total, runs = _reader(str(tmp_path / "cache"))

    before = read_total(path)
    time.sleep(1.1)
    _edit_in_place_preserving_mtime(path)

    assert read_total(path) != before, "a full hash missed the edit"
    assert len(runs) == 2


def test_a_coarse_timestamp_restore_is_caught_anywhere(sampled_regime, tmp_path):
    """The half of the mtime-restoring family that the exact comparison sees.

    A tool that puts the timestamp back from whole seconds -- ``tar``,
    ``rsync -a``, a script round-tripping ``st_mtime`` -- cannot reproduce the
    original nanoseconds, so the restored value differs and the edit is caught
    on every platform, sampled or not. Before the comparison became exact, a
    10 ms tolerance swallowed the difference (measured: 200 ns).
    """
    path = _big_csv(str(tmp_path / "big.csv"))
    read_total, runs = _reader(str(tmp_path / "cache"))

    first = read_total(path)
    time.sleep(1.1)
    _edit_in_place_preserving_mtime(path, exact_ns=False)

    assert read_total(path) != first, "a coarse timestamp restore hid an edit"
    assert len(runs) == 2


def test_a_nanosecond_exact_restore_is_still_blind_on_windows(sampled_regime,
                                                              tmp_path):
    """The characterisation of what remains, at a lowered threshold.

    A file above ``file_hash_full_max_bytes`` is sampled, and a tool that puts
    back the exact nanoseconds -- ``cp -p``, ``shutil.copystat`` -- leaves
    nothing for the mtime comparison to see. POSIX still catches it through
    the inode change time; Windows has no second timestamp to fall back on.
    If this ever starts recomputing, the hole is closed and this file should
    say so.
    """
    path = _big_csv(str(tmp_path / "big.csv"))
    read_total, runs = _reader(str(tmp_path / "cache"))

    first = read_total(path)
    time.sleep(1.1)
    _edit_in_place_preserving_mtime(path, exact_ns=True)
    second = read_total(path)

    if sys.platform == "win32":
        assert second == first and len(runs) == 1
    else:
        assert second != first and len(runs) == 2   # ctime saves it


def test_an_unedited_big_file_still_hits(tmp_path):
    """The control: none of this may cost an untouched dependency its hit."""
    path = _big_csv(str(tmp_path / "big.csv"))
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def rows(p):
        runs.append(1)
        time.sleep(0.25)
        with open(p, "rb") as fh:
            return sum(1 for _ in fh)

    assert rows(path) == rows(path)
    assert len(runs) == 1
