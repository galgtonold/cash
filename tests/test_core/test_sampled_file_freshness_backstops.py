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
* **the default is 256 MiB** (64 MiB until round 19), not the 8 MiB this
  shipped with, so the ordinary CSV, parquet or .npy is hashed in full and the hole does not reach it on any
  platform. A full hash costs about 0.72 ms per MiB, but the digest is
  memoized per process, so only the first check of a file pays it.

Above 256 MiB the sampled regime is still the sampled regime, and the timestamp
comparison is what stands in for the bytes the hash never reads. It compares
the integer nanoseconds exactly now, and what that catches depends on the
resolution the restoring tool stores: whole seconds (a plain ``tar`` ustar
header, rsync's protocol) cannot reproduce them and is caught; the exact
nanoseconds (``cp -p``, ``shutil.copystat``, GNU tar's pax headers) still are
not -- on Windows, where there is no inode change time. Both arms are
characterised here, at a lowered threshold rather than with a 130 MiB fixture.
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

# A 9 MiB fixture is BELOW the 256 MiB default (full-hashed) and above this
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


def _edit_in_place_preserving_mtime(path, restore="exact"):
    """Rewrite one interior row and put the timestamp back.

    How precisely the timestamp comes back is what decides whether the exact
    comparison sees the edit, so the restoring tool's own resolution matters:

    * ``"exact"`` -- the integer nanoseconds, which is what ``cp -p`` passes to
      ``utimensat``, and what ``shutil.copystat`` and ``robocopy /COPY:T`` do.
      Indistinguishable from no edit at all.
    * ``"seconds"`` -- whole seconds, the resolution a plain ``tar`` ustar
      header and rsync's protocol carry. The sub-second part is dropped, so
      the restored value always differs and the edit is always caught.

    A third case is deliberately NOT asserted anywhere: a float round-trip
    (``os.utime(p, (st.st_atime, st.st_mtime))``, what a Python script that
    reads and writes ``st_mtime`` does). Whether the double reproduces the
    original nanoseconds is a property of the value, not of the code --
    measured over 300 live NTFS timestamps, it survived 46% of the time and
    landed 100-200 ns off the rest. An arm built on that is a coin flip, and
    it flipped: this test passed alone and failed after its neighbour, because
    the two files' mtimes happened to differ in representability.

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
    if restore == "exact":
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif restore == "seconds":
        os.utime(path, ns=(before.st_atime_ns,
                           (before.st_mtime_ns // 1_000_000_000) * 1_000_000_000))
    else:
        raise AssertionError(f"unknown restore mode {restore!r}")
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


def test_a_whole_second_restore_is_caught_anywhere(sampled_regime, tmp_path):
    """The half of the mtime-restoring family that the exact comparison sees.

    A tool whose format carries only whole seconds -- a plain ``tar`` ustar
    header, rsync's protocol -- drops the sub-second part, so the restored
    value always differs from the recorded one and the edit is caught on every
    platform. Under the old 10 ms tolerance the difference had to exceed 10 ms
    to count; a restore that lands in the same second never did.

    (GNU tar's pax extended headers do carry nanoseconds, and that shape
    belongs with ``cp -p`` in the test below rather than here.)
    """
    path = _big_csv(str(tmp_path / "big.csv"))
    read_total, runs = _reader(str(tmp_path / "cache"))

    first = read_total(path)
    time.sleep(1.1)
    _edit_in_place_preserving_mtime(path, restore="seconds")

    assert read_total(path) != first, "a whole-second timestamp restore hid an edit"
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
    _edit_in_place_preserving_mtime(path, restore="exact")
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


def test_the_default_full_hash_threshold_covers_a_memmapped_npy():
    """Round 19: an 80 MiB .npy written through ``np.memmap(mode="r+")`` on
    Windows kept its size, mtime and NTFS change time -- only its content
    moved, and above the 64 MiB threshold content was sampled, so the reader
    was served stale 3 of 3. Decided: raise the default to 256 MiB. The config
    field and the snapshot's fallback must agree."""
    from cash.config import CashConfig
    assert _HASH_FULL_MAX_BYTES_DEFAULT == 256 * 1024 * 1024
    assert CashConfig().file_hash_full_max_bytes == _HASH_FULL_MAX_BYTES_DEFAULT
