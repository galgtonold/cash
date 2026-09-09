"""What a >8 MiB dependency's freshness check can and cannot see.

Two round-16 testers, independently, reproduced the same wrong answer (5/5 and
3/3): a CSV above the full-hash threshold, one field rewritten IN PLACE so the
size is unchanged, and the mtime restored afterwards -- `cp -p`, `rsync -a`,
`tar -x` and any script that saves and restores timestamps all do that -- was
served from cache with the old total. The sample covers head/middle/tail, so it
misses an interior edit; the mtime backstop was then put back by hand.

Two things change here, and neither is a claim that the hole is closed on every
platform:

* the snapshot records ``st_ctime`` for sampled files. On POSIX that is the
  inode CHANGE time: it moves on any write and no ordinary tool restores it, so
  the edit is caught. On Windows it is the CREATION time and does not move --
  measured on this machine, which is why the next item exists.
* ``file_hash_full_max_bytes`` makes the threshold configurable, so a user can
  buy immunity with time: a full hash costs about 0.72 ms per MiB on every
  freshness check, i.e. on every hit that depends on the file (23 ms at 32 MiB,
  46 ms at 64 MiB, measured).

The default is unchanged. Raising it is a cost decision that belongs to whoever
runs the pipeline, and the docs now state the price.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

from cash import Cash
from cash.notebook.file_dep_snapshot import (
    _HASH_FULL_MAX_BYTES_DEFAULT,
    snapshot_file_deps,
)


def _big_csv(path, mib=9):
    """A file above the sampling threshold, with a known interior field."""
    row = b"1,alpha,10.00\n"
    rows_per_mib = (1024 * 1024) // len(row)
    with open(path, "wb") as fh:
        for _ in range(rows_per_mib * mib):
            fh.write(row)
    assert os.path.getsize(path) > _HASH_FULL_MAX_BYTES_DEFAULT
    return path


_EDITS = iter(range(11, 99))


def _edit_in_place_preserving_mtime(path):
    """Rewrite one interior row and put the timestamps back.

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
    os.utime(path, (before.st_atime, before.st_mtime))
    return before


def test_a_sampled_file_records_its_ctime(tmp_path):
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
def test_the_edit_is_caught_on_posix(tmp_path):
    """POSIX: the inode change time gives it away even with mtime restored."""
    path = _big_csv(str(tmp_path / "big.csv"))
    snap = snapshot_file_deps({path})
    time.sleep(1.1)
    _edit_in_place_preserving_mtime(path)

    from cash.notebook.file_dep_snapshot import file_dep_is_fresh
    fresh, reason = file_dep_is_fresh(path, snap[path])
    assert not fresh
    assert reason == "ctime-sampled"


def test_raising_the_threshold_catches_it_anywhere(tmp_path, monkeypatch):
    """The knob, end to end through a cached function.

    Both arms in one test on purpose: the default arm is the characterisation
    of what is still possible, and the raised arm is what a user can do about
    it. If the default ever starts catching this, the first assertion fails and
    somebody re-reads this file -- which is the point.
    """
    import cash

    path = _big_csv(str(tmp_path / "big.csv"))

    def total(cache_dir):
        c = Cash(cache_dir=cache_dir, register_magic=False)
        runs: list[int] = []

        @c.cache(assume_safe=True)
        def read_total(p):
            runs.append(1)
            time.sleep(0.25)          # clear the persistence floor
            with open(p, "rb") as fh:
                return sum(float(line.split(b",")[2]) for line in fh if line.strip())

        return read_total, runs

    # --- default threshold: the sampled regime, mtime restored -> not caught
    read_total, runs = total(str(tmp_path / "cache_default"))
    first = read_total(path)
    time.sleep(1.1)
    _edit_in_place_preserving_mtime(path)
    second = read_total(path)
    if sys.platform == "win32":
        assert second == first and len(runs) == 1, (
            "Windows has no second timestamp to fall back on; if this now "
            "recomputes, the hole is closed and this file should say so"
        )

    # --- threshold raised above the file: caught everywhere
    monkeypatch.setattr(cash.config.get_config(), "file_hash_full_max_bytes",
                        64 * 1024 * 1024, raising=False)
    cash.configure(file_hash_full_max_bytes=64 * 1024 * 1024)
    try:
        read_total2, runs2 = total(str(tmp_path / "cache_full"))
        before = read_total2(path)
        time.sleep(1.1)
        _edit_in_place_preserving_mtime(path)
        after = read_total2(path)
        assert after != before, "a full hash still missed the edit"
        assert len(runs2) == 2
    finally:
        cash.configure(file_hash_full_max_bytes=_HASH_FULL_MAX_BYTES_DEFAULT)


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
