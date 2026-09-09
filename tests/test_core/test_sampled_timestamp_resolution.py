"""The sampled regime's timestamp backstop compares exactly, not loosely.

Above ``file_hash_full_max_bytes`` the content hash covers three regions of the
file, so an interior edit is caught by the timestamp comparison or not at all.
That comparison carried a **10 ms tolerance**, inherited from the check that
predated content hashing (``cd763a6``), where it absorbed jitter across the
whole decision rather than guarding a backstop.

Measured before this change, on a 65 MiB file with one byte rewritten in place
outside every sampled region:

    edit lands  9.03 ms after the recorded mtime -> served FRESH (stale read)
    edit lands  9.50 ms after                    -> served FRESH
    edit lands 11.03 ms after                    -> caught ('mtime-sampled')

so the window was exactly as wide as the constant. Nothing needed it: the
integer nanoseconds are what CPython reads from the OS (``st_mtime`` is derived
from them, not the reverse), they survive the metadata round-trip bit-exact --
checked by reading a stored entry back off disk -- and 400 one-byte appends
gave 400 distinct timestamps on ext4 and tmpfs, 54 on NTFS, against a 10 ms
tolerance no filesystem needs.

The tolerance stays for snapshots written before the nanoseconds were recorded,
so upgrading does not invalidate every large file at once.

The timings here are set with ``os.utime`` rather than raced for: a test that
has to land an edit inside a 10 ms window is a test that fails on a loaded box.
"""
from __future__ import annotations

import os

import pytest

from cash.notebook import file_dep_snapshot
from cash.notebook.file_dep_snapshot import (
    _LEGACY_TIMESTAMP_TOLERANCE_SECONDS,
    file_dep_is_fresh,
    snapshot_file_deps,
)

pytestmark = pytest.mark.core

_SAMPLE_ABOVE = 1024 * 1024                 # lowered threshold: 1 MiB
_BIG = 2 * 1024 * 1024                      # 2 MiB -> sampled regime


@pytest.fixture(autouse=True)
def sampled_regime(monkeypatch):
    """Sampled, without a 65 MiB fixture: the regime is what is under test."""
    monkeypatch.setattr(
        file_dep_snapshot, "_full_hash_max_bytes", lambda: _SAMPLE_ABOVE
    )


def _big_file(tmp_path, name="big.bin"):
    p = tmp_path / name
    p.write_bytes(b"a" * _BIG)
    return str(p)


def _unsampled_offset(size):
    """An offset in none of the head / middle / tail regions."""
    region = file_dep_snapshot._HASH_SAMPLE_REGION_BYTES
    half = region // 2
    regions = [(0, region), (size // 2 - half, region), (size - region, region)]
    off = size // 4
    assert not any(s <= off < s + n for s, n in regions), "test setup"
    return off


def _edit_in_place(path, mtime_ns):
    """Same size, one byte, outside the sample -- landing at *mtime_ns*."""
    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.seek(_unsampled_offset(size))
        fh.write(b"Z")
    assert os.path.getsize(path) == size, "test setup: size must not move"
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, mtime_ns))


def test_an_edit_inside_the_old_tolerance_is_caught(tmp_path):
    """THE WINDOW: 5 ms of mtime movement used to read as no movement at all."""
    path = _big_file(tmp_path)
    snap = snapshot_file_deps({path})[path]
    assert file_dep_is_fresh(path, snap) == (True, None)

    slack_ns = int(_LEGACY_TIMESTAMP_TOLERANCE_SECONDS * 1e9) // 2      # 5 ms
    _edit_in_place(path, snap["mtime_ns"] + slack_ns)

    fresh, reason = file_dep_is_fresh(path, snap)
    assert fresh is False, (
        "a same-size edit that moved the mtime by less than the old tolerance "
        "was served from cache"
    )
    assert reason == "mtime-sampled"


def test_one_nanosecond_is_enough(tmp_path):
    """The floor is now the filesystem's, not a constant's.

    Skipped where the filesystem cannot represent it -- FAT32 and friends
    round to whole seconds, and there the tolerance was never the limit.
    """
    path = _big_file(tmp_path)
    snap = snapshot_file_deps({path})[path]
    target = snap["mtime_ns"] + 1
    _edit_in_place(path, target)
    if os.stat(path).st_mtime_ns != target:
        pytest.skip("this filesystem cannot store nanosecond timestamps")

    assert file_dep_is_fresh(path, snap)[0] is False


def test_an_untouched_file_still_hits(tmp_path):
    """The control that matters: exactness must not cost a real hit.

    Stat the file repeatedly -- an unstable ``st_mtime_ns`` between calls would
    turn every check into a recompute, which is the failure mode a tolerance
    exists to prevent.
    """
    path = _big_file(tmp_path)
    snap = snapshot_file_deps({path})[path]
    for _ in range(5):
        file_dep_snapshot._HASH_MEMO.clear()
        assert file_dep_is_fresh(path, snap) == (True, None)


def test_a_legacy_snapshot_keeps_the_tolerance(tmp_path):
    """An entry written before the nanoseconds were recorded still works.

    Without this, upgrading would invalidate every sampled file at once: the
    stored float cannot reproduce the live nanoseconds exactly.
    """
    path = _big_file(tmp_path)
    snap = snapshot_file_deps({path})[path]
    legacy = {k: v for k, v in snap.items() if not k.endswith("_ns")}
    assert "mtime_ns" not in legacy

    assert file_dep_is_fresh(path, legacy) == (True, None)

    # and it still catches what it always caught
    _edit_in_place(path, os.stat(path).st_mtime_ns + 5_000_000_000)
    assert file_dep_is_fresh(path, legacy)[0] is False


def test_a_full_hashed_file_ignores_timestamps_entirely(tmp_path):
    """Below the threshold none of this applies -- content decides, as before.

    The touch-tolerance (CAS-98) is the reason the exact comparison is scoped
    to the sampled regime rather than applied everywhere.
    """
    small = tmp_path / "small.csv"
    small.write_text("a,b\n1,2\n", encoding="utf-8")
    snap = snapshot_file_deps({str(small)})[str(small)]

    st = os.stat(small)
    os.utime(small, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    assert file_dep_is_fresh(str(small), snap) == (True, None), (
        "a touch on a full-hashed file must remain a hit"
    )
