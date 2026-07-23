"""Regression: a same-size edit to a >8 MiB file OUTSIDE the sampled hash
regions must invalidate the dependency.

``file_content_hash`` hashes files up to ``_HASH_FULL_MAX_BYTES`` in full but
SAMPLES larger files at three fixed head/middle/tail regions. ``file_dep_is_fresh``
used to treat that hash as authoritative whenever the size matched (mtime
ignored, CAS-98) — so a same-size in-place edit outside every sampled region
produced an identical hash and was silently served STALE.

The fix re-instates mtime as a backstop for the SAMPLED regime only: a matching
sampled hash is trusted only when the mtime also matches. A real edit bumps
mtime, so the stale read is caught; full-hashed (<= cap) files keep the
touch-tolerant CAS-98 behavior unchanged.

Run through ``scripts/fails_first.py`` to confirm the outside-sample case fails
without the fix.
"""
from __future__ import annotations

import os

import pytest

from cash.notebook.file_dep_snapshot import (
    _HASH_FULL_MAX_BYTES,
    _HASH_SAMPLE_REGION_BYTES,
    file_dep_is_fresh,
    snapshot_file_deps,
)

pytestmark = pytest.mark.core

_BIG = _HASH_FULL_MAX_BYTES + 1024 * 1024  # ~9 MiB -> sampled regime


def _sampled_offsets(size: int) -> list[tuple[int, int]]:
    half = _HASH_SAMPLE_REGION_BYTES // 2
    return [
        (0, _HASH_SAMPLE_REGION_BYTES),
        (max(0, size // 2 - half), _HASH_SAMPLE_REGION_BYTES),
        (max(0, size - _HASH_SAMPLE_REGION_BYTES), _HASH_SAMPLE_REGION_BYTES),
    ]


def _outside_offset(size: int) -> int:
    """A byte offset that lies in none of the three sampled regions."""
    for off in (256 * 1024, 1024 * 1024, size // 4, size // 3):
        if not any(s <= off < s + ln for s, ln in _sampled_offsets(size)):
            return off
    raise AssertionError("no unsampled offset found")


def _write_big(path, fill=b"a"):
    path.write_bytes(fill * _BIG)


def _edit_byte(path, offset, byte):
    """Overwrite a single byte in place (size unchanged) and push mtime forward
    to model a real edit that happens later than the snapshot."""
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(byte)
    st = os.stat(path)
    os.utime(path, (st.st_atime + 5, st.st_mtime + 5))


def test_outside_sample_same_size_edit_invalidates(tmp_path):
    """THE FOOTGUN: same-size edit outside all sampled regions must be STALE."""
    f = tmp_path / "big.bin"
    _write_big(f)
    snap = snapshot_file_deps({str(f)})[str(f)]
    assert file_dep_is_fresh(str(f), snap) == (True, None)

    size = os.path.getsize(f)
    off = _outside_offset(size)
    _edit_byte(f, off, b"Z")
    assert os.path.getsize(f) == size, "test setup: size must be unchanged"

    fresh, reason = file_dep_is_fresh(str(f), snap)
    assert fresh is False, "same-size edit outside sampled regions served STALE"
    assert reason == "mtime-sampled"


def test_inside_sample_edit_invalidates(tmp_path):
    """Control: an edit inside a sampled region is caught by the hash itself."""
    f = tmp_path / "big.bin"
    _write_big(f)
    snap = snapshot_file_deps({str(f)})[str(f)]
    _edit_byte(f, 10, b"Z")  # offset 10 is inside the head sample
    fresh, reason = file_dep_is_fresh(str(f), snap)
    assert fresh is False
    assert reason == "content"


def test_big_file_unchanged_is_fresh(tmp_path):
    """No over-invalidation: an untouched big file stays FRESH (same mtime)."""
    f = tmp_path / "big.bin"
    _write_big(f)
    snap = snapshot_file_deps({str(f)})[str(f)]
    assert file_dep_is_fresh(str(f), snap) == (True, None)


def test_small_file_touch_still_hits(tmp_path):
    """CAS-98 preserved for full-hashed files: a touch must NOT invalidate."""
    f = tmp_path / "small.txt"
    f.write_text("alpha-payload")  # well under the cap -> full hash
    snap = snapshot_file_deps({str(f)})[str(f)]
    st = os.stat(f)
    os.utime(f, (st.st_atime + 100, st.st_mtime + 100))  # touch: mtime moves
    assert os.stat(f).st_size == st.st_size
    assert file_dep_is_fresh(str(f), snap) == (True, None), (
        "full-hashed touch must remain fresh (CAS-98)"
    )
