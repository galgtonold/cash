"""A file dependency's content is hashed in full, whatever its size.

Above a size threshold a file was hashed from three regions (head, middle,
tail) and its timestamps stood in for the rest. A same-size edit between the
regions, with the modification time put back (``cp -p``, ``shutil.copystat``,
``rsync -a``) and no inode change time to give it away (Windows, where
``st_ctime`` is the creation time), was served from the cache. And a bare
``touch`` of such a file recomputed, while the same touch of a smaller file
did not.

The threshold is set small here, so a 4 MiB file stands in for a big one:
``raising=False`` because the setting it patches is what this change removed.
"""

from __future__ import annotations

import os

import pytest

from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import file_dep_is_fresh, snapshot_file_deps

_SIZE = 4 * 1024 * 1024
#: Between the head (first 256 KiB) and middle (2 MiB +- 128 KiB) regions.
_BETWEEN_REGIONS = 1_000_000


@pytest.fixture
def big_file(tmp_path, monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "full_hash_max_bytes", lambda: 1024, raising=False)
    path = tmp_path / "big.bin"
    path.write_bytes(bytes(_SIZE))
    return str(path)


def _as_windows_sees_it(path, stored):
    """*stored* with the inode change time the file has now: on Windows that
    field is the creation time, which an in-place write does not move."""
    return {**stored, "ctime_ns": os.stat(path).st_ctime_ns}


def test_a_same_size_edit_with_its_mtime_put_back_is_seen(big_file):
    stored = snapshot_file_deps({big_file})[big_file]
    before = os.stat(big_file)
    with open(big_file, "r+b") as fh:
        fh.seek(_BETWEEN_REGIONS)
        fh.write(b"x")
    os.utime(big_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert os.path.getsize(big_file) == _SIZE

    assert file_dep_is_fresh(big_file, _as_windows_sees_it(big_file, stored)) == (False, "content")


def test_a_touched_big_file_is_still_fresh(big_file):
    """The content decides: a new timestamp alone is no change, at any size."""
    stored = snapshot_file_deps({big_file})[big_file]
    st = os.stat(big_file)
    os.utime(big_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    assert file_dep_is_fresh(big_file, stored) == (True, None)


def test_an_unchanged_big_file_is_fresh(big_file):
    """Control for the two above."""
    stored = snapshot_file_deps({big_file})[big_file]
    assert file_dep_is_fresh(big_file, _as_windows_sees_it(big_file, stored)) == (True, None)
