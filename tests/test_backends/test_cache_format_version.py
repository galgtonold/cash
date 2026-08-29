"""Cache-format versioning for the on-disk FileBackend.

The on-disk layout (one ``*.entry`` file per entry) can change shape
between Cash versions. To avoid silently decoding a stale layout after an
upgrade, FileBackend stamps the cache directory with the format version it
wrote and, on init, **auto-invalidates** any cache whose stamp doesn't match
the version this build expects. This automates the manually-documented
"run ``%cash_repair --full`` after upgrading" step.
"""

from __future__ import annotations

import glob
import os
import pickle

from cash.backends.file_backend import CACHE_FORMAT_VERSION, FileBackend
from cash.backends.entry_format import ENTRY_SUFFIX, pack_entry, read_entry

VERSION_FILENAME = "CACHE_VERSION"


def _entry_files(cache_dir: str) -> list[str]:
    """Every entry file, current layout or legacy.

    v2 is one ``*.entry`` per entry; v1 was a ``*.meta`` / ``*.data`` pair.
    A migration test has to see both, or it would report a wipe that only
    removed the half it knew to look for.
    """
    return [
        f
        for pattern in (f"*{ENTRY_SUFFIX}", "*.meta", "*.data")
        for f in glob.glob(os.path.join(cache_dir, pattern))
    ]


def test_fresh_cache_stamps_current_version(tmp_path):
    """A brand-new cache dir gets the current format stamp, no wipe."""
    backend = FileBackend(str(tmp_path))
    backend.set("k", {"v": 1})
    meta, value = backend.get("k")
    assert value == {"v": 1}

    version_file = tmp_path / VERSION_FILENAME
    assert version_file.exists()
    assert version_file.read_text().strip() == str(CACHE_FORMAT_VERSION)


def test_matching_version_preserves_entries(tmp_path):
    """Reopening a cache written by the same format version keeps the data
    (the restart-and-restore path must not wipe a compatible cache)."""
    b1 = FileBackend(str(tmp_path))
    b1.set("k", {"v": 1})
    b1.get("k")  # force a write + flush of the value
    b1.shutdown()

    # Simulate a kernel restart: a new backend over the same directory.
    b2 = FileBackend(str(tmp_path))
    meta, value = b2.get("k")
    assert value == {"v": 1}, "compatible cache should survive a reopen"


def test_stale_version_marker_wipes_cache(tmp_path):
    """An older format stamp triggers auto-invalidation of existing entries."""
    backend = FileBackend(str(tmp_path))
    backend.set("k", {"v": 1})
    backend.get("k")
    backend.shutdown()
    assert _entry_files(str(tmp_path)), "precondition: entries exist on disk"

    # Pretend the cache was written by an older, incompatible format.
    (tmp_path / VERSION_FILENAME).write_text(str(CACHE_FORMAT_VERSION - 1))

    reopened = FileBackend(str(tmp_path))
    meta, value = reopened.get("k")
    assert value is None, "stale-format entry must be invalidated, not decoded"
    assert not _entry_files(str(tmp_path)), "stale entries should be removed"
    # Marker is refreshed to the current version.
    assert (tmp_path / VERSION_FILENAME).read_text().strip() == str(CACHE_FORMAT_VERSION)


def test_unstamped_legacy_cache_with_entries_is_wiped(tmp_path):
    """A cache from before this feature (entries present, no marker) is
    treated as incompatible and cleared on first open."""
    # Hand-build a legacy-looking entry pair with no version marker.
    key_hash = "deadbeef" * 8
    (tmp_path / f"{key_hash}.meta").write_bytes(pickle.dumps({"key": "legacy"}))
    (tmp_path / f"{key_hash}.data").write_bytes(pickle.dumps("legacy-value"))
    assert not (tmp_path / VERSION_FILENAME).exists()

    backend = FileBackend(str(tmp_path))
    backend._ensure_initialized()

    assert not _entry_files(str(tmp_path)), "unstamped legacy entries should be wiped"
    assert (tmp_path / VERSION_FILENAME).read_text().strip() == str(CACHE_FORMAT_VERSION)


def test_empty_unstamped_dir_is_not_an_error(tmp_path):
    """An empty dir with no marker is just a fresh cache: stamp it, no warning-worthy wipe."""
    backend = FileBackend(str(tmp_path))
    backend._ensure_initialized()
    assert (tmp_path / VERSION_FILENAME).read_text().strip() == str(CACHE_FORMAT_VERSION)


def test_clear_preserves_version_marker(tmp_path):
    """clear() removes entries but leaves the format stamp in place."""
    backend = FileBackend(str(tmp_path))
    backend.set("k", {"v": 1})
    backend.get("k")
    backend.clear()
    assert (tmp_path / VERSION_FILENAME).exists()
    assert (tmp_path / VERSION_FILENAME).read_text().strip() == str(CACHE_FORMAT_VERSION)
