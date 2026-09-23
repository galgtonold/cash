"""What the CLI needs from `FileBackend`, so it never edits the directory itself.

``cash inspect`` lists entries, and ``cash clear --function/--entry/--expired``
deletes some and tells running processes. Done on raw files, the backend's own
bookkeeping -- the byte total the cap is enforced against, the metadata it
keeps for keys it touched -- was left describing entries that were gone.
"""

from __future__ import annotations

import os
import time

from cash.backends.cache_dir import entry_totals
from cash.backends.file_backend import FileBackend


def _filled(tmp_path, *keys):
    backend = FileBackend(str(tmp_path / "c"), flush_interval=0)
    for key in keys:
        backend.set(key, key * 100, {"execution_time": 1.0})
    backend._writes.wait_all()
    return backend


def test_entries_lists_each_entry_by_id_key_and_size(tmp_path):
    backend = _filled(tmp_path, "mod.f:1", "mod.g:2")
    found = {e.key: e for e in backend.entries()}
    assert set(found) == {"mod.f:1", "mod.g:2"}
    entry = found["mod.f:1"]
    path = os.path.join(backend.cache_dir, f"{entry.id}.entry")
    assert entry.size == os.path.getsize(path)
    assert entry.metadata["execution_time"] == 1.0


def test_listing_a_directory_does_not_stamp_or_create_it(tmp_path):
    """Looking at a cache must not change it, or a mistyped path gains a stamp."""
    plain = tmp_path / "not-a-cache"
    plain.mkdir()
    assert FileBackend(str(plain), flush_interval=0).entries() == []
    assert os.listdir(plain) == []
    assert FileBackend(str(tmp_path / "missing"), flush_interval=0).entries() == []
    assert not (tmp_path / "missing").exists()


def test_bump_generation_moves_the_token_a_running_process_watches(tmp_path):
    backend = _filled(tmp_path, "k")
    before = backend.generation_token()
    time.sleep(0.01)
    FileBackend(backend.cache_dir, flush_interval=0).bump_generation()
    assert backend.generation_token() not in (None, before)


def test_bump_generation_leaves_an_unstamped_directory_unstamped(tmp_path):
    FileBackend(str(tmp_path), flush_interval=0).bump_generation()
    assert os.listdir(tmp_path) == []


def test_cleanup_expired_deletes_through_the_backend(tmp_path):
    """The byte total the cap is enforced against drops with the files."""
    backend = FileBackend(str(tmp_path / "c"), flush_interval=0, max_size_bytes=10**9)
    backend.set("old", b"x" * 5000, {"ttl": 1, "created_at": time.time() - 60})
    backend.set("live", b"y" * 10, {})
    backend._writes.wait_all()
    held = backend.evictor.current_bytes
    assert backend.cleanup_expired(lambda m: m.get("ttl") is not None) == 1
    assert [e.key for e in backend.entries()] == ["live"]
    assert backend.evictor.current_bytes < held - 5000
    assert backend.get_metadata("old") is None


def test_entry_totals_counts_only_entry_files(tmp_path):
    backend = _filled(tmp_path, "a", "b")
    (tmp_path / "c" / "notes.txt").write_text("not an entry")
    count, size = entry_totals(backend.cache_dir)
    assert count == 2
    assert size == sum(e.size for e in backend.entries())
    assert entry_totals(str(tmp_path / "missing")) is None
