"""A ``frozen=True`` result changed while its disk write runs, and changed
back before the write is checked, is stored as it was returned.

Written from its own memory, the chunk digest was taken first and the pack
bytes later, so a change in between landed on disk under the digest of the
returned content: read back wrong, and reused for a later entry holding
that content. Its buffers are copied before the store returns instead.
"""

from __future__ import annotations

import numpy as np

from cash.backends import memory_backend
from cash.backends.chunk_store import ChunkStore
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.content_hashers import builtin_hash

ROWS = 1 << 18


def _tiers(folder):
    return TieredBackend([InMemoryBackend(), FileBackend(str(folder), flush_interval=0)])


def test_a_change_made_and_undone_during_the_pack_write_reaches_neither_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_backend, "DISK_ALONE_MIN_BYTES", 1 << 20)
    value = np.arange(ROWS, dtype=np.float64)
    real_pack = ChunkStore._write_pack

    def racing_pack(self, fresh):
        # A user thread's ``arr[i] = x; ...; arr[i] = old`` across the write.
        value[7] = 12345.0
        try:
            return real_pack(self, fresh)
        finally:
            value[7] = 7.0

    monkeypatch.setattr(ChunkStore, "_write_pack", racing_pack)
    tiers = _tiers(tmp_path / "c")
    tiers.set("frozen", value, {"execution_time": 50.0, "content_digest": builtin_hash(value), "func_name": "f"})
    tiers.shutdown()
    monkeypatch.setattr(ChunkStore, "_write_pack", real_pack)

    again = _tiers(tmp_path / "c")
    _meta, back = again.get("frozen")
    assert back is not None and back[7] == 7.0
    # An ordinary entry with the same content shares its chunk.
    again.set("other", np.arange(ROWS, dtype=np.float64), {"execution_time": 50.0, "func_name": "g"})
    again.shutdown()
    last = _tiers(tmp_path / "c")
    _meta, other = last.get("other")
    last.shutdown()
    np.testing.assert_array_equal(other, np.arange(ROWS, dtype=np.float64))
