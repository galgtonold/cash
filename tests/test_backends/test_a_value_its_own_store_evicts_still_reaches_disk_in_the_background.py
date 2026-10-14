"""A value the RAM tier evicts as it stores it is written without a wait.

p18's ``chg`` (a 1.1 GB frame, ``# @cash:persist``) was stored in the RAM
tier, and the byte cap evicted that very entry at once: it was worth the
least per byte. The disk write then found no RAM copy to take, so the
frame was pickled before the store returned (1.2-6.4 s). And a look at the
entry's metadata a moment later (the file snapshots its producer recorded)
waited for the disk write to finish (1.6-3.9 s).

The RAM tier's copy of an entry its own store evicted is still cash's own:
the disk write takes it and serializes it in the background, as when the
entry stayed. A look at the metadata of an entry whose write is still
queued is answered from that write, without waiting for it.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from cash.backends import FileBackend, InMemoryBackend, TieredBackend

MiB = 1 << 20


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(2 * MiB, dtype=float), "s": ["a"] * (2 * MiB)})  # 34 MiB in RAM


def test_the_disk_write_serializes_the_evicted_ram_copy_in_the_background(tmp_path, monkeypatch):
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    ram = InMemoryBackend(max_size_bytes=48 * MiB)  # the frame fits, the four do not
    tiers = TieredBackend([ram, disk])
    for i in range(3):  # worth more per byte than the frame: they stay
        tiers.set(f"small{i}", np.ones(MiB), {"execution_time": 60.0})  # 8 MiB each
    serialized_on: list[str] = []
    serialize = disk._serialize

    def recording(value, serializer, *, copy):
        serialized_on.append(threading.current_thread().name)
        return serialize(value, serializer, copy=copy)

    monkeypatch.setattr(disk, "_serialize", recording)
    frame = _frame()
    metadata = {"execution_time": 0.01, "force_persist": True}
    tiers.set("frame", frame, metadata)
    assert ram.peek_metadata("frame") is None, "the frame stayed in RAM: the test checks nothing"
    assert "DISK" in metadata["storage"]
    frame.loc[0, "x"] = -1.0  # after the store: never on disk
    disk.shutdown()
    assert threading.current_thread().name not in serialized_on, "the frame was serialized before the store returned"
    pd.testing.assert_frame_equal(FileBackend(str(tmp_path / "c")).get("frame")[1], _frame())


def test_a_look_at_a_queued_entry_does_not_wait_for_its_write(tmp_path, monkeypatch):
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    release = threading.Event()
    write = disk._do_set_sync

    def held_write(*args):
        release.wait(30)
        return write(*args)

    monkeypatch.setattr(disk, "_do_set_sync", held_write)
    disk.set("k", [1, 2, 3], {"execution_time": 1.0, "file_dependencies": {"data.csv": {"size": 3}}})
    looked: list = []
    look = threading.Thread(target=lambda: looked.append(disk.peek_metadata("k")))
    look.start()
    look.join(10)
    try:
        assert not look.is_alive(), "the look waited for the write"
        assert looked[0]["file_dependencies"] == {"data.csv": {"size": 3}}
    finally:
        release.set()
        look.join(30)
    assert disk.get("k")[1] == [1, 2, 3]
    assert disk.peek_metadata("k")["file_dependencies"] == {"data.csv": {"size": 3}}
    disk.shutdown()
