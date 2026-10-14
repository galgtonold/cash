"""A value the RAM tier keeps no copy of its own is still pickled in the background.

The disk write serializes the RAM tier's own copy on the writer thread
(``FileBackend.set(..., private=True)``). A value that tier refused -- over
its eviction target, or slower to restore than to compute -- left no copy,
so the disk write pickled the caller's value before the store returned: for
p28's 900k-row frame of three text columns, 0.2 s of every pass of a
15-pass loop, and seconds for a frame of a few hundred MB.

Such a value is copied in full on the caller's thread instead (arrays and
labels, at about the cost of their memory), and that copy is pickled,
checksummed and written in the background. A table the RAM tier copies
block by block (Python strings without pyarrow, nullable columns) is copied
in full, labels too, and that copy goes to the writer. Nothing the caller
does to the value after the store reaches the entry, and a value that cannot
be copied that way (a table of lists) is still serialized before the store
returns.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from cash.backends import FileBackend, InMemoryBackend, TieredBackend

MiB = 1 << 20
ROWS = MiB // 2


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(ROWS, dtype=float),
            "s": pd.Series([f"t{i % 977}" for i in range(ROWS)], dtype="str"),
        },
        index=pd.Index(np.arange(ROWS) * 2, name="i"),
    )


def _tiers(tmp_path, monkeypatch, ram=None):
    """RAM capped below the frame, so it refuses it (unless *ram* is
    given); and which threads serialize what the disk writes."""
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    ram = ram or InMemoryBackend(max_size_bytes=4 * MiB)
    serialized_on: list[str] = []
    serialize = disk._serialize

    def recording(value, serializer, *, copy):
        serialized_on.append(threading.current_thread().name)
        return serialize(value, serializer, copy=copy)

    monkeypatch.setattr(disk, "_serialize", recording)
    return TieredBackend([ram, disk]), ram, disk, serialized_on


def test_a_refused_frame_is_serialized_by_the_writer(tmp_path, monkeypatch):
    tiers, ram, disk, serialized_on = _tiers(tmp_path, monkeypatch)
    frame = _frame()
    metadata = {"execution_time": 5.0, "force_persist": True}
    tiers.set("frame", frame, metadata)
    assert ram.peek_metadata("frame") is None, "the RAM tier kept the frame: the test checks nothing"
    assert "DISK" in metadata["storage"]
    frame.loc[0, "x"] = -1.0  # after the store: never on disk
    frame.loc[0, "s"] = "changed"
    disk.shutdown()
    assert serialized_on, "nothing was written"
    assert threading.current_thread().name not in serialized_on, "the frame was serialized before the store returned"
    pd.testing.assert_frame_equal(FileBackend(str(tmp_path / "c")).get("frame")[1], _frame(), check_exact=True)


def test_a_refused_tuple_of_a_frame_and_a_dict_is_serialized_by_the_writer(tmp_path, monkeypatch):
    """``h, ratings = run_elo(...)``: a call's result, as a loop stores it."""
    tiers, _ram, disk, serialized_on = _tiers(tmp_path, monkeypatch)
    frame, ratings = _frame(), {"a": 1500.0, "b": 1480.5}
    tiers.set("result", (frame, ratings), {"execution_time": 5.0, "force_persist": True})
    frame["x"] *= 2
    ratings["a"] = 0.0
    disk.shutdown()
    assert serialized_on and threading.current_thread().name not in serialized_on
    stored_frame, stored_ratings = FileBackend(str(tmp_path / "c")).get("result")[1]
    pd.testing.assert_frame_equal(stored_frame, _frame(), check_exact=True)
    assert stored_ratings == {"a": 1500.0, "b": 1480.5}


def test_a_refused_frame_of_lists_is_serialized_before_the_store_returns(tmp_path, monkeypatch):
    tiers, _ram, disk, serialized_on = _tiers(tmp_path, monkeypatch)
    frame = pd.DataFrame({"tags": [[i] for i in range(ROWS // 4)], "x": np.arange(ROWS // 4, dtype=float)})
    tiers.set("lists", frame, {"execution_time": 5.0, "force_persist": True})
    frame["tags"].iloc[0].append("changed")
    disk.shutdown()
    assert serialized_on == [threading.current_thread().name]
    assert FileBackend(str(tmp_path / "c")).get("lists")[1]["tags"].iloc[0] == [0]


def _python_text_frame() -> pd.DataFrame:
    """p28's ``h`` without pyarrow: text in Python strings, and a nullable column."""
    return pd.DataFrame(
        {
            "home": pd.Series([f"L{i % 24:02d}T{i % 20:02d}" for i in range(ROWS)], dtype="string[python]"),
            "goals": pd.Series(np.arange(ROWS) % 5, dtype="Int64"),
            "p": np.linspace(0, 1, ROWS),
        },
        index=pd.Index([f"m{i}" for i in range(ROWS)], dtype="string[python]"),
    )


def test_a_table_the_ram_tier_copied_with_pandas_is_serialized_by_the_writer(tmp_path, monkeypatch):
    tiers, ram, disk, serialized_on = _tiers(tmp_path, monkeypatch, ram=InMemoryBackend())
    frame = _python_text_frame()
    assert type(frame["home"].array).__name__ == "StringArray"
    tiers.set("h", frame, {"execution_time": 5.0, "force_persist": True})
    assert ram.holds_own_copy("h"), "the RAM tier's copy keeps something of the caller's"
    # After the store, through handles that write in place.
    frame["home"].array[0] = "changed"
    frame["goals"].array[1] = 99
    frame.index.array[2] = "relabelled"
    frame.columns.array[2] = "q"
    disk.shutdown()
    assert serialized_on and threading.current_thread().name not in serialized_on
    pd.testing.assert_frame_equal(FileBackend(str(tmp_path / "c")).get("h")[1], _python_text_frame(), check_exact=True)
