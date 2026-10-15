"""A column of Python strings two entries hold is written to disk once.

Numbers go to the chunk store as buffers, stored once by their content; a
column of Python strings (``StringDtype("python")``, pandas 3's ``str``
without pyarrow) was pickled into every entry's stream instead. A loop whose
results share the column with their input (``h = df.assign(...)``) wrote
and pickled it again every iteration. Its strings now pickle into one
buffer of their own, which goes to the chunk store like a column of
numbers; a column the RAM tier keeps is pickled once while it is kept.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.serialization import PickleSerializer
from cash.backends.tiered_backend import TieredBackend
from cash.decorator.arg_hashing import cow_frame_digest

ROWS = 300_000  # pickled strings well above a chunk's minimum


def _frame() -> pd.DataFrame:
    names = [f"team-{i % 997}" for i in range(ROWS)]
    names[5] = None
    return pd.DataFrame({"name": pd.array(names, dtype=pd.StringDtype("python")), "x": np.arange(ROWS, dtype=float)})


def _chunks(tmp_path) -> list:
    folder = tmp_path / "cache" / "chunks"
    return sorted(p for p in folder.iterdir() if p.suffix == ".chunk") if folder.exists() else []


def test_a_string_column_comes_back_as_it_was_stored():
    df = _frame()
    stream, buffers = PickleSerializer().serialize_split(df)
    back = pickle.loads(stream, buffers=buffers)
    pd.testing.assert_frame_equal(back, df)
    assert type(back["name"].array) is pd.arrays.StringArray
    assert back["name"].isna().sum() == 1
    assert len(stream) < 10_000  # the strings went apart from the stream


def test_a_derived_tables_string_column_adds_no_chunk(tmp_path):
    df = _frame()
    backend = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    backend.set("a", df, {"execution_time": 1.0})
    backend._writes.wait_all()
    before = _chunks(tmp_path)
    assert len(before) == 2  # name's strings and x
    derived = df.assign(y=np.ones(ROWS))
    backend.set("b", derived, {"execution_time": 1.0})
    backend._writes.wait_all()
    added = set(_chunks(tmp_path)) - set(before)
    assert len(added) == 1  # y alone; name and x are on disk already
    pd.testing.assert_frame_equal(backend.get("b")[1], derived)


@pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")
def test_a_column_the_ram_tier_keeps_is_pickled_once(tmp_path):
    from cash.backends import serialization

    df = _frame()
    cow_frame_digest(df)  # hashed, as a call's argument is: its columns are known
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "cache"), flush_interval=0)])
    before = serialization._KEPT_ITEMS.pickled
    for k in range(4):
        tiers.set(f"r{k}", df.assign(p=float(k)), {"execution_time": 5.0, "force_persist": True})
    tiers.backends[1]._writes.wait_all()
    assert serialization._KEPT_ITEMS.pickled - before == 1
    for k in range(4):
        got = FileBackend(str(tmp_path / "cache"), flush_interval=0).get(f"r{k}")[1]
        pd.testing.assert_frame_equal(got, df.assign(p=float(k)))
