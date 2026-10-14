"""A pandas frame with Arrow-backed columns comes back from every tier as it
was stored, and a store of it copies nothing it does not have to.

pandas' own pickle of an Arrow-backed column (pandas 3's default ``str``,
``ArrowDtype``) joins its chunks first (``combine_chunks``): a copy of
every value, 0.8 s for the ten text columns of a 5.6M-row frame, which the
disk write then copied again. Such a column is pickled as its Arrow chunks
now, whose buffers go out of band as they are. These pin that the dtype,
the missing values, a dictionary column and the categories of a
categorical come back the same from RAM, from disk and in a new process,
that a small slice of a big column is written as its rows, and that
nothing done to the caller's frame after the store reaches what is kept.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.serialization import OUT_OF_BAND_MIN_BYTES, PickleSerializer
from cash.backends.tiered_backend import TieredBackend
from tests._scripts import run_python

COLUMNS = {
    "str": lambda: pd.Series(["a", None, "a long string", ""] * 3, dtype="str"),
    "string-pyarrow": lambda: pd.Series(["a", None, "b", "c"] * 3, dtype="string[pyarrow]"),
    "arrow-string": lambda: pd.Series(["x", None, "y", "z"] * 3, dtype=pd.ArrowDtype(pa.string())),
    "arrow-large-string": lambda: pd.Series(["x", "yy", None, "zzz"] * 3, dtype=pd.ArrowDtype(pa.large_string())),
    "arrow-dictionary": lambda: pd.Series(
        pa.array(["lo", "hi", None, "lo"] * 3).dictionary_encode(),
        dtype=pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string())),
    ),
    "arrow-int-nulls": lambda: pd.Series([1, None, -(2**62), 4] * 3, dtype="int64[pyarrow]"),
    "arrow-all-null": lambda: pd.Series([None] * 12, dtype=pd.ArrowDtype(pa.string())),
    "arrow-list": lambda: pd.Series([[1, 2], None, [], [3]] * 3, dtype=pd.ArrowDtype(pa.list_(pa.int64()))),
    "categorical-of-str": lambda: pd.Series(pd.Categorical(["b", "a", None, "b"] * 3, categories=["b", "a", "c"])),
}


def _frame() -> pd.DataFrame:
    df = pd.DataFrame({name: make() for name, make in COLUMNS.items()})
    df["x"] = np.arange(len(df), dtype=float)
    df.index = pd.Index([f"r{i}" for i in range(len(df))], dtype="str", name="row")
    # Two chunks per Arrow column, as a concat of two reads leaves them.
    return pd.concat([df.iloc[:5], df.iloc[5:]])


def _assert_same(restored: pd.DataFrame, stored: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(restored, stored, check_exact=True)
    assert list(map(repr, restored.dtypes)) == list(map(repr, stored.dtypes))
    for name in stored.columns:
        assert type(restored[name].array) is type(stored[name].array), name
        assert restored[name].isna().tolist() == stored[name].isna().tolist(), name
    assert repr(restored.index.dtype) == repr(stored.index.dtype)
    assert list(restored["categorical-of-str"].cat.categories) == ["b", "a", "c"]


def test_the_frame_has_chunked_arrow_columns():
    df = _frame()
    assert df["str"].array._pa_array.num_chunks == 2, "the test checks no chunked column"


def test_a_pickle_of_the_frame_is_the_frame():
    df = _frame()
    _assert_same(PickleSerializer().deserialize(PickleSerializer().serialize(df)), df)


@pytest.mark.parametrize("copy", [True, False])
def test_the_split_pickle_of_the_frame_is_the_frame(copy):
    df = _frame()
    serializer = PickleSerializer()
    stream, buffers = serializer.serialize_split(df, copy=copy)
    _assert_same(serializer.deserialize_split(stream, buffers), df)


def test_a_disk_entry_and_a_ram_hit_are_the_frame(tmp_path):
    df = _frame()
    ram = InMemoryBackend()
    tiers = TieredBackend([ram, FileBackend(str(tmp_path / "c"), flush_interval=0)])
    tiers.set("k", df, {"execution_time": 60.0, "force_persist": True})
    _assert_same(ram.get("k")[1], df)
    tiers.backends[1].shutdown()
    _assert_same(FileBackend(str(tmp_path / "c")).get("k")[1], df)


def test_a_new_process_reads_the_frame_back(tmp_path):
    df = _frame()
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    disk.set("k", df, {"execution_time": 60.0})
    disk.shutdown()
    script = (
        "import pandas as pd\n"
        "from cash.backends.file_backend import FileBackend\n"
        "pd.to_pickle(FileBackend('c').get('k')[1], 'got.pkl')\n"
    )
    run_python("-c", script, cwd=tmp_path)
    _assert_same(pd.read_pickle(tmp_path / "got.pkl"), df)


def test_a_small_slice_of_a_big_column_is_written_as_its_rows():
    big = pd.DataFrame({"s": pd.Series([f"value {i}" for i in range(200_000)], dtype="str")})
    head = big.iloc[1000:1005]
    assert head["s"].array._pa_array.chunk(0).get_total_buffer_size() > 1_000_000, "the slice owns its buffers"
    stream, buffers = PickleSerializer().serialize_split(head)
    written = len(stream) + sum(memoryview(b).nbytes for b in buffers)
    assert written < OUT_OF_BAND_MIN_BYTES, f"{written} bytes written for five rows"
    pd.testing.assert_frame_equal(pickle.loads(stream, buffers=buffers), head)


def test_an_edit_of_the_callers_frame_after_the_store_reaches_nothing_kept(tmp_path):
    df = _frame()
    expected = df.copy(deep=True)
    ram = InMemoryBackend()
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    tiers = TieredBackend([ram, disk])
    tiers.set("k", df, {"execution_time": 60.0, "force_persist": True})
    df.loc["r0", "str"] = "changed"
    df.loc["r1", "arrow-int-nulls"] = 7
    df["x"] *= 2
    values = df["x"].to_numpy()
    if values.flags.writeable:
        values[:] = -1.0
    _assert_same(ram.get("k")[1], expected)
    disk.shutdown()
    _assert_same(FileBackend(str(tmp_path / "c")).get("k")[1], expected)
