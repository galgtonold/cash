"""A polars frame restored from a disk tier is the frame the call returned.

Polars' own pickle writes a frame as one IPC blob inside the pickle stream:
an 80 MB frame took 190 ms to write and 90 ms to read, and the file tier
copied it once more into its blob (FileBackend set/get 222/270 ms, against
60/80 ms for a pandas frame of that size). Frames are pickled as their
Arrow data now, whose buffers go apart from the stream. These pin that the
frame comes back with every dtype polars has (an Enum is no Categorical,
a time zone stays), its sorted flags, and its name.
"""

from __future__ import annotations

import datetime as dt
import pickle
from decimal import Decimal

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("pyarrow")

from polars.testing import assert_frame_equal, assert_series_equal

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.serialization import OUT_OF_BAND_MIN_BYTES, PickleSerializer
from cash.backends.tiered_backend import TieredBackend

N = 3

COLUMNS = {
    "int8": lambda: pl.Series([1, None, -3], dtype=pl.Int8),
    "int16": lambda: pl.Series([1, 2, -3], dtype=pl.Int16),
    "int32": lambda: pl.Series([1, 2, -3], dtype=pl.Int32),
    "int64": lambda: pl.Series([1, 2, -(2**62)], dtype=pl.Int64),
    "int128": lambda: pl.Series([1, 2, -(2**100)], dtype=pl.Int128),
    "uint8": lambda: pl.Series([1, 2, 255], dtype=pl.UInt8),
    "uint16": lambda: pl.Series([1, 2, 3], dtype=pl.UInt16),
    "uint32": lambda: pl.Series([1, 2, 3], dtype=pl.UInt32),
    "uint64": lambda: pl.Series([1, 2, 2**63], dtype=pl.UInt64),
    "float32": lambda: pl.Series([1.5, None, float("nan")], dtype=pl.Float32),
    "float64": lambda: pl.Series([1.5, float("inf"), -0.0], dtype=pl.Float64),
    "bool": lambda: pl.Series([True, None, False], dtype=pl.Boolean),
    "str": lambda: pl.Series(["a", None, "a long string past the inline view of twelve bytes"]),
    "categorical": lambda: pl.Series(["b", "a", None], dtype=pl.Categorical),
    "enum": lambda: pl.Series(["lo", "hi", None], dtype=pl.Enum(["hi", "mid", "lo"])),
    "date": lambda: pl.Series([dt.date(2024, 1, 1), None, dt.date(1900, 2, 28)]),
    "datetime-us-tz": lambda: pl.Series([dt.datetime(2024, 3, 31, 1, 30), None, dt.datetime(2024, 10, 27, 2)]).cast(
        pl.Datetime("us", "Europe/Berlin")
    ),
    "datetime-ns": lambda: pl.Series([dt.datetime(2024, 1, 1), None, dt.datetime(2262, 1, 1)]).cast(pl.Datetime("ns")),
    "datetime-ms-utc": lambda: pl.Series([dt.datetime(2024, 1, 1), None, dt.datetime(1970, 1, 1)]).cast(
        pl.Datetime("ms", "UTC")
    ),
    "time": lambda: pl.Series([dt.time(1, 2, 3), None, dt.time(23, 59, 59, 999999)]),
    "duration-ms": lambda: pl.Series([dt.timedelta(days=1), None, dt.timedelta(milliseconds=-5)]).cast(
        pl.Duration("ms")
    ),
    "duration-ns": lambda: pl.Series([dt.timedelta(1), None, dt.timedelta(0)], dtype=pl.Duration("ns")),
    "decimal": lambda: pl.Series([Decimal("1.25"), None, Decimal("-99999999.99")], dtype=pl.Decimal(10, 2)),
    "list": lambda: pl.Series([[1, 2], None, []], dtype=pl.List(pl.Int64)),
    "list-of-enum": lambda: pl.Series([["x"], ["y", "x"], None], dtype=pl.List(pl.Enum(["y", "x"]))),
    "array": lambda: pl.Series([[1, 2], [3, 4], None], dtype=pl.Array(pl.Int32, 2)),
    "struct": lambda: pl.Series(
        [{"a": 1, "e": "x"}, None, {"a": None, "e": "y"}],
        dtype=pl.Struct({"a": pl.Int64, "e": pl.Enum(["y", "x"])}),
    ),
    "binary": lambda: pl.Series([b"\x00\x01", None, b"x" * 40], dtype=pl.Binary),
    "null": lambda: pl.Series([None, None, None], dtype=pl.Null),
}


def _column(name):
    try:
        return COLUMNS[name]().alias(name)
    except Exception as exc:
        pytest.skip(f"this polars cannot build {name}: {exc}")


def _through_stream(value):
    serializer = PickleSerializer()
    return serializer.deserialize(serializer.serialize(value))


def _through_split(value):
    serializer = PickleSerializer()
    stream, buffers = serializer.serialize_split(value)
    return serializer.deserialize_split(stream, buffers)


def _through_file_tier(value, path):
    writer = FileBackend(str(path), flush_interval=0)
    writer.set("k", value, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    reader = FileBackend(str(path), flush_interval=0)  # a new process
    _meta, back = reader.get("k")
    reader.shutdown()
    return back


@pytest.fixture(params=["stream", "split", "file tier"])
def round_trip(request, tmp_path):
    if request.param == "stream":
        return _through_stream
    if request.param == "split":
        return _through_split
    return lambda value: _through_file_tier(value, tmp_path / "cache")


def _same_frame(back, df):
    assert type(back) is pl.DataFrame
    assert back.schema == df.schema
    assert_frame_equal(back, df, check_dtypes=True, check_exact=True)
    assert _sorted_flags(back.flags.values()) == _sorted_flags(df.flags.values())


def _same_series(back, s):
    assert type(back) is pl.Series
    assert (back.name, back.dtype) == (s.name, s.dtype)
    assert_series_equal(back, s, check_dtypes=True, check_exact=True)
    assert _sorted_flags([back.flags]) == _sorted_flags([s.flags])


def _sorted_flags(flags):
    """The sorted flags, not polars 1's FAST_EXPLODE: a hint it may drop."""
    return [(f["SORTED_ASC"], f["SORTED_DESC"]) for f in flags]


@pytest.mark.parametrize("name", sorted(COLUMNS))
def test_a_series_of_every_dtype_comes_back(name, round_trip):
    s = _column(name)
    _same_series(round_trip(s), s)


@pytest.mark.parametrize("name", sorted(COLUMNS))
def test_a_frame_of_every_dtype_comes_back(name, round_trip):
    df = pl.DataFrame([_column(name), pl.Series("i", range(N))])
    _same_frame(round_trip(df), df)
    empty = df.clear()
    _same_frame(round_trip(empty), empty)


def test_an_enum_comes_back_an_enum_not_a_categorical(round_trip):
    df = pl.DataFrame([_column("enum"), _column("list-of-enum"), _column("struct")])
    back = round_trip(df)
    assert back.schema["enum"] == pl.Enum(["hi", "mid", "lo"])
    assert back.schema["list-of-enum"] == pl.List(pl.Enum(["y", "x"]))
    assert back.schema["struct"] == pl.Struct({"a": pl.Int64, "e": pl.Enum(["y", "x"])})


def test_every_column_at_once_comes_back(round_trip):
    columns = []
    for name in sorted(COLUMNS):
        try:
            columns.append(COLUMNS[name]().alias(name))
        except Exception:
            continue
    df = pl.DataFrame(columns)
    _same_frame(round_trip(df), df)


@pytest.mark.parametrize(
    "value",
    [
        pl.DataFrame(),
        pl.DataFrame({"a": [1, 2]}).select([]),
        pl.DataFrame(schema={"a": pl.Int64, "e": pl.Enum(["x"])}),
        pl.Series("empty", [], dtype=pl.String),
        pl.Series([], dtype=pl.Null),
    ],
    ids=["no columns", "zero width", "no rows", "empty series", "empty null series"],
)
def test_an_empty_frame_comes_back(value, round_trip):
    back = round_trip(value)
    if isinstance(value, pl.Series):
        _same_series(back, value)
    else:
        _same_frame(back, value)


def test_sorted_flags_come_back(round_trip):
    up = pl.Series("up", [1, 2, 3]).set_sorted()
    down = pl.Series("down", ["c", "b", "a"]).set_sorted(descending=True)
    plain = pl.Series("plain", [3, 1, 2])
    _same_series(round_trip(up), up)
    _same_series(round_trip(down), down)
    df = pl.DataFrame([up, down, plain])
    back = round_trip(df)
    _same_frame(back, df)
    assert back.flags["up"]["SORTED_ASC"] and back.flags["down"]["SORTED_DESC"]
    assert not any(back.flags["plain"].values())


def test_a_frame_of_many_chunks_comes_back(round_trip):
    parts = [pl.DataFrame({"x": range(i, i + 1000), "s": [str(i)] * 1000}) for i in range(0, 5000, 1000)]
    df = pl.concat(parts, rechunk=False)
    assert df.n_chunks() > 1
    _same_frame(round_trip(df), df)


def test_a_frame_held_in_a_container_comes_back(round_trip):
    df = pl.DataFrame({"x": [1, 2]})
    value = {"frames": [df, df], "series": df["x"], "n": 1}
    back = round_trip(value)
    _same_frame(back["frames"][0], df)
    assert back["frames"][0] is back["frames"][1]  # one object, as it went in
    _same_series(back["series"], df["x"])


def test_a_large_frame_is_written_apart_from_the_stream():
    df = pl.DataFrame({"x": pl.int_range(1_000_000, eager=True), "f": pl.int_range(1_000_000, eager=True) / 3})
    stream, buffers = PickleSerializer().serialize_split(df)
    assert len(stream) < OUT_OF_BAND_MIN_BYTES
    assert sum(memoryview(b).nbytes for b in buffers) >= df.estimated_size()


def test_a_frame_from_disk_is_the_callers_to_change(tmp_path):
    df = pl.DataFrame({"x": pl.int_range(100_000, eager=True), "s": ["v"] * 100_000})
    back = _through_file_tier(df, tmp_path / "cache")
    back[0, "x"] = -1
    back.insert_column(0, pl.Series("new", [0] * back.height))
    assert back[0, "x"] == -1 and back.columns == ["new", "x", "s"]
    again = FileBackend(str(tmp_path / "cache"), flush_interval=0).get("k")[1]
    _same_frame(again, df)


@pytest.mark.parametrize("as_frame", [True, False], ids=["frame", "series"])
def test_an_object_column_is_refused_with_a_pickling_error(as_frame):
    """Polars refuses one too: polars 1 with a panic, which derives from
    BaseException and so passed every ``except Exception`` of a store."""
    s = pl.Series("o", [[1], {"a": 2}], dtype=pl.Object)
    value = pl.DataFrame({"o": s, "n": [1, 2]}) if as_frame else s
    for serializer in (PickleSerializer().serialize, PickleSerializer().serialize_split):
        with pytest.raises(pickle.PicklingError):
            serializer({"result": value})


@pytest.mark.expects_failed_writes  # the disk tier refuses it
def test_a_frame_with_an_object_column_stays_in_ram(tmp_path):
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "cache"), flush_interval=0)])
    df = pl.DataFrame({"o": pl.Series([[1], [2]], dtype=pl.Object), "n": [1, 2]})
    tiers.set("k", df, {"execution_time": 5.0, "decorator_entry": True})
    meta, back = tiers.get("k")
    tiers.shutdown()
    assert meta["source"] == "RAM"
    assert back["o"].to_list() == [[1], [2]]


def test_an_entry_written_by_polars_own_pickle_still_loads():
    df = pl.DataFrame({"x": [1, 2], "e": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))})
    _same_frame(PickleSerializer().deserialize(pickle.dumps(df, protocol=5)), df)
