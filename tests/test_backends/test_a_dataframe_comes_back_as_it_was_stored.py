"""A DataFrame restored from disk is the frame the call returned.

Parquet wrote a RangeIndex out as an int64 column, so a decorated function
returning ``pd.DataFrame({"a": np.arange(n)})`` handed back a plain Index on
the next process's hit: code that checked the index type, or the memory it
used, saw a different frame on a hit than on a miss. Frames Parquet cannot
store as they are (non-string labels, an index frequency, a column pyarrow
cannot convert) are pickled instead.

Parquet also converted what it could convert: list and tuple cells came back
as numpy arrays, dict cells as structs padded with None, UUIDs as bytes, an
object column of ints and None as float64, ``datetime64[s]`` as ``[ms]``,
integer axis names as strings, ``attrs`` through JSON (tuples as lists, int
keys as strings) and ``flags`` reset; a DataFrame subclass came back as a
plain DataFrame (a GeoDataFrame as WKB bytes with no crs). Only frames of
dtypes Parquet provably keeps are stored as Parquet now.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import uuid

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

import cash
from cash.backends import serialization
from cash.backends.serialization import ParquetSerializer

FRAMES = {
    "range index": lambda: pd.DataFrame({"a": np.arange(5)}),
    "stepped range index": lambda: pd.DataFrame({"a": np.arange(5)}, index=pd.RangeIndex(10, 20, 2, name="i")),
    "categorical and tz columns": lambda: pd.DataFrame(
        {
            "c": pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True),
            "t": pd.date_range("2024-01-01", periods=2, tz="Europe/Berlin"),
        }
    ),
    "index with a frequency": lambda: pd.DataFrame({"a": [1, 2, 3]}, index=pd.date_range("2024", periods=3, freq="D")),
    "integer column labels": lambda: pd.DataFrame({0: [1, 2], 1: [3, 4]}),
    "duplicate column labels": lambda: pd.DataFrame([[1, 2]], columns=["a", "a"]),
    "mixed object column": lambda: pd.DataFrame({"o": [1, "a"]}),
    "no columns": lambda: pd.DataFrame(),
    "list cells": lambda: pd.DataFrame({"tags": [["a", "b"], ["c"]]}),
    "tuple cells": lambda: pd.DataFrame({"pair": [(1, 2), (3, 4)]}),
    "dict cells": lambda: pd.DataFrame({"rec": [{"x": 1}, {"y": 2}]}),
    "uuid cells": lambda: pd.DataFrame({"id": [uuid.UUID(int=1), uuid.UUID(int=2)]}),
    "object ints with a None": lambda: pd.DataFrame({"n": [1, None]}, dtype=object),
    "object strings": lambda: pd.DataFrame({"s": pd.array(["a", "b"], dtype=object)}),
    "object string index": lambda: pd.DataFrame({"v": [1, 2]}, index=pd.Index(["p", "q"], dtype=object)),
    "object string column labels": lambda: pd.DataFrame([[1, 2]], columns=pd.Index(["a", "b"], dtype=object)),
    "datetime64[s]": lambda: pd.DataFrame({"t": pd.to_datetime(["2020-01-01"]).astype("datetime64[s]")}),
    "integer index name": lambda: pd.DataFrame({"v": [1, 2]}, index=pd.Index([1, 2], name=0)),
    "attrs": lambda: _with(pd.DataFrame({"v": [1]}), attrs={"shape": (2, 3), 7: "k"}),
    "no duplicate labels flag": lambda: pd.DataFrame({"v": [1]}).set_flags(allows_duplicate_labels=False),
    "subclass": lambda: _with(Table({"a": [1, 2]}), unit="m"),
    "python strings": lambda: pd.DataFrame({"s": pd.array(["a", None], dtype="string[python]")}),
}


class Table(pd.DataFrame):
    """A DataFrame subclass with its own attribute, as GeoDataFrame has ``crs``."""

    _metadata = ["unit"]

    @property
    def _constructor(self):
        return Table


def _with(df, **attributes):
    for name, value in attributes.items():
        setattr(df, name, value)
    return df


def _assert_same_frame(df, back):
    """*back* is *df*: class, dtypes, axes, names, attrs, flags, values and cell types."""
    assert type(back) is type(df)
    pd.testing.assert_frame_equal(df, back, check_index_type=True, check_column_type=True, check_freq=True)
    assert type(back.index) is type(df.index)
    assert back.index.dtype == df.index.dtype and back.columns.dtype == df.columns.dtype
    assert list(back.index.names) == list(df.index.names)
    assert list(back.columns.names) == list(df.columns.names)
    assert back.attrs == df.attrs
    assert {k: type(v) for k, v in back.attrs.items()} == {k: type(v) for k, v in df.attrs.items()}
    assert back.flags.allows_duplicate_labels == df.flags.allows_duplicate_labels
    for name in getattr(df, "_metadata", []):
        assert getattr(back, name, None) == getattr(df, name, None)
    for column in df.columns:
        assert [type(v) for v in back[column]] == [type(v) for v in df[column]], column


@pytest.mark.parametrize("name", FRAMES)
def test_the_serializer_gives_the_frame_back_unchanged(name):
    df = FRAMES[name]()
    s = ParquetSerializer()
    back = s.deserialize(s.serialize(df))
    _assert_same_frame(df, back)


def test_a_plain_frame_is_still_stored_as_parquet():
    """Positive control: the common frame keeps the Parquet format."""
    assert ParquetSerializer().serialize(FRAMES["range index"]()).startswith(b"PAR1")


def test_the_frames_parquet_keeps_are_stored_as_parquet():
    """Positive control for the allowlist: these keep the Parquet format."""
    frames = [
        pd.DataFrame({"f": [1.5, 2.5], "i": [1, 2], "b": [True, False], "s": ["x", None]}),
        pd.DataFrame({"t": pd.date_range("2024-01-01", periods=2, tz="UTC")}, index=["p", "q"]),
        pd.DataFrame({"n": pd.array([1, None], dtype="Int64")}, index=pd.to_datetime(["2024-01-01", "2024-01-05"])),
    ]
    s = ParquetSerializer()
    for df in frames:
        raw = s.serialize(df)
        assert raw.startswith(b"PAR1"), df.dtypes
        _assert_same_frame(df, s.deserialize(raw))


def test_a_wide_frame_is_pickled():
    """Parquet pays per column: a disk hit on 20,000 columns took 2 s where
    the pickle of the frame takes a few hundredths."""
    cap = getattr(serialization, "PARQUET_MAX_COLUMNS", 100)
    narrow = pd.DataFrame(np.ones((2, cap)), columns=[f"c{i}" for i in range(cap)])
    wide = pd.DataFrame(np.ones((2, cap + 1)), columns=[f"c{i}" for i in range(cap + 1)])
    s = ParquetSerializer()
    assert s.serialize(narrow).startswith(b"PAR1")
    raw = s.serialize(wide)
    assert not raw.startswith(b"PAR1")
    _assert_same_frame(wide, s.deserialize(raw))


def test_a_disk_hit_in_a_new_process_keeps_the_range_index(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            import numpy as np, pandas as pd
            import cash

            @cash.cache
            def make(n):
                fd = os.open(os.environ["RUNS"], os.O_WRONLY | os.O_APPEND | os.O_CREAT)
                os.write(fd, b"x")
                os.close(fd)
                return pd.DataFrame({"a": np.arange(n)})

            print(type(make(10).index).__name__)
            """
        ),
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_CACHE_DIR=str(tmp_path / "cache"), RUNS=str(runs), PYTHONDONTWRITEBYTECODE="1")
    # The cash under test, whichever checkout it is in.
    src = os.path.dirname(os.path.dirname(cash.__file__))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    seen = []
    for _ in range(2):
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
        assert p.returncode == 0, p.stderr
        seen.append(p.stdout.strip().splitlines()[-1])
    assert runs.read_bytes() == b"x", "the second process did not hit the disk entry"
    assert seen == ["RangeIndex", "RangeIndex"], seen
