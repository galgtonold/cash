"""A DataFrame restored from disk is the frame the call returned.

Frames were once stored as Parquet, which changed them: a RangeIndex came
back as a plain int64 Index, list and tuple cells as numpy arrays, dict cells
as structs padded with None, UUIDs as bytes, an object column of ints and
None as float64, ``datetime64[s]`` as ``[ms]``, integer axis names as
strings, ``attrs`` through JSON (tuples as lists, int keys as strings) and
``flags`` reset; a DataFrame subclass came back as a plain DataFrame (a
GeoDataFrame as WKB bytes with no crs). Frames are pickled now, and these
pin that each of them comes back as it went in.
"""

from __future__ import annotations

import textwrap
import uuid

import pytest

from tests._scripts import run_python

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.backends.serialization import PickleSerializer

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
    s = PickleSerializer()
    back = s.deserialize(s.serialize(df))
    _assert_same_frame(df, back)


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
    seen = []
    for _ in range(2):
        p = run_python(script, cwd=tmp_path, cache_dir=tmp_path / "cache", env={"RUNS": str(runs)})
        seen.append(p.stdout.strip().splitlines()[-1])
    assert runs.read_bytes() == b"x", "the second process did not hit the disk entry"
    assert seen == ["RangeIndex", "RangeIndex"], seen
