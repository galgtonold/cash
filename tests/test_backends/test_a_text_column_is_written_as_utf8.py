"""A column of Python strings is written as its text in UTF-8, exactly.

pandas 3 keeps text without pyarrow as a numpy array of ``str`` objects
(``StringDtype("python")``), and pickle wrote each string as an object of
its own: a loop's results pickled at about 11 MB/s. A disk entry now holds
such a column as its strings joined into one UTF-8 buffer (each object a
column holds many times once, with a code per row), and reads back the
column it was: its dtype (storage and missing value), each missing value
of the kind it was, every string. A string UTF-8 cannot hold (a lone
surrogate) or holding the joining ``"\\0"`` leaves the column to pickle as
before, and an entry written the old way still loads.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from cash import kept_state
from cash.backends import serialization
from cash.backends.file_backend import FileBackend
from cash.backends.serialization import PickleSerializer, stored_reducers

ROWS = 20_000  # over `serialization.STRINGS_APART_MIN_ITEMS`
TEAMS = [f"Team {i} é😀" for i in range(40)]


def _python(values, na_value=pd.NA) -> pd.arrays.StringArray:
    return pd.array(values, dtype=pd.StringDtype("python", na_value=na_value))


def _repeated(na_value=pd.NA) -> pd.arrays.StringArray:
    """Each team one object held by many rows, as a CSV reader leaves it."""
    objects = np.array(TEAMS, dtype=object)
    rows = objects[np.arange(ROWS) % len(TEAMS)]
    rows[7] = None
    return _python(rows, na_value)


def _distinct(na_value=np.nan) -> pd.arrays.StringArray:
    rows = [f"row {i} ✓" if i % 9 else "" for i in range(ROWS)]
    rows[3] = None
    return _python(rows, na_value)


def _items(array) -> list:
    return np.asarray(array._ndarray if hasattr(array, "_ndarray") else array, dtype=object).ravel(order="K").tolist()


def _same_items(back, expected) -> bool:
    """Every item the same value of the same type; NaN as NaN."""
    left, right = _items(back), _items(expected)
    return len(left) == len(right) and all(
        type(a) is type(b) and (a is b or a == b or (a != a and b != b)) for a, b in zip(left, right)
    )


def _write_and_read(tmp_path, value):
    """*value* written to a disk entry and read back by a fresh backend,
    with every byte the cache folder then holds."""
    folder = tmp_path / "cache"
    backend = FileBackend(str(folder), flush_interval=0)
    backend.set("k", value, {"execution_time": 1.0})
    backend._writes.wait_all()
    back = FileBackend(str(folder), flush_interval=0).get("k")[1]
    on_disk = b"".join(p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file())
    return back, on_disk


CASES = {
    "repeated, NA": lambda: _repeated(pd.NA),
    "repeated, nan": lambda: _repeated(np.nan),
    "distinct, nan": lambda: _distinct(np.nan),
    "distinct, NA": lambda: _distinct(pd.NA),
    "all missing": lambda: _python([None] * ROWS),
    "sliced": lambda: _distinct()[1::2],
}


@pytest.mark.parametrize("make", CASES.values(), ids=CASES.keys())
def test_a_text_column_comes_back_exactly_and_is_written_as_utf8(tmp_path, make):
    column = make()
    frame = pd.DataFrame({"name": column, "x": np.arange(len(column), dtype=float)})
    # The column alone too: a table's construction may copy a slice.
    back, on_disk = _write_and_read(tmp_path, {"frame": frame, "column": column})
    pd.testing.assert_frame_equal(back["frame"], frame)
    for got in (back["frame"]["name"].array, back["column"]):
        assert type(got) is type(column)
        assert repr(got.dtype) == repr(column.dtype)
        assert _same_items(got, column)
    strings = [item for item in _items(column) if type(item) is str][:3]
    if len(strings) == 3 and len(set(strings)) == 3:
        # The strings side by side in UTF-8, not each pickled on its own.
        assert "\x00".join(strings).encode() in on_disk


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_a_short_or_empty_column_comes_back_exactly(monkeypatch, rows):
    monkeypatch.setattr(serialization, "STRINGS_APART_MIN_ITEMS", 0)
    column = _python(["", "😀", None, "a", "é"][:rows])
    stream, buffers = PickleSerializer().serialize_split(pd.Series(column))
    back = pickle.loads(stream, buffers=buffers)
    assert repr(back.dtype) == repr(column.dtype)
    assert _same_items(back.array, column)


@pytest.mark.parametrize("odd", ["a\ud800b", "a\x00b"], ids=["lone surrogate", "nul"])
def test_a_string_utf8_cannot_join_leaves_the_column_pickled_as_it_was(tmp_path, odd):
    rows = [f"r{i}" for i in range(ROWS)]
    rows[11] = odd
    column = _python(rows)
    back, _on_disk = _write_and_read(tmp_path, pd.Series(column))
    assert back.array[11] == odd
    assert _same_items(back.array, column)


def test_an_object_column_of_text_and_missing_values_comes_back_exactly(tmp_path):
    rows = np.array([f"r{i % 50} ü" for i in range(2 * ROWS)], dtype=object).reshape(2, ROWS, order="F")
    rows[0, 3], rows[1, 4], rows[1, 5] = None, np.nan, pd.NA
    back, on_disk = _write_and_read(tmp_path, rows)
    assert back.shape == rows.shape and back.dtype == object
    assert back.flags.f_contiguous == rows.flags.f_contiguous
    assert _same_items(back, rows)
    assert back[0, 3] is None and back[1, 4] != back[1, 4] and back[1, 5] is pd.NA
    assert "\x00".join(["r0 ü", "r1 ü", "r2 ü"]).encode() in on_disk


def test_an_object_column_of_other_things_is_pickled_as_numpy_does(tmp_path):
    rows = np.array([f"r{i}" for i in range(ROWS)], dtype=object)
    rows[9] = ("a", 1)  # neither text nor missing
    back, _on_disk = _write_and_read(tmp_path, rows)
    assert _same_items(back, rows)


@pytest.mark.parametrize(("make", "share"), [(_repeated, 0.6), (_distinct, 0.95)], ids=["repeated", "distinct"])
def test_a_text_column_takes_fewer_bytes_than_pandas_pickles(make, share):
    frame = pd.DataFrame({"name": make(np.nan)})
    stream, buffers = PickleSerializer().serialize_split(frame)
    written = len(stream) + sum(memoryview(b).nbytes for b in buffers)
    assert written < share * len(pickle.dumps(frame, protocol=5))


def test_an_entry_written_with_its_strings_pickled_still_loads():
    frame = pd.DataFrame({"name": _repeated(np.nan), "x": np.arange(ROWS, dtype=float)})
    old = kept_state.dumps(frame, protocol=5, extra=stored_reducers())  # the way entries were written
    back = PickleSerializer().deserialize(old)
    pd.testing.assert_frame_equal(back, frame)
    assert _same_items(back["name"].array, frame["name"].array)
