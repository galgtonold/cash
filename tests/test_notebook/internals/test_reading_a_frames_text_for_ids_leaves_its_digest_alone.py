"""Reading a frame's Python-string text for ids leaves its kept digest alone.

After each statement cash reads a frame's text columns and labels for
numbers that could be an ``id()`` (`library_parts`). It read them through
``to_numpy()``, which for strings held as Python objects (pandas 3 without
pyarrow) goes by ``.array``: a handle the frame memo records, and for an
index one its cache keeps. So the frame was read in full again by every
statement after the first. Strings use pandas' Python storage explicitly,
so this runs with pyarrow installed too.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.decorator import arg_hashing
from cash.decorator.arg_hashing import cow_frame_digest, exposure_mark, frame_memoable
from cash.notebook.statement.identity_holders import library_parts

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

TEXT = pd.StringDtype("python", na_value=np.nan)


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "team": pd.array([f"t{i % 40}" for i in range(2_000)], dtype=TEXT),
            "x": np.arange(2_000, dtype=float),
        },
        index=pd.Index([f"r{i}" for i in range(2_000)], dtype=TEXT),
    )
    frame.columns = pd.Index(list(frame.columns), dtype=TEXT)
    return frame


def _walk(value) -> list:
    """`library_parts` of *value* and of every pandas part it gives."""
    seen = []
    todo = [value]
    while todo:
        parts = library_parts(todo.pop())
        for part in parts or ():
            seen.append(part)
            if isinstance(part, (pd.Index, pd.Series, pd.DataFrame)):
                todo.append(part)
    return seen


def test_reading_text_for_ids_records_no_handle():
    frame = _frame()
    digest = cow_frame_digest(frame)  # the memo stores it: handles are watched from here
    mark = exposure_mark()
    parts = _walk(frame)
    assert any(isinstance(part, np.ndarray) and part.dtype == object and "t1" in part for part in parts)
    del parts
    assert exposure_mark() == mark
    assert "array" not in frame.columns._cache
    assert "array" not in frame.index._cache
    assert frame_memoable(frame)
    assert cow_frame_digest(frame) == digest


def test_reading_text_for_ids_reads_what_to_numpy_gives():
    frame = _frame()
    columns = (frame["team"], frame.columns, frame.index)
    expected = [column.copy(deep=True).to_numpy(dtype=object).tolist() for column in columns]
    cow_frame_digest(frame)
    mark = exposure_mark()
    for column, values in zip(columns, expected):
        (read,) = [p for p in library_parts(column) if isinstance(p, np.ndarray)]
        assert read.tolist() == values
    assert exposure_mark() == mark
