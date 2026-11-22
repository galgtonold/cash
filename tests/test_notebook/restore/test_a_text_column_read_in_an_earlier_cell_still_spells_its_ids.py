"""An Arrow text column whose reading for ids is kept from cell to cell
still gives every ``id()`` it spells, and never the reading of other text.

The reading is kept by what names the column's text: each chunk's buffers
(address and size), offset and length, while an array holding those
buffers is alive. A slice of the same buffers, or new text built where
freed text was, is other text and is read for itself: a value spelling
``str(id(p))`` in it keeps the frame from being restored on its own.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import end_cell_reads, forget_flat_reads, identity_held_reason

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


class P:
    pass


ROWS = 5_000


def _texts(key: str | None) -> list[str]:
    texts = [f"w{i:07d}" for i in range(ROWS)]  # 8 bytes, as long as an id()'s text: read
    if key is not None:
        texts[-1] = key
    return texts


def _frame(column) -> pd.DataFrame:
    return pd.DataFrame({"s": pd.array(column, dtype="string[pyarrow]")})


@pytest.fixture(autouse=True)
def _fresh():
    forget_flat_reads()
    yield
    forget_flat_reads()


def test_the_next_cell_still_finds_the_id():
    p = P()
    df = _frame(_texts(str(id(p))))
    assert identity_held_reason("df", df, {"p": p, "df": df}) is not None
    end_cell_reads()
    kept = df.assign(k=1)  # the same Arrow array
    assert identity_held_reason("kept", kept, {"p": p, "df": df, "kept": kept}) is not None
    df["n"] = range(ROWS)
    merged = df.merge(df[["n"]], on="n")  # the same buffers, wrapped anew
    assert merged["s"].array._pa_array is not df["s"].array._pa_array
    assert identity_held_reason("merged", merged, {"p": p, "df": df, "merged": merged}) is not None


def test_a_slice_of_the_same_buffers_is_read_for_itself():
    p = P()
    whole = pa.chunked_array([pa.array(_texts(str(id(p))), type=pa.large_string())])
    head = whole.slice(0, ROWS - 1)  # the same buffers, without the id
    first = _frame(pd.arrays.ArrowStringArray(head))
    assert identity_held_reason("first", first, {"p": p, "first": first}) is None
    df = _frame(pd.arrays.ArrowStringArray(whole))
    assert identity_held_reason("df", df, {"p": p, "df": df}) is not None


def test_new_text_where_freed_text_was_is_read_for_itself():
    # Memory the test owns, so the new text is sure to be where the old was:
    # the same addresses, sizes, offset and length, other bytes.
    p = P()
    key = str(id(p))
    texts = _texts("x" * len(key))
    data = np.frombuffer("".join(texts).encode(), dtype=np.uint8).copy()
    offsets = np.concatenate([[0], np.cumsum([len(t) for t in texts])]).astype(np.int64)

    def frame() -> pd.DataFrame:
        array = pa.Array.from_buffers(pa.large_string(), ROWS, [None, pa.py_buffer(offsets), pa.py_buffer(data)])
        return _frame(pd.arrays.ArrowStringArray(pa.chunked_array([array])))

    old = frame()
    assert identity_held_reason("old", old, {"p": p, "old": old}) is None
    assert identity_holders._ARROW_READS  # read and kept
    del old
    gc.collect()
    assert identity_holders._ARROW_READS == {}  # went with the array
    data[-len(key) :] = np.frombuffer(key.encode(), dtype=np.uint8)
    new = frame()
    assert new["s"].iloc[-1] == key
    assert identity_held_reason("new", new, {"p": p, "new": new}) is not None
