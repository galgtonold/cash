"""A big table's Arrow text in someone else's memory is kept checked, not copied.

`frame_sharing.share_checked` kept a big table's numbers and dates uncopied,
but copied every Arrow chunk that may be someone else's memory. ``read_csv``
hands its text columns over in the parser's memory, so every store of a frame
read from a CSV, and of every frame that kept one of its text columns,
copied that text again: 0.7 s for the four text columns of p18's 5.6 million
lab rows, on each of four statements. Now such a chunk is shared too, under
the tier's own wrapper, and its bytes go into the checked digest: a write
into that memory makes the next hit a miss, as a write past pandas into the
numbers does. Once the tier adopts the table, the chunk is copied into
memory Arrow allocated, and nothing checks it any more.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend

pytestmark = [
    pytest.mark.skipif(not frame_sharing._checked_works(), reason="needs pandas copy-on-write"),
    pytest.mark.skipif(frame_sharing._arrow_layout() is None, reason="Arrow's buffer layout not known here"),
]

#: Rows enough for 16 MiB of numbers: kept uncopied.
ROWS = frame_sharing.CHECKED_MIN_BYTES // 8 * 11 // 10
WORD = b"abcdefgh"


def _foreign_text(rows: int) -> tuple[pd.arrays.ArrowExtensionArray, np.ndarray]:
    """Text over numpy memory the caller keeps and can write, as a parser's
    memory under Arrow's name: ``WORD`` in every row."""
    data = np.frombuffer(WORD * rows, dtype=np.uint8).copy()
    offsets = np.arange(0, (rows + 1) * len(WORD), len(WORD), dtype=np.int32)
    chunk = pa.Array.from_buffers(pa.string(), rows, [None, pa.py_buffer(offsets), pa.py_buffer(data)])
    assert not frame_sharing._chunk_owned(chunk)  # someone else's memory, as read_csv's is
    return pd.arrays.ArrowExtensionArray(pa.chunked_array([chunk])), data


def _table() -> tuple[pd.DataFrame, np.ndarray]:
    text, data = _foreign_text(ROWS)
    return pd.DataFrame({"x": np.arange(float(ROWS)), "code": text}), data


def _expected() -> pd.DataFrame:
    """What `_table` holds, in memory of its own (``df.copy()`` shares Arrow's)."""
    text = pd.arrays.ArrowExtensionArray(pa.chunked_array([pa.array([WORD.decode()] * ROWS)]))
    return pd.DataFrame({"x": np.arange(float(ROWS)), "code": text})


def _text_address(frame: pd.DataFrame) -> int:
    return frame["code"].array._pa_array.chunks[0].buffers()[2].address


def test_the_store_shares_the_text_instead_of_copying_it():
    backend = InMemoryBackend()
    df, data = _table()
    backend.set("k", df, {"execution_time": 1.0})
    assert backend._checked["k"]
    stored = backend._store["k"][1]
    assert _text_address(stored) == data.ctypes.data
    assert stored["code"].array is not df["code"].array  # a wrapper of the tier's own


def test_a_write_into_that_memory_makes_the_next_hit_a_miss():
    backend = InMemoryBackend()
    df, data = _table()
    backend.set("k", df, {"execution_time": 1.0})
    data[:8] = np.frombuffer(b"ZZZZZZZZ", dtype=np.uint8)
    assert backend.get("k") == (None, None)


def test_an_unchanged_hit_is_served_and_adopted_with_the_text_copied():
    backend = InMemoryBackend()
    df, data = _table()
    expected = _expected()
    backend.set("k", {"df": df}, {"execution_time": 1.0})
    del df  # the caller rebinds: the numbers are the tier's alone
    gc.collect()
    first = backend.get("k")[1]["df"]
    pd.testing.assert_frame_equal(first, expected)
    assert "k" not in backend._checked  # adopted
    stored = backend._store["k"][1]["df"]
    assert frame_sharing._chunk_owned(stored["code"].array._pa_array.chunks[0])
    data[:8] = np.frombuffer(b"ZZZZZZZZ", dtype=np.uint8)  # reaches neither the entry nor a hit
    pd.testing.assert_frame_equal(backend.get("k")[1]["df"], expected)


def test_a_dictionary_column_is_still_copied():
    backend = InMemoryBackend()
    df, _data = _table()
    codes = pa.array((["a", "b"] * ROWS)[:ROWS]).dictionary_encode()
    df["cat"] = pd.arrays.ArrowExtensionArray(pa.chunked_array([codes]))
    expected = _expected()
    expected["cat"] = df["cat"].copy()
    backend.set("k", df, {"execution_time": 1.0})
    pd.testing.assert_frame_equal(backend.get("k")[1], expected)
