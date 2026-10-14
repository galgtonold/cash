"""How an Arrow-backed pandas column is pickled for a store.

pandas' own pickle joins a column's chunks into one array first
(``ArrowExtensionArray.__getstate__`` -> ``combine_chunks``), a copy of
every value, and the disk write copied the result once more. A store
pickles the chunks themselves: the writer of a value cash owns hands
Arrow's own buffers to the file as they are, a caller's value has each
buffer copied once, and the digest a call's result is referred to by
reads the same buffers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from pandas.core.arrays.arrow import ArrowExtensionArray

from cash.backends.serialization import OUT_OF_BAND_MIN_BYTES, PickleSerializer
from cash.notebook.call_refs import digest_and_size

ROWS = 100_000


def _part(start: int) -> pd.DataFrame:
    rows = range(start, start + ROWS)
    return pd.DataFrame(
        {
            "s": pd.Series([f"text {i}" for i in rows], dtype="str"),
            "n": pd.Series(np.array(rows), dtype="int64[pyarrow]"),
        }
    )


def _frame() -> pd.DataFrame:
    """Two chunks per column, as a concat of two reads leaves them."""
    return pd.concat([_part(0), _part(ROWS)], ignore_index=True)


@pytest.fixture
def no_join(monkeypatch):
    def refuse(self):
        raise AssertionError("pandas' own pickle joined the chunks")

    monkeypatch.setattr(ArrowExtensionArray, "__getstate__", refuse)


def _arrow_addresses(df: pd.DataFrame) -> set[int]:
    return {
        buffer.address
        for name in df.columns
        for chunk in df[name].array._pa_array.chunks
        for buffer in chunk.buffers()
        if buffer is not None and buffer.size >= OUT_OF_BAND_MIN_BYTES
    }


def test_a_private_value_hands_out_arrows_own_buffers(no_join):
    df = _frame()
    _stream, buffers = PickleSerializer().serialize_split(df, copy=False)
    handed = {np.frombuffer(b, dtype=np.uint8).ctypes.data for b in buffers}
    assert _arrow_addresses(df) <= handed


def test_a_callers_value_has_each_buffer_copied_once(no_join):
    df = _frame()
    stream, buffers = PickleSerializer().serialize_split(df, copy=True)
    handed = {np.frombuffer(b, dtype=np.uint8).ctypes.data for b in buffers}
    assert not _arrow_addresses(df) & handed, "a buffer of the caller's was handed out"
    arrow_bytes = sum(df[name].array._pa_array.get_total_buffer_size() for name in df.columns)
    assert sum(len(b) for b in buffers) <= arrow_bytes + len(stream)


def test_the_digest_of_a_call_result_reads_the_chunks(no_join):
    df = _frame()
    found = digest_and_size(df)
    assert found is not None
    assert found == digest_and_size(df)
