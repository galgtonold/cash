"""An Arrow column of thousands of small chunks keys as its rows do in one
chunk, missing values and the bytes under them included.

``pd.concat`` of 20,000 small frames keeps a chunk per frame in each text
column, and the digest read each one in several NumPy calls; runs of small
chunks are now joined before they are read, which must not change a key.
"""

from __future__ import annotations

import hashlib

import pytest

from cash.content_hashers import fold_arrow_column

pa = pytest.importorskip("pyarrow")


def _digest(column) -> str:
    h = hashlib.sha256()
    fold_arrow_column(h, column)
    return h.hexdigest()


@pytest.mark.parametrize(
    "arrow_type, values",
    [
        ("string", ["ab", None, "", "cde"]),
        ("large_string", ["ab", None, "", "cde"]),
        ("binary", [b"ab", None, b"", b"cde"]),
        ("bool_", [True, None, False, True]),
    ],
)
def test_many_small_chunks_key_like_one_chunk(arrow_type, values):
    typ = getattr(pa, arrow_type)()
    chunks = [pa.array(values[: 1 + i % 4], type=typ) for i in range(500)]
    chunks[17] = pa.array(values * 3, type=typ).slice(2, 7)
    chunks.insert(300, pa.array(values * 2000, type=typ))  # a big one between runs
    chunks.append(pa.array([], type=typ))
    column = pa.chunked_array(chunks, type=typ)
    assert _digest(column) == _digest(column.combine_chunks())
    other = pa.chunked_array([*chunks[:-2], pa.array(values[:1][::-1] * 2, type=typ)], type=typ)
    assert _digest(other) != _digest(column)


def test_the_bytes_under_a_missing_text_still_tell_columns_apart():
    def under_null(text: bytes):
        offsets = pa.array([0, 3, 3 + len(text)], type=pa.int32()).buffers()[1]
        valid = pa.array([True, False]).buffers()[1]
        return pa.Array.from_buffers(pa.string(), 2, [valid, offsets, pa.py_buffer(b"abc" + text)], null_count=1)

    one = pa.chunked_array([under_null(b"xyz")] * 100)
    two = pa.chunked_array([under_null(b"xyw")] * 100)
    assert _digest(one) != _digest(two)
