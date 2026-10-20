"""A column of many chunks is copied in one call, chunk for chunk.

p08's 30 parsed log files come in 384 chunks each. The store copied them
one at a time, a text chunk cast to ``large_string`` and back. The C
extension now copies all of a column's chunks in one call into buffers
Arrow allocated (``copy_view_chunks``, ``copy_fixed_chunks``): the same
values, chunks, nulls and keys, in memory nothing else holds.
"""

import hashlib
import random

import pytest

pa = pytest.importorskip("pyarrow")

from cash import _native
from cash.backends import frame_sharing

pytestmark = pytest.mark.skipif(_native.kernels is None, reason="no C extension")


def _word(rnd):
    return "".join(rnd.choice("abcdé中xyz") for _ in range(rnd.choice([0, 1, 5, 12, 13, 40, 300])))


def _column(kind, nulls, cut):
    rnd = random.Random(f"{kind}{nulls}{cut}")
    chunks = []
    for n in [0, 1, 7, 300, 0, 64, 1000]:
        if kind in ("string_view", "binary_view"):
            values = [_word(rnd) for _ in range(n)]
            if kind == "binary_view":
                values = [v.encode() for v in values]
        elif kind == "float32":
            values = [rnd.random() for _ in range(n)]
        else:
            values = [rnd.randint(-300, 300) for _ in range(n)]
        values = [None if nulls and rnd.random() < 0.2 else v for v in values]
        chunk = pa.array(values, type=getattr(pa, kind)())
        if cut and len(chunk) > 2:
            chunk = chunk.slice(1, len(chunk) - 2)
        chunks.append(chunk)
    return pa.chunked_array(chunks, type=getattr(pa, kind)())


def _not_owned(monkeypatch):
    """Columns Arrow allocated itself, read as memory someone else holds,
    as a polars frame's is."""
    monkeypatch.setattr(frame_sharing, "_chunk_owned", lambda chunk: False)


@pytest.mark.parametrize("kind", ["string_view", "binary_view", "int16", "int64", "float32"])
@pytest.mark.parametrize("nulls", [False, True])
@pytest.mark.parametrize("cut", [False, True])
def test_the_copy_is_the_column_chunk_for_chunk(kind, nulls, cut, monkeypatch):
    column = _column(kind, nulls, cut)
    real_owned = frame_sharing._chunk_owned
    _not_owned(monkeypatch)
    calls = []
    together = frame_sharing._copied_together
    monkeypatch.setattr(frame_sharing, "_copied_together", lambda *a: calls.append(1) or together(*a))

    copy = frame_sharing._owned_arrow(column)

    assert calls and copy is not column
    assert copy.type == column.type and copy.equals(column)
    assert [len(c) for c in copy.chunks] == [len(c) for c in column.chunks]
    assert [c.null_count for c in copy.chunks] == [c.null_count for c in column.chunks]
    assert all(real_owned(chunk) for chunk in copy.chunks)
    spans = [b for c in column.chunks for b in c.buffers()]
    assert not any(frame_sharing._overlapping(c.buffers(), spans) for c in copy.chunks)
    if kind.endswith("view"):
        from cash.content_hashers import fold_view_column

        large = "large_string" if kind == "string_view" else "large_binary"
        one, two = hashlib.sha256(), hashlib.sha256()
        fold_view_column(one, column, large)
        fold_view_column(two, copy, large)
        assert one.digest() == two.digest()


def test_a_polars_text_chunk_keeps_only_the_bytes_its_rows_use(monkeypatch):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"t": [f"row {i}, past the inline bytes" for i in range(20_000)]})
    pieces = pl.concat([frame.slice(i * 1_000, 1_000) for i in range(20)], rechunk=False)
    column = pieces.to_arrow(compat_level=pl.CompatLevel.newest()).column(0)
    held = sum(b.size for c in column.chunks for b in c.buffers()[2:])

    copy = frame_sharing._owned_arrow(column)

    assert copy.equals(column)
    assert sum(b.size for c in copy.chunks for b in c.buffers()[2:]) <= held
    assert sum(b.size for c in copy.chunks for b in c.buffers()[2:]) == sum(
        len(v.as_py().encode()) for v in column if len(v.as_py()) > 12
    )
