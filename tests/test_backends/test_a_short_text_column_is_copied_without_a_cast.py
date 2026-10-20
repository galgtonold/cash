"""A polars text chunk whose values all sit inline in their views is copied
by its buffers, not written out and back.

A string or binary view of 12 bytes or fewer holds its value in the view
itself, so the chunk lists no data buffer: the validity bitmap and the
views are all it holds. Its copy cast it to ``large_string`` and back,
3.3 s of p21's 10M-row events frame, whose text columns are all short.
"""

import pytest

pa = pytest.importorskip("pyarrow")

from cash.backends import frame_sharing


@pytest.mark.parametrize("kind", ["string_view", "binary_view"])
@pytest.mark.parametrize("cut", [None, (1, 3)])
def test_an_inline_view_chunk_is_copied_by_its_buffers(kind, cut, monkeypatch):
    arrow_type = getattr(pa, kind)()
    values = ["a", None, "bc", "x" * 12, ""]
    if kind == "binary_view":
        values = [None if v is None else v.encode() for v in values]
    chunk = pa.array(values, type=arrow_type)
    if cut is not None:
        chunk = chunk.slice(*cut)
    assert not any(b is not None and b.size for b in chunk.buffers()[2:])
    owned = frame_sharing._owned_chunk(chunk)
    assert owned.type == chunk.type
    assert owned.equals(chunk)
    assert owned.to_pylist() == chunk.to_pylist()
    assert owned.null_count == chunk.null_count
    assert not frame_sharing._overlapping(owned.buffers(), chunk.buffers())
    assert frame_sharing._chunk_owned(owned)


def test_a_chunk_with_long_values_is_still_copied_whole():
    chunk = pa.array(["a" * 20, None, "b"], type=pa.string_view())
    assert any(b is not None and b.size for b in chunk.buffers()[2:])
    owned = frame_sharing._owned_chunk(chunk)
    assert owned.equals(chunk)
    assert not frame_sharing._overlapping(owned.buffers(), chunk.buffers())


def test_a_polars_frame_of_short_text_is_owned_after_the_copy():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"kind": ["view", "click", None, "purchase"] * 1000, "n": list(range(4000))})
    table = frame.to_arrow(compat_level=pl.CompatLevel.newest())
    owned = frame_sharing.owned_arrow_data(table)
    assert owned.equals(table)
    for column in owned.columns:
        for chunk in column.chunks:
            assert frame_sharing._chunk_owned(chunk)


def _addresses(chunk):
    return {b.address for b in chunk.buffers() if b is not None and b.size}


@pytest.mark.parametrize("cut", [None, (1, 3)])
def test_the_copy_shares_no_buffer_round_trips_and_keys_as_the_cast_copy(cut):
    import pickle

    pl = pytest.importorskip("polars")
    from cash import content_hashers

    chunk = pa.array(["a", None, "bc", "x" * 12, "", "view"], type=pa.string_view())
    if cut is not None:
        chunk = chunk.slice(*cut)
    owned = frame_sharing._owned_chunk(chunk)
    assert not _addresses(owned) & _addresses(chunk)
    assert owned.buffers()[0] is None or owned.buffers()[0].address != chunk.buffers()[0].address
    assert pickle.loads(pickle.dumps(owned)).equals(chunk)
    cast = pa.concat_arrays([chunk.cast(pa.large_string())]).cast(chunk.type)
    keys = {
        content_hashers.hash_polars(pl.from_arrow(pa.table({"s": array})))
        for array in (chunk, owned, cast, pickle.loads(pickle.dumps(owned)))
    }
    assert len(keys) == 1
