"""The closure check reads below an Arrow column of a plain type only when
the value holds a function with a closure: a frame concatenated from
thousands of small ones keeps a chunk per piece, each read through its
``__reduce_ex__``, and none can hold a function."""

from __future__ import annotations

import pytest

from cash import closure_sharing

pa = pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")


def _frame():
    return pd.concat([pd.DataFrame({"t": ["a", "b"], "n": [1, 2]}) for _ in range(300)], ignore_index=True)


def _arrow_reductions(monkeypatch) -> list:
    seen: list = []
    reduce = closure_sharing.reduce_counted

    def counted(item, protocol):
        if isinstance(item, (pa.Array, pa.ChunkedArray)):
            seen.append(item)
        return reduce(item, protocol)

    monkeypatch.setattr(closure_sharing, "reduce_counted", counted)
    return seen


def test_no_arrow_chunk_is_read_without_a_function(monkeypatch):
    frame = _frame()
    if not isinstance(frame["t"].array, pd.arrays.ArrowStringArray):
        pytest.skip("this pandas keeps text in Python objects")
    seen = _arrow_reductions(monkeypatch)
    assert closure_sharing.closures_sharing({"df": frame}, 5.0).names == []
    assert seen == []


def test_the_chunks_are_read_when_a_function_is_held(monkeypatch):
    frame = _frame()
    if not isinstance(frame["t"].array, pd.arrays.ArrowStringArray):
        pytest.skip("this pandas keeps text in Python objects")
    seen = _arrow_reductions(monkeypatch)
    box = [1]
    assert closure_sharing.closures_sharing({"df": frame, "f": lambda: box}, 5.0).names == []
    # Walked as before, chunk by chunk.
    assert len(seen) > 300


def test_an_extension_array_is_read_as_before(monkeypatch):
    seen = _arrow_reductions(monkeypatch)
    storage = pa.array([1, 2])
    tensor = pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(pa.int64(), [1]), pa.FixedSizeListArray.from_arrays(storage, 1)
    )
    assert closure_sharing.closures_sharing({"x": [tensor]}, 5.0).names == []
    assert seen
