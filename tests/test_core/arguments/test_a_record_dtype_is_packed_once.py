"""A forest's trees share one padded node dtype: it is repacked once, not
once per tree, and every array is packed to the bytes it was before."""

from __future__ import annotations

import pytest

from cash import kept_state

np = pytest.importorskip("numpy")
recfunctions = pytest.importorskip("numpy.lib.recfunctions")

NODE = np.dtype({"names": ["left", "threshold", "missing"], "formats": ["i8", "f8", "u1"], "aligned": True})


def _nodes(seed: int):
    rng = np.random.default_rng(seed)
    a = np.zeros(50, dtype=NODE)
    a["left"] = rng.integers(0, 50, 50)
    a["threshold"] = rng.random(50)
    a["missing"] = rng.integers(0, 2, 50)
    return a


def test_each_array_is_packed_as_numpy_packs_it():
    for seed in range(5):
        a = _nodes(seed)
        packed = kept_state.packed_records(a)
        expected = a.astype(recfunctions.repack_fields(NODE, recurse=True))
        assert packed.dtype == expected.dtype
        assert packed.tobytes() == expected.tobytes()


def test_the_shared_dtype_is_repacked_once():
    kept_state._packed_dtype.cache_clear()
    for seed in range(20):
        kept_state.packed_records(_nodes(seed))
    info = kept_state._packed_dtype.cache_info()
    assert (info.misses, info.hits) == (1, 19)


def test_a_dtype_with_no_padding_is_left_as_it_is():
    a = np.zeros(4, dtype=[("x", "f8"), ("y", "i8")])
    assert kept_state.packed_records(a) is a


@pytest.mark.parametrize("first", ["plain", "record"])
def test_a_record_typed_dtype_is_packed_as_itself_whichever_comes_first(first):
    """``np.dtype((np.record, NODE))`` equals ``NODE`` and hashes alike, but
    numpy packs it keeping its record type: one cache entry for both gave
    whichever was packed first, and a key that hung on what was hashed
    before."""
    import pickle

    record = np.dtype((np.record, NODE))
    order = [NODE, record] if first == "plain" else [record, NODE]
    kept_state._packed_dtype.cache_clear()
    for dtype in order:
        a = _nodes(0).view(dtype)
        packed = kept_state.packed_records(a)
        expected = a.astype(recfunctions.repack_fields(dtype, recurse=True))
        assert packed.dtype.type is expected.dtype.type
        assert pickle.dumps(packed.dtype) == pickle.dumps(expected.dtype)
        assert packed.tobytes() == expected.tobytes()
