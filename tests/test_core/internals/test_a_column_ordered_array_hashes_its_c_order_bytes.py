"""A buffer that is not in C order is hashed over its C-order bytes.

A frame's ``to_numpy()`` is column-ordered: E8's MNIST array (70,000 x 784
int64) was copied to C order whole by ``memoryview.tobytes`` before its
digest, 0.8 s and another 440 MB each time. Put in order a block of rows
at a time it takes 0.25 s; the bytes, so the digest, are the same.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cash import bulk_digest as bd

ARRAYS = {
    "column-ordered floats": lambda: np.asfortranarray(np.arange(60.0).reshape(6, 10)),
    "a strided slice": lambda: np.arange(100, dtype=np.int32)[::3],
    "a transposed 3-d block": lambda: np.arange(120, dtype=np.int16).reshape(2, 3, 20).transpose(2, 0, 1),
    "booleans": lambda: (np.arange(40) % 3 == 0).reshape(4, 10).T,
    "complex numbers": lambda: (np.arange(24) + 1j).reshape(4, 6).T,
    "big-endian ints": lambda: np.arange(5000, dtype=">i8").reshape(50, 100).T,
    "a big column-ordered array": lambda: np.asfortranarray(np.random.default_rng(0).random((3000, 400))),
    "a big transposed 3-d block": lambda: np.random.default_rng(1).random((40, 30, 900)).transpose(2, 1, 0),
    "rows wider than a block": lambda: np.asfortranarray(np.random.default_rng(2).random((3, 200_000))),
}


def _folded(data):
    h = hashlib.sha256()
    bd.fold_buffer(h, data)
    return h.hexdigest()


@pytest.fixture(params=[bd._BLOCK_BYTES, 100_000], ids=["one block", "many blocks"])
def block_bytes(request, monkeypatch):
    monkeypatch.setattr(bd, "_BLOCK_BYTES", request.param)


@pytest.mark.parametrize("make", ARRAYS.values(), ids=ARRAYS.keys())
def test_the_digest_is_that_of_tobytes(make, block_bytes):
    view = memoryview(make())
    assert not view.c_contiguous
    ordered = memoryview(view.tobytes())
    assert bytes(bd._as_bytes_view(view)) == view.tobytes()
    assert _folded(view) == _folded(ordered)
    assert bd.bulk_digest(view) == bd.bulk_digest(ordered)
    hasher = bd.BulkHasher()
    hasher.update(view)
    assert hasher.digest() == bd.bulk_digest(ordered)
