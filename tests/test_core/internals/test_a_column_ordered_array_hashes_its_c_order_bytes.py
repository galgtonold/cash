"""A buffer that is not in C order is hashed over its C-order bytes.

A frame's ``to_numpy()`` is column-ordered: E8's MNIST array (70,000 x 784)
was copied to C order by ``memoryview.tobytes`` before its digest, 0.79 s
each time. numpy copies it in 0.35 s; the bytes, so the digest, are the
same.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cash.bulk_digest import _as_bytes_view, fold_buffer

ARRAYS = {
    "column-ordered floats": lambda: np.asfortranarray(np.arange(60.0).reshape(6, 10)),
    "a strided slice": lambda: np.arange(100, dtype=np.int32)[::3],
    "a transposed 3-d block": lambda: np.arange(120, dtype=np.int16).reshape(2, 3, 20).transpose(2, 0, 1),
    "booleans": lambda: (np.arange(40) % 3 == 0).reshape(4, 10).T,
    "complex numbers": lambda: (np.arange(24) + 1j).reshape(4, 6).T,
    "a big column-ordered array": lambda: np.asfortranarray(np.random.default_rng(0).random((3000, 400))),
}


def _folded(data):
    h = hashlib.sha256()
    fold_buffer(h, data)
    return h.hexdigest()


@pytest.mark.parametrize("make", ARRAYS.values(), ids=ARRAYS.keys())
def test_the_bytes_and_the_digest_are_those_of_tobytes(make):
    view = memoryview(make())
    assert not view.c_contiguous
    assert bytes(_as_bytes_view(view)) == view.tobytes()
    assert _folded(view) == _folded(memoryview(view.tobytes()))
