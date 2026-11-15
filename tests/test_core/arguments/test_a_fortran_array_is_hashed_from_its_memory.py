"""A Fortran-ordered array (a nibabel image's data) is hashed from its memory
as it lies, not from a C-order copy made on every hit, and still keys by its
shape, dtype, layout and values: apart from its C-ordered copy, from its
transpose holding the same bytes, and anew when a value in it changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash import Cash, content_hashers
from cash.content_hashers import hash_numpy


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def test_an_f_array_keys_apart_from_every_array_holding_its_bytes():
    a = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    same_bytes_c = np.frombuffer(a.T.tobytes(), dtype=a.dtype).reshape(3, 4)  # a's memory, read in C order
    others = {
        "c copy": np.ascontiguousarray(a),
        "transpose": a.T,  # C-contiguous over the same memory
        "same bytes in C order": same_bytes_c,
        "other dtype": a.astype(np.float32, order="F"),
        "reshaped": np.asfortranarray(a.reshape(4, 3, order="F")),
    }
    digests = {name: hash_numpy(v) for name, v in others.items()}
    digests["f"] = hash_numpy(a)
    assert None not in digests.values()
    assert len(set(digests.values())) == len(digests), digests
    # Equal F arrays built apart key alike; so does an F view of its memory.
    assert hash_numpy(np.asfortranarray(np.arange(12.0).reshape(3, 4))) == digests["f"]
    assert hash_numpy(np.ascontiguousarray(a.T).T) == digests["f"]


def test_an_f_array_is_read_without_a_c_order_copy(monkeypatch):
    a = np.asfortranarray(np.arange(24.0).reshape(2, 3, 4))
    handed = []
    real = content_hashers.fold_buffer

    def watched(h, data):
        handed.append(data)
        return real(h, data)

    monkeypatch.setattr(content_hashers, "fold_buffer", watched)
    assert hash_numpy(a) is not None
    (data,) = handed
    assert isinstance(data, memoryview) and data.c_contiguous  # the memory itself, not a copy
    assert data.obj.base is a or data.obj is a or np.shares_memory(data.obj, a)


def test_a_changed_value_in_an_f_array_misses(c):
    runs = []

    @c.cache
    def total(image):
        runs.append(1)
        return float(image.sum())

    image = np.asfortranarray(np.ones((4, 5, 6)))
    assert total(image) == 120.0
    assert total(np.asfortranarray(np.ones((4, 5, 6)))) == 120.0
    assert len(runs) == 1
    image[3, 4, 5] = 2.0
    assert total(image) == 121.0
    assert len(runs) == 2
    assert total(np.ascontiguousarray(image)) == 121.0  # C order: keyed apart
    assert len(runs) == 3
