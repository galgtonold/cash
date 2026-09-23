"""The RAM tier sizes a value with the same rules as every other size cash takes.

It had its own walk, which knew frames and anything with ``nbytes`` but not a
scipy sparse matrix: one of 8 MB counted as its ~50-byte Python wrapper, so
the byte cap never saw it. It now uses ``memory_footprint``, which shares the
frame, array and sparse rules with ``estimate_object_size`` -- and, unlike that
sampler, counts every value of a statement's variables.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from cash.backends.memory_backend import InMemoryBackend  # noqa: E402
from cash.object_hashing import estimate_object_size, memory_footprint  # noqa: E402


def _stored_size(value) -> int:
    backend = InMemoryBackend()
    backend.set("k", value)
    return backend._store["k"][0]["size"]


def test_a_sparse_matrix_counts_its_data():
    sp = pytest.importorskip("scipy.sparse")
    m = sp.random(2_000, 2_000, density=0.1, format="csr", random_state=0)
    data = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
    assert _stored_size({"variables": {"m": m}}) >= data
    assert memory_footprint(m) == estimate_object_size(m) == data


def test_the_big_value_between_small_ones_is_counted():
    """A sample of a dict's first and last values would miss it."""
    big = np.zeros(1_000_000)
    variables = {"a": 1, "big": big, "c": 2}
    assert _stored_size({"variables": variables}) >= big.nbytes
    assert estimate_object_size(variables) < big.nbytes, "the sampler's blind spot this guards against"


def test_a_value_referenced_twice_is_counted_once():
    big = np.zeros(1_000_000)
    assert memory_footprint({"df": big, "alias": big}) < 1.5 * big.nbytes


def test_sizes_are_python_ints():
    """Entry metadata holding a ``numpy.int64`` cannot be read without numpy."""
    assert type(memory_footprint(np.zeros(10))) is int
    assert type(memory_footprint({"a": np.zeros(10), "b": [np.zeros(3)]})) is int
