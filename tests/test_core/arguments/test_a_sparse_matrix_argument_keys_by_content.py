"""A scipy sparse matrix argument is keyed by its content.

Every sparse argument -- a TF-IDF matrix, a one-hot encoding, the standard
input of the sklearn functions people cache -- ran uncached: sizing it for a
cost note called ``len()``, which scipy defines only to raise, and the call
was reported as having an argument that "could not be hashed".
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cash import Cash, FileBackend
from cash.exceptions import CashCacheIneffectiveWarning

sp = pytest.importorskip("scipy.sparse")

FORMATS = ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"]


@pytest.fixture
def key(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    return lambda *args: c._args.hash_payload(args, {})


def _matrix(fmt, value=1.0, size=40):
    dense = np.zeros((size, 5))
    dense[size // 2, 3] = value
    dense[1, 1] = 2.0
    return sp.csr_matrix(dense).asformat(fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_sparse_matrix_keys_by_its_values(key, fmt):
    assert key(_matrix(fmt)) == key(_matrix(fmt))
    assert key(_matrix(fmt)) != key(_matrix(fmt, value=3.0))


def test_what_code_can_read_keys_apart(key):
    a = _matrix("csr")
    assert key(a) != key(a.astype(np.float32))
    assert key(a) != key(a.tocsc())
    assert key(a) != key(sp.csr_array(a))
    assert key(a) != key(_matrix("csr", size=41))
    with_zero = a.copy()
    with_zero.data[0] = 0.0  # an explicit zero is stored, and `.data` shows it
    dropped = with_zero.copy()
    dropped.eliminate_zeros()
    assert (dropped != with_zero).nnz == 0
    assert key(with_zero) != key(dropped)


def test_a_sparse_matrix_inside_a_list_keys_by_content(key):
    assert key([_matrix("csr")]) == key([_matrix("csr")])
    assert key([_matrix("csr")]) != key([_matrix("csr", value=3.0)])


def test_a_sparse_argument_caches_without_a_warning(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    runs = []

    @c.cache
    def col_sums(X):
        runs.append(1)  # @cash:assume-safe
        return np.asarray(X.sum(axis=0)).ravel().tolist()

    X = sp.random(200, 5, density=0.1, random_state=0, format="csr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", CashCacheIneffectiveWarning)
        first = col_sums(X)
        assert col_sums(X.copy()) == first
    assert len(runs) == 1
    Y = X.copy()
    Y.data[0] += 1.0
    assert col_sums(Y) != first
    assert len(runs) == 2
