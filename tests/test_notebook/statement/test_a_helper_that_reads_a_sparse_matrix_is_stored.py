"""``s = heavy(X)`` with a sparse ``X`` that ``heavy`` only reads is stored.

scipy sorts the indices of a CSR/CSC/COO matrix IN PLACE when it is merely
read (``m.sum()``, ``m @ v``): the values stay, the layout of ``.indices`` and
``.data`` changes. The fingerprint a call to a user function takes of its
arguments compared that layout, so a TF-IDF matrix handed to a helper that
sums it was reported as "In-place mutation on: X" and the statement ran every
time. The fingerprint now compares the canonical form; a real write still
moves it.
"""

from __future__ import annotations

import pytest

from cash.mutation_fingerprint import mutation_fingerprint
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

sp = pytest.importorskip("scipy.sparse")
np = pytest.importorskip("numpy")


def _unsorted(fmt: str):
    """A matrix with unsorted indices, as TfidfVectorizer returns it."""
    make = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
    m = make((np.array([3.0, 1.0, 2.0]), np.array([2, 0, 1]), np.array([0, 3])), shape=(1, 3) if fmt == "csr" else (3, 1))
    m.has_canonical_format = False
    return m


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_a_read_that_sorts_the_indices_is_not_a_change(fmt):
    m = _unsorted(fmt)
    before = mutation_fingerprint(m)
    m.sum()
    assert m.has_canonical_format
    assert mutation_fingerprint(m) == before


def test_a_write_into_the_values_is_still_a_change():
    m = _unsorted("csr")
    before = mutation_fingerprint(m)
    m.data[0] += 1.0
    assert mutation_fingerprint(m) != before


HELPERS = (
    "import time\n"
    "import numpy as np\n"
    "import scipy.sparse as sp\n"
    "def heavy(m):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return float(m.sum())\n"
    "X = sp.csr_matrix((np.array([3.0, 1.0, 2.0]), np.array([2, 0, 1]), np.array([0, 3])), shape=(1, 3))\n"
    "X.has_canonical_format = False"
)


def test_a_helper_that_sums_a_sparse_matrix_is_stored(cash_magics, statement_processor):
    run_cash_cell(cash_magics, HELPERS, cells=[HELPERS])
    metrics = statement_processor.process_statement("s = heavy(X)")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert not metrics.get("uncacheable_reasons"), metrics.get("uncacheable_reasons")
    again = statement_processor.process_statement("s = heavy(X)")
    assert again["status"] == CacheStatus.RESTORED
