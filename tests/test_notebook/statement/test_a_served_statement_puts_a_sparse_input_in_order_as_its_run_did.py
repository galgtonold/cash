"""A served statement puts a sparse input in canonical form, as its run did.

scipy sorts a CSR matrix's indices in place when it is merely read
(``NMF().fit(X)``, ``X > 0``). The values do not change, so cash does not
count it as changing ``X``, and a served ``nmf = NMF(...).fit(X)`` left
``X`` unsorted where a run sorts it: ``X.indices`` and ``X.data`` came in
another order than without cash. The run notes the inputs it put in order;
a hit does the same.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from cash.notebook.cache_status import CacheStatus

sparse = pytest.importorskip("scipy.sparse")

DEFINITIONS = """import numpy as np
import scipy.sparse as sp
rows = np.array([0, 0, 1, 1, 2])
cols = np.array([3, 1, 2, 0, 1])
"""
MAKE = "X = sp.csr_matrix((np.arange(1.0, 6.0), (rows, cols)), shape=(3, 4))\nX.indices[:] = X.indices[[1, 0, 3, 2, 4]]\nX.data[:] = X.data[[1, 0, 3, 2, 4]]\nX.has_sorted_indices = False"
READ = "# @cash:persist\ntotal = (X > 0).sum()"

COMPUTED, RESTORED = CacheStatus.COMPUTED, CacheStatus.RESTORED


def _run(magics, cells):
    statuses = []
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
    ):
        for cell in cells:
            result = magics._cell_executor.execute_cell(cell, ttl=None, cell_id=magics.resolve_cell_id())
            magics._finalize_cell_body(cell, result)
            statuses = [CacheStatus.parse(m.get("status")) for m in result.all_metrics]
    return statuses


@pytest.fixture
def run(cash_magics):
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield lambda cells: _run(cash_magics, cells)


def test_a_hit_sorts_it_as_the_run_did(run, mock_shell):
    cells = [DEFINITIONS, MAKE, READ]
    assert run(cells) == [COMPUTED]
    ran = mock_shell.user_ns["X"]
    assert ran.has_canonical_format
    ran_indices, ran_data = ran.indices.tolist(), ran.data.tolist()
    assert run(cells) == [RESTORED]
    served = mock_shell.user_ns["X"]
    assert served is not ran
    assert served.has_canonical_format
    assert (served.indices.tolist(), served.data.tolist()) == (ran_indices, ran_data)
    assert mock_shell.user_ns["total"] == 5


def test_an_input_already_in_order_is_left_alone(run, mock_shell):
    cells = [DEFINITIONS, "X = sp.csr_matrix((np.arange(1.0, 6.0), (rows, cols)), shape=(3, 4))", READ]
    run(cells)
    assert run(cells) == [RESTORED]
    assert mock_shell.user_ns["X"].has_canonical_format
