"""A statement re-run above a cell reads each input as a run from the top
has it at that statement, or nothing above is re-run.

An edit above re-runs ``X, Y = [base], [10, 20]``, and with it the
``Y.append(k)`` below that goes on filling ``Y``. That append reads the
``k = 5`` bound just above it, which nothing re-ran, and a later cell binds
``k = 100``: the append read that one, and ``Y`` became ``[10, 20, 100]``
where a run from the top gives ``[10, 20, 5]``. In E8 (homl ch07) the same
gap re-ran ``accuracy_score(y_test, y_pred)`` on the ``y_pred`` of a loop
below it, and the cell failed with ``UpstreamStateError`` on values a run
from the top never has.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on",
    "base = 1",
    "X, Y = [base], [10, 20]",
    "k = 5\nY.append(k)",
    "k = 100",
    "print('X', X)",
]


def test_a_statement_brought_in_after_its_producer_reads_the_k_above_it(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, "base = 2")

    nb_runner.run_cells([6])

    out = nb_runner.get_output(6)
    assert "X [2]" in out, out
    assert nb_runner.peek("Y") == "[10, 20, 5]", nb_runner.peek("Y")
    assert nb_runner.peek("k") == "100"
