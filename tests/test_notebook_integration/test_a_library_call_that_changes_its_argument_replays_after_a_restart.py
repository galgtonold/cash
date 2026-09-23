"""A library call seen changing its argument is replayed after a restart.

Round 30, r30s4 (BLOCKING, 3/3 + 1/1 with no edit at all): after a kernel
restart, a cell below ``sc.pp.calculate_qc_metrics(adata, inplace=True)``
stopped with ``KeyError: 'n_genes_by_counts'``. The repair restored ``adata``
from ``adata = make()`` and never re-ran the QC call. The runtime had seen that
call change ``adata`` and recorded it; the simulation only considered a call's
arguments that were live in the namespace -- and after a restart nothing is.

Not scanpy-specific: any library call that changes its argument in place.
``heapq.heapify(xs)`` shows the silent form -- no error, the wrong answer.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print\nimport heapq, time",
    "def build():\n    time.sleep(0.3)\n    return [9, 4, 7, 1, 8, 2] * 200\nxs = build()",
    "heapq.heapify(xs)",
    "smallest_first = xs[0]",
    "print('FIRST', smallest_first)",
]


@pytest.mark.parametrize("edit", [False, True], ids=["no_edit", "edit_below"])
def test_after_a_restart_the_call_is_replayed_before_its_readers(nb_runner, edit):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "FIRST 1" in nb_runner.get_output(5), nb_runner.get_raw_output(5)

    nb_runner.restart()
    nb_runner.run_cell(1)
    if edit:
        nb_runner.set_cell_source(4, "smallest_first = xs[0] + 0")
    nb_runner.run_cell(5)
    assert "FIRST 1" in nb_runner.get_output(5), (
        "after a restart the heapify was not replayed:\n" + nb_runner.get_raw_output(5)
    )
