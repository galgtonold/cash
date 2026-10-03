"""A repair rebuilds an intermediate the notebook deletes; it never reuses a stale one.

``m = ...``, ``m = [x + 1 for x in m]``, ``total = sum(m)``, ``del m`` in one
cell. A repair from below re-runs the first three and not the ``del``, so the
``m`` built from the old ``clean`` stays in the namespace. After the next edit
of ``clean``, the repair trusted that ``m`` because it was present (the end of
the notebook has no lineage for a deleted name to compare it with), re-ran
``m = [x + 1 for x in m]`` on it and left the statement above out: ``total``
came from the replaced cleaning, with the increment applied twice.

The notebook arm, with the frame shape of the original report, is
``tests/test_notebook_integration/upstream/test_a_deleted_intermediate_is_rebuilt_not_reused.py``.
"""

from tests._cell_driver import run_cash_cell


def _cleaning(k: int) -> str:
    return f"d = [x * 2 for x in raw]\nclean = [a + b + {k} for a, b in zip(raw, d)]"


AGGREGATE = "m = [c * 10 for c in clean]\nm = [x + 1 for x in m]\ntotal = sum(m)\ndel m"
PEAK = "d = total + 1"
REPORT = "result = d * 1"


def _expected(k: int) -> int:
    return sum((a + 2 * a + k) * 10 + 1 for a in range(5)) + 1


def test_an_intermediate_deleted_below_is_rebuilt_after_a_second_edit(cash_magics, mock_shell):
    cells = ["raw = list(range(5))", _cleaning(0), AGGREGATE, PEAK, REPORT]
    for code in cells:
        run_cash_cell(cash_magics, code, cells=cells)
    assert mock_shell.user_ns["result"] == _expected(0)

    cells[1] = _cleaning(100)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, REPORT, cells=cells)
    assert mock_shell.user_ns["result"] == _expected(100)

    cells[1] = _cleaning(200)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, REPORT, cells=cells)
    assert mock_shell.user_ns["result"] == _expected(200), "total was built from a stale m"
