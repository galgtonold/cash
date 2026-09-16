"""An edited statement that has not run yet is run before its value is used.

Round 24's r24s5 changed ``models = {k: fit_series(g, params_for(k[1])) ...}``
to ``models = {k: fit_series(g, PARAMS, cutoff) ...}`` in cell 7, together
with ``def fit_series``, then ran a cell further down that needed only the
function, and then the export, whose back-test loop reads ``models``. The
loop re-ran; ``models = {...}`` did not, and the loop used the dict the OLD
statement had built. Old and new code happened to agree there.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

CELLS = [
    "import cash\n%cash_on",
    "N = 2",
    "def f(x):\n    return x * 10\nmodels = f(N)",
    "g = f(1)\nprint('G', g)",
    "rows = []\nfor i in range(2):\n    rows.append(models + i)\nprint('ROWS', rows)",
    "print('TOTAL', sum(rows))",
]
EDITED = "def f(x, k=10):\n    return x * k\nmodels = f(N, 100)"


@pytest.mark.parametrize("read_at", [5, 6], ids=["loop_cell", "cell_below_the_loop"])
@pytest.mark.parametrize("via_function_cell", [True, False], ids=["function_used_first", "direct"])
def test_the_loop_below_sees_the_edited_statement(nb_runner, via_function_cell, read_at):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "ROWS [20, 21]" in nb_runner.get_output(5)

    nb_runner.set_cell_source(3, EDITED)
    if via_function_cell:
        nb_runner.run_cell(4)
        assert "G 10" in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_cell(read_at)

    want = "ROWS [200, 201]" if read_at == 5 else "TOTAL 401"
    assert want in nb_runner.get_output(read_at), nb_runner.get_output(read_at)
