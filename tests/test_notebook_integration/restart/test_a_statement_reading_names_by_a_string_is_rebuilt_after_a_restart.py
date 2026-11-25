"""After a kernel restart, a cell below a statement that reads a variable through ``eval`` or ``globals()`` runs.

``K = 2``, ``y = eval("K * 10")``, ``print(y)``: Run All, restart, run only
the last cell. The rebuild ran the ``eval`` statement before ``K`` was
rebuilt, and the cell stopped with a NameError or an UpstreamStateError
where Restart & Run All prints 20.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

CASES = {
    "eval_in_the_statement": ["K = 2", 'y = eval("K * 10")'],
    "globals_in_a_function_it_calls": ["K = 2\ndef f(x):\n    return x * globals()['K']", "y = f(10)"],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_cell_below_runs_after_a_restart(nb_runner, case):
    cells = [*CASES[case], "print('R', y)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 20" in nb_runner.get_output(3), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(3)

    assert "R 20" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
