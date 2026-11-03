"""A loop re-run to bring a cell below up to date prints nothing in that cell.

The loop is another cell's statement: a plain run of the cell below never
shows its lines. Replayed statements ran silent already, but a loop's body
statements stream what they print as they finish, silent or not (the
cell's own loop runs that way), so ``print('Training the', estimator)`` of
a training loop re-run for ``voting_clf.score(X_valid, y_valid)`` printed
under the score (E8, homl ch07).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on",
    "base = 1",
    "out = []\nfor i in range(2):\n    print('Training', i)\n    out.append(i + base)",
    "print('sum', sum(out))",
]


def test_the_rerun_loop_prints_nothing_below(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Training 1" in nb_runner.get_output(3), "the loop's own cell shows its lines"
    nb_runner.set_cell_source(2, "base = 2")

    nb_runner.run_cell(4)

    out = nb_runner.get_output(4)
    assert "sum 5" in out, "the loop was re-run on the edited base"
    assert "Training" not in out, out


def test_the_loop_run_in_its_own_cell_still_prints(nb_runner):
    """The control: run as its cell, the loop shows what it prints."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, "base = 2")

    nb_runner.run_cell(2)
    nb_runner.run_cell(3)

    assert "Training 0" in nb_runner.get_output(3) and "Training 1" in nb_runner.get_output(3)
    nb_runner.run_cell(4)
    assert "sum 5" in nb_runner.get_output(4)
