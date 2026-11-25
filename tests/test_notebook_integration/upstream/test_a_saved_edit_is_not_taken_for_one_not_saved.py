"""A value the old code of an edited, saved cell left in memory is rebuilt,
also when cash has no earlier look at the notebook to compare with.

Running the ``%cash_on`` cell again starts cash's view of the notebook
afresh. A cell above was edited and saved before that and not run: its name
holds what the old statement made, and that statement is no longer in the
notebook. That is not an edit the user ran and did not save. The statement
that writes the name last in the notebook has never run, so the name is
rebuilt from the notebook.
"""

import pytest

C_ON = "import cash\n%cash_on"


@pytest.mark.timeout(180)
def test_after_the_cash_on_cell_runs_again(nb_runner):
    nb_runner.create_notebook([C_ON, "x = 1", "x = x + 100", "w = x * 2", "print(w)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("(x, w)") == "(101, 202)"

    nb_runner.set_cell_source(3, "x = x + 200")
    nb_runner.run_cells([1, 5])
    assert nb_runner.peek("(x, w)") == "(201, 402)"
    assert nb_runner.get_output(5).strip().endswith("402")


@pytest.mark.timeout(180)
def test_a_change_in_place(nb_runner):
    nb_runner.create_notebook([C_ON, "rows = [1.0, 2.0]", "rows[0] = 100.0", "w = sum(rows)", "print(w)"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, "rows[0] = 200.0")
    nb_runner.run_cells([1, 5])
    assert nb_runner.peek("(rows, w)") == "([200.0, 2.0], 202.0)"
