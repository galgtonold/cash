"""An edit above a ``%reset`` cell that keeps the variables reaches the cell below.

``%reset out``, ``%reset in`` and ``%reset_selective`` with a pattern that
matches nothing leave ``x`` in the namespace. Running only the last cell
after editing ``x`` must repair ``x`` first, as it does with no reset cell.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


@pytest.mark.parametrize("mid", ["pass", "%reset -f out", "%reset -f in", "%reset_selective -f zzz"])
def test_the_cell_below_sees_the_edit(nb_runner, mid):
    nb_runner.create_notebook(["import cash\n%cash_on", "x = 1", mid, "y = x + 1\nprint('y =', y)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y = 2" in nb_runner.get_output(4)

    nb_runner.set_cell_source(2, "x = 2")
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "y = 3" in out, out
    assert nb_runner.peek("x") == "2"
