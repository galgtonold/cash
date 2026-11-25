"""After a restart and a first run of a lower cell, a saved edit to a cell
above reaches the cells below it.

Run All, restart, run the last cell: cash rebuilds what it reads from the
cells above. Then a cell above is edited and saved, and the last cell is run
again. The value the old code left in memory is not an edit the user ran and
did not save: the notebook is newer than it, and cash rebuilds it.

Oracle: the edited notebook run top to bottom in a plain interpreter.
"""

import pytest

pytestmark = pytest.mark.fresh_kernel

C_ON = "import cash\n%cash_on"


def _plain(cells: list[str], name: str = "w") -> str:
    ns: dict = {"__name__": "__main__"}
    for cell in cells:
        exec(cell, ns)
    return repr(ns[name])


def _edit_after_a_rebuild(nb_runner, cells: list[str], position: int, edit: str) -> str:
    """Run All, restart, run the last cell, save *edit* as the cell at
    *position* (0-based among *cells*), run the last cell again: ``w``."""
    last = len(cells) + 1
    nb_runner.create_notebook([C_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_cells([1, last])
    assert nb_runner.peek("w") == _plain(cells[:-1])
    nb_runner.set_cell_source(position + 2, edit)
    nb_runner.run_cell(last)
    return nb_runner.peek("w")


@pytest.mark.timeout(180)
def test_a_rebinding(nb_runner):
    cells = ["x = 1", "x = x + 100", "w = x * 2", "print(w)"]
    edit = "x = x + 200"
    assert _edit_after_a_rebuild(nb_runner, cells, 1, edit) == _plain([cells[0], edit, cells[2]])
    assert nb_runner.get_output(5).strip().endswith("402")
    assert nb_runner.peek("x") == "201"


@pytest.mark.timeout(180)
def test_a_list_item(nb_runner):
    cells = ["rows = [1.0, 2.0]", "rows[0] = 100.0", "w = sum(rows)", "print(w)"]
    edit = "rows[0] = 200.0"
    assert _edit_after_a_rebuild(nb_runner, cells, 1, edit) == _plain([cells[0], edit, cells[2]])


@pytest.mark.timeout(180)
def test_an_append_and_a_dict_item(nb_runner):
    cells = ["rows = [1.0]\nd = {'k': 1}", "rows.append(100.0)\nd['k'] = 100", "w = sum(rows) + d['k']", "print(w)"]
    edit = "rows.append(200.0)\nd['k'] = 200"
    assert _edit_after_a_rebuild(nb_runner, cells, 1, edit) == _plain([cells[0], edit, cells[2]])


@pytest.mark.timeout(180)
def test_the_body_of_a_loop(nb_runner):
    loop = "acc = []\nfor i in items:\n    acc.append(i * 2)"
    cells = ["items = [1, 2]", loop, "w = sum(acc)", "print(w)"]
    edit = loop.replace("i * 2", "i * 3")
    assert _edit_after_a_rebuild(nb_runner, cells, 1, edit) == _plain([cells[0], edit, cells[2]])


@pytest.mark.timeout(180)
def test_the_first_cell(nb_runner):
    """Positive control: an edit to the first cell was always seen."""
    cells = ["x = 1", "x = x + 100", "w = x * 2", "print(w)"]
    edit = "x = 5"
    assert _edit_after_a_rebuild(nb_runner, cells, 0, edit) == _plain([edit, *cells[1:3]])
