"""A cell holding a line of only spaces is found in the saved notebook without a cell id.

An editor leaves the indentation on a blank line inside a block, and the
saved ``.ipynb`` keeps it, while IPython empties every whitespace-only line
before it runs the cell. Without a cell id (classic Notebook, nbclient, a
replay harness) cash matched the cell by its text, so the two never matched:
the cell counted as unsaved and the check of the cells above it was skipped,
silently. An edit saved above then never reached it, unlike a restart and Run
All.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

READER = "def show(v):\n    print('x is', v)\n    \n    return v\n\nshow(x)"


@pytest.mark.parametrize(
    "reader",
    [READER, READER.replace("    \n", "\t\n", 1), READER.replace("\n", "\r\n") + "\r\n"],
    ids=["spaces", "tab", "crlf_and_trailing_newline"],
)
def test_an_edit_above_reaches_a_cell_with_a_line_of_spaces(nb_runner, reader):
    nb_runner.create_notebook(["import cash\n%cash_on", "x = 1", reader])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "x is 1" in nb_runner.get_output(3)

    # edited and saved, not run: a run from the top gives the reader x = 2
    nb_runner.set_cell_source(2, "x = 2")
    nb_runner.run_cell(3)

    assert "x is 2" in nb_runner.get_output(3), nb_runner.get_output(3)
