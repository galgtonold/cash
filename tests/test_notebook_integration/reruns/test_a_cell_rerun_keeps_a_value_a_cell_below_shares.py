"""A cell re-run on its own keeps a value that a cell below shares.

A cell that changes a value in place (``history.append('x')``) normally starts
from the value as the cells above leave it when re-run on its own. When a cell
below made another name share it (``log = history``), a rebuilt value would
be a new object under one name only, and the two names would stop sharing.
cash keeps the value, as a plain re-run does, and warns NOTEBOOK-SHARED-KEPT.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

CASES = {
    "an alias below": (["history = []", "history.append('x')", "log = history"], "log is history"),
    "a dict below": (["feats = []", "feats.append(1)", "cfg = {'f': feats}"], "cfg['f'] is feats"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_two_names_still_share_after_the_rerun(nb_runner, name):
    cells, shared = CASES[name]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    assert nb_runner.peek(shared) == "True"
    assert "NOTEBOOK-SHARED-KEPT" in nb_runner.get_raw_output(2)


def test_with_nothing_below_the_rerun_starts_from_the_cells_above(nb_runner):
    """Control: no name below shares the list, so it is rebuilt and the append
    is not applied twice."""
    nb_runner.create_notebook(["history = []", "history.append('x')", "n = 1"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    assert nb_runner.peek("history") == "['x']"
    assert "NOTEBOOK-SHARED-KEPT" not in nb_runner.get_raw_output(2)
