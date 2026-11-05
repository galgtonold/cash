"""Which cell a run from the top cannot get through.

A cell that reads, when it runs, a name only a cell below it binds raises
``NameError`` there; the upstream check refuses to re-run anything at or
below it.
"""

from cash.notebook.upstream.notebook_vetting import first_forward_read

CELLS = (
    "import pandas as pd",
    "nps = {}\nfor v, g in sv.groupby('k'):\n    nps[v] = len(g)",
    "sv = pd.DataFrame({'k': [1]})",
    "out = dict(nps)",
)


def test_the_cell_reading_a_name_bound_below_is_found():
    assert first_forward_read(CELLS, 4) == (1, "sv", 2)


def test_only_cells_above_the_running_one_count():
    assert first_forward_read(CELLS, 1) is None


def test_a_name_read_inside_a_function_is_read_when_it_is_called():
    cells = ("def f():\n    return sv", "sv = 1", "f()")
    assert first_forward_read(cells, 3) is None


def test_a_name_the_cell_binds_itself_or_a_cell_above_binds_is_no_forward_read():
    cells = ("x = 1", "x = x + 1\ny = 2\nz = y", "x = 5")
    assert first_forward_read(cells, 3) is None


def test_a_comprehension_variable_is_no_read_of_the_notebook():
    cells = ("src = sum(i * i for i in range(3))", "for i in range(3):\n    pass", "out = src")
    assert first_forward_read(cells, 3) is None
