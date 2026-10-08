"""What a line magic or a shell command binds or changes, as both the runtime
and the upstream simulation read it (``cash.notebook.magic_effects``).

The names it binds get a lineage of their own, so the upstream check does not
take the Python above the magic for their producer and rebuild them without it.
"""

from __future__ import annotations

import pytest

from cash.notebook.magic_effects import is_rerun_magic, magic_cell_of, magic_effects, simulation_cell


def _effects(cell, modules=()):
    source, tree = simulation_cell(cell)
    return magic_effects(tree.body[-1], set(modules).__contains__)


@pytest.mark.parametrize(
    ("cell", "changed"),
    [
        ("%time df = clean(df)", {"df"}),
        ("%time model.fit(X)", {"model"}),
        ("%timeit -n1 -r1 lst.append(1)", {"lst"}),
        ("%time df.head()", set()),
        ("files = !ls", {"files"}),
        ("res = {}\nres['ls'] = !ls", {"res"}),
        ("t = %time f(x)", {"t"}),
        ("!pip install numpy", set()),
        ("%matplotlib inline", set()),
        ("%time np.random.seed(0)", set()),
    ],
)
def test_what_it_changes(cell, changed):
    assert _effects(cell, modules={"np"})[0] == changed


def test_it_reads_what_its_statement_reads():
    changed, read = _effects("%time model.fit(X)")
    assert read == {"model", "X"}


@pytest.mark.parametrize("cell", ["x = 1", "%%bash\necho hi", "%%capture out\nx = 1", "x = (1,\n"])
def test_a_cell_without_line_magics_is_not_read_as_ipython(cell):
    assert simulation_cell(cell) is None


def test_a_cell_is_read_line_for_line_as_ipython_writes_it():
    source, tree = simulation_cell("x = 1\n%time y = x\nz = y")
    assert source.splitlines() == ["x = 1", "get_ipython().run_line_magic('time', 'y = x')", "z = y"]
    assert len(tree.body) == 3


def test_a_cell_of_magics_only_is_read_too():
    source, tree = simulation_cell("%time model.fit()\nfiles = !ls")
    assert len(tree.body) == 2


def test_the_cell_of_the_magic_that_changed_a_name():
    cells = ["x = 1", "model = M(x)\n%time model.fit()", "y = 2"]
    assert magic_cell_of("model", cells) == (1, "%time model.fit()")
    assert magic_cell_of("x", cells) is None


@pytest.mark.parametrize(
    ("cell", "rerun"),
    [
        ("%time model.fit()", True),
        ("t = %time f(x)", True),
        ("%timeit -n1 -r1 lst.append(1)", True),
        ("%prun -s cumulative f()", True),
        ("files = !ls", False),
        ("res = {}\nres['ls'] = !ls", False),
        ("%sx ls", False),
        ("%matplotlib inline", False),
        ("%time !ls", False),
        ("%time %time f()", False),
        ("%time get_ipython().system('ls')", False),
    ],
)
def test_only_a_magic_that_runs_python_is_run_again(cell, rerun):
    """A rebuild runs `%time`, `%timeit` and `%prun` again, as a top-to-bottom
    run does; a shell command or any other magic, never."""
    assert is_rerun_magic(simulation_cell(cell)[1].body[-1]) is rerun
