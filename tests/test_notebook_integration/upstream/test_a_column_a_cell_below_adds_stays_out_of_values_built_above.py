"""A column a cell below adds in place stays out of a value built above.

``played`` is made in cell 3, ``hist`` is built from it in cell 4 and read in
cell 5, and cell 6 adds a column to ``played`` in place. Run from the top,
``hist`` never has that column. Re-running cell 5 re-ran cell 4 on the
changed frame, and ``hist`` silently took the column: cash recorded cell 6's
statement unparsed (``played['gd'] = ...``), did not find that text in the
cell, which writes ``played["gd"] = ...``, and so took the change for an
unsaved edit above cell 5.

When an edit does make cell 4 re-run, it must read ``played`` as the cells
above leave it, as a run from the top does.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

HELPER = "def run(m, k=2):\n    return m.assign(p=m['h'] * k), k\n"
CELLS = [
    "import cash\n%cash_on",
    "import pandas as pd\nimport elo_helper\nsrc = pd.DataFrame({'h': [1, 2, 3, 4], 'a': [0, 2, 1, 4]})",
    "played = src.dropna()\nplayed = played.reset_index(drop=True)",
    "hist, fin = elo_helper.run(played, k=2)",
    "out = hist.sum()\nprint('COLS', list(hist.columns), 'P', int(out['p']))",
    'played["gd"] = played["h"] - played["a"]\nplayed.groupby("h")["gd"].mean()',
]


def _columns(nb_runner, name):
    # Through ``globals()``: a peek naming the variable is a statement cash
    # brings up to date first.
    return nb_runner.peek(f"list(globals()[{name!r}].columns)")


@pytest.fixture
def notebook(nb_runner):
    (nb_runner.work_dir / "elo_helper.py").write_text(HELPER, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "COLS ['h', 'a', 'p'] P 20" in nb_runner.get_output(5)
    assert "gd" in _columns(nb_runner, "played")
    return nb_runner


def test_a_reader_run_again_keeps_the_value_built_above(notebook):
    notebook.run_cell(5)

    assert "COLS ['h', 'a', 'p'] P 20" in notebook.get_output(5), notebook.get_output(5)[-600:]
    assert _columns(notebook, "hist") == "['h', 'a', 'p']"


def test_an_edited_builder_reads_the_frame_as_the_cells_above_leave_it(notebook):
    notebook.set_cell_source(4, "hist, fin = elo_helper.run(played, k=3)")
    notebook.run_cell(5)

    assert "COLS ['h', 'a', 'p'] P 30" in notebook.get_output(5), notebook.get_output(5)[-600:]
    assert _columns(notebook, "hist") == "['h', 'a', 'p']"
