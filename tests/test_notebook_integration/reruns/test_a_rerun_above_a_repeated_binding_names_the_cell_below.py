"""An edited cell re-run above a binding that a cell above repeats word for word.

The shape of a book chapter that loads its frame twice: ``df = pd.read_csv(...)``
near the top, a cell that changes ``df`` in place, the edited cell that
reads it, and the same load again in a later section. After Run All the
live ``df`` is the later load. The edited cell re-run alone reads ``df`` as
the cells above leave it, as restart + Run All does, and cash says that a
cell below bound the value it does not read (NOTEBOOK-CHANGED-BELOW), naming
the cell that ran the binding last, not the copy above.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

pytestmark = [pytest.mark.timeout(300)]

LOAD = "df = pd.DataFrame({'review': ['a', 'b', 'c', 'd', 'e', 'f'], 's': range(6)})"


def _cells(cut: int) -> list[str]:
    return [
        "import time\nimport pandas as pd",
        LOAD,
        f"df['review'] = (time.sleep({F}), df['review'].str.upper())[1]",
        f"X_train = df.loc[:{cut}, 'review'].to_numpy()\nprint(list(X_train))",
        # The same statement; the cell differs, so the harness, which sends
        # no cell ids, can tell the two cells apart.
        f"{LOAD}\ndf.shape",
    ]


def _plain(cells: list[str]) -> dict:
    ns: dict = {}
    for src in cells:
        exec(compile(src, "<plain>", "exec"), ns)
    return ns


def test_the_edited_cell_names_the_load_below(nb_runner):
    nb_runner.create_notebook(_cells(3))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "NOTEBOOK-CHANGED-BELOW" not in nb_runner.get_raw_output(4)
    nb_runner.set_cell_source(4, _cells(2)[3])
    nb_runner.run_cell(4)
    raw = nb_runner.get_raw_output(4)
    assert "NOTEBOOK-CHANGED-BELOW" in raw, raw
    assert "'df'" in raw and "cell 5 below binds anew" in raw, raw
    # The value: what restart + Run All of the edited notebook has there.
    want = _plain(_cells(2)[:4])
    assert nb_runner.peek("list(X_train)") == repr(list(want["X_train"]))


def test_running_the_copy_above_first_is_quiet(nb_runner):
    """The load above run again binds ``df`` above the edited cell: what
    plain Jupyter reads too. The test above is the positive control."""
    nb_runner.create_notebook(_cells(3))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    nb_runner.set_cell_source(4, _cells(2)[3])
    nb_runner.run_cell(4)
    assert "NOTEBOOK-CHANGED-BELOW" not in nb_runner.get_raw_output(4)
