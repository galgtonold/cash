"""A slow function of the user's module that returns a list its argument
holds -- ``training_rows(ds)`` returning ``ds.rows``, ``ds`` an object of a
class from that module -- hands back that very list on the next Run All, as
plain Python does: a later ``rows.append(3)`` is seen
through the result. Served from the call cache it was a copy, and the cell
printed ``2 False`` where plain Jupyter prints ``3 True``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIB = """import time
class Dataset:
    def __init__(self, rows):
        self.rows = rows
def training_rows(ds):
    time.sleep(0.3)
    return ds.rows
"""

CELLS = [
    "import heldrowslib\nrows = [1, 2]",
    "ds = heldrowslib.Dataset(rows)\ntrain = heldrowslib.training_rows(ds)",
    "rows.append(3)",
    "print(len(train), train is rows)",
]


def _check(nb_runner):
    assert nb_runner.get_output(4).strip() == "3 True"
    assert nb_runner.peek("train is rows") == "True"


def test_a_second_run_all_hands_back_the_list_itself(nb_runner, tmp_path):
    (tmp_path / "heldrowslib.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    _check(nb_runner)
    nb_runner.run_all()
    _check(nb_runner)

