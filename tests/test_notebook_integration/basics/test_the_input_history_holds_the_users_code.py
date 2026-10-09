"""IPython's input history holds the code the user ran, with cash on.

Cash hands IPython a stand-in cell after running a cell's statements, and
IPython stored that stand-in (``pass``) as the cell's input: ``_i``, ``In``,
``%history`` and ``%save`` lost the user's code.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]


def test_in_and_underscore_i_hold_the_cells_source(nb_runner):
    nb_runner.create_notebook(["k = 2", "k * 10", "seen = _i"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("seen") == repr("k * 10")
    assert nb_runner.peek("'pass' in In") == "False"
    assert nb_runner.peek("[s for s in In if s in ('k = 2', 'k * 10')]") == repr(["k = 2", "k * 10"])


# %save writes the kernel's whole input history; a kernel reused from an
# earlier test would add that test's cells to the file.
@pytest.mark.fresh_kernel
def test_save_writes_the_cells_source(nb_runner):
    nb_runner.create_notebook(["k = 2", "k * 10", "%save -f saved.py 1-99999\nsaved = open('saved.py').read()"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    saved = nb_runner.peek("saved")
    assert "k = 2" in saved and "k * 10" in saved
    assert "pass" not in saved
