"""A call of a notebook function restores after a restart when a global the
function reads was edited and run, and its ``def`` cell was not.

The call's key took the function's lineage, which folds the global as it
was when the ``def`` ran; after the restart the ``def`` ran over the edited
global, so every model fit calling ``make_prep()`` was fitted again with
its code and data unchanged (64 s in one churn notebook run headless). The
key reads the function by its code, and the global as it is at the call.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
CELLS = [
    SETUP,
    "import time\ncols = ['a', 'b']",
    "def make():\n    return [c * 2 for c in cols]",
    "r = (time.sleep(0.6), make())[1]\nprint('R', r)",
]


@pytest.mark.parametrize("notebook_known", [False, True], ids=["headless", "notebook"])
def test_the_call_restores_after_a_restart(nb_runner, notebook_known):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel(inject_notebook_path=notebook_known)
    nb_runner.run_all()
    nb_runner.set_cell_source(2, "import time\ncols = ['a', 'b', 'c']")
    nb_runner.run_cells([2, 4])
    out = nb_runner.get_output(4)
    assert "R ['aa', 'bb', 'cc']" in out, out
    assert not shows_cached(out), f"control: the edit re-keys the call\n{out}"

    nb_runner.restart()
    nb_runner.run_all()

    out = nb_runner.get_output(4)
    assert "R ['aa', 'bb', 'cc']" in out, out
    assert shows_cached(out), out

    nb_runner.set_cell_source(2, "import time\ncols = ['x']")
    nb_runner.run_cells([2, 4])
    out = nb_runner.get_output(4)
    assert "R ['xx']" in out, out
