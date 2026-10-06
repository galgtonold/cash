"""A cell run with an unsaved edit is not undone by the next cell.

Edit ``x = slow(1)`` to ``x = slow(2)``, run it, don't save, run the cell
below: the check used to rebuild ``x`` from the saved ``x = slow(1)`` (from
the cache, or by running it) and the next cell silently saw the old value.
Without cell ids cash cannot tell which cell the edit belongs to, so it stops
and asks for a save instead; once saved, the run goes through.
"""

import time

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = pytest.mark.upstream

SLOW = "import time\ndef slow(v):\n    time.sleep(0.2)\n    return v\n"


@pytest.mark.parametrize("producer", ["x = 1", "x = slow(1)"], ids=["plain", "cached"])
def test_an_unsaved_run_is_kept_or_refused(nb_runner, producer):
    nb_runner.create_notebook([SLOW, producer, "y = x * 10"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.nb.cells[1].source = producer.replace("1", "2")
    nb_runner.run_cell(2)
    with pytest.raises(CellExecutionError) as raised:
        nb_runner.run_cell(3)
    assert "Save the notebook" in str(raised.value)
    assert nb_runner.peek("x") == "2", "the run value was undone"

    time.sleep(0.05)  # a new mtime marks the save
    nb_runner._save_notebook()
    nb_runner.run_cell(3)
    assert nb_runner.peek("(x, y)") == "(2, 20)"
