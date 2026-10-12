"""Rows of an array a loop keeps as views move the array when changed.

``big = np.arange(6.0).reshape(3, 2)`` is itself a view of the buffer
``arange`` made, so ``big[i].base`` is that buffer, not ``big``: nothing
linked ``parts`` to ``big`` when the cell ran (it did when ``big`` came back
from the cache, owning its memory, which made the bug come and go). After
an edit to ``parts[0] *= 2`` and Run All, the cached ``slow(big)`` was served
for the old array.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\nimport time\nimport numpy as np\n"
    "def slow(x):\n    time.sleep(0.3)\n    return repr(np.asarray(x).ravel()[:6])",
    "big = np.arange(6.0).reshape(3, 2)\nparts = {}",
    "for i in range(3):\n    parts[i] = big[i]",
    "parts[0] *= 2",
    "r = slow(big)\nprint(r)",
]


def test_run_all_after_an_edited_change_reads_the_changed_array(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "array([0., 2., 2., 3., 4., 5.])" in nb_runner.get_output(5), "control: the first run"
    nb_runner.set_cell_source(4, "parts[0] *= 5")

    nb_runner.run_all()

    assert "array([0., 5., 2., 3., 4., 5.])" in nb_runner.get_output(5), nb_runner.get_output(5)
