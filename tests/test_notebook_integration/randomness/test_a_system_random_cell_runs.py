"""A cell that binds ``random.SystemRandom()`` runs as it does without cash.

Cash hashes a new variable after its statement runs, by pickling it, and
``SystemRandom`` refuses to pickle: it has no state to give. The error ended
the cell under cash. The value is hashed by identity instead, and the cells
below it run.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]

CELLS = [
    "import random\nsr = random.SystemRandom()\nprint('draw', sr.randrange(10) < 10)",
    "picks = [sr.random() < 1 for _ in range(3)]\nprint('picks', picks)",
]


def test_the_cells_run_and_print_what_plain_python_prints(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "draw True" in nb_runner.get_output(1)
    assert "picks [True, True, True]" in nb_runner.get_output(2)
    assert nb_runner.peek("type(sr).__name__").strip().strip("'") == "SystemRandom"
