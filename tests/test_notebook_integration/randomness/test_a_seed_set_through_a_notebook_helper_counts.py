"""A seed set through a notebook helper (``set_seed(42)``) counts as a seed.

Changing ``set_seed(42)`` to ``set_seed(43)`` and running Run All, or Restart
& Run All, served the seeded draw below with the seed-42 numbers.
"""

from __future__ import annotations

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

np = pytest.importorskip("numpy")

HELPER = (
    "import time, random\n"
    "import numpy as np\n"
    "def slow(v):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2})\n"
    "    return v\n"
    "def set_seed(s):\n"
    "    random.seed(s); np.random.seed(s)\n"
)
DRAW = "X = slow(np.random.rand(2))\nprint('r', X.round(4).tolist())"


def _plain(seed):
    np.random.seed(seed)
    return f"r {np.random.rand(2).round(4).tolist()}"


def test_a_new_seed_on_run_all_and_after_a_restart(nb_runner):
    nb_runner.create_notebook([HELPER + "set_seed(42)", DRAW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _plain(42) in nb_runner.get_output(2)

    nb_runner.set_cell_source(1, HELPER + "set_seed(43)")
    nb_runner.run_all()
    assert _plain(43) in nb_runner.get_output(2)

    nb_runner.set_cell_source(1, HELPER + "set_seed(7)")
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert _plain(7) in nb_runner.get_output(2)
