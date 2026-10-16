"""A seed set by a function of the user's module keys the draws below it.

``seeding.set_seed(2)`` calls ``np.random.seed`` inside a module cash does not
read as seeding, so no statement recorded the seed, and the draw below was
keyed as unseeded: its first value was frozen, and a run seeded with 2 was
served seed 1's draw with a RESTORED badge. The process sees the seed (cash
wraps ``np.random.seed``), and where the stream then stands keys the draw.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(300)]

np = pytest.importorskip("numpy")

from tests._nbharness.badge import shows_cached

SEEDING = "import numpy as np\n\ndef set_seed(s):\n    np.random.seed(s)\n"
C_ON = "import cash\n%cash_on\n%cash_badge print"
SEED_CELL = "import numpy as np\nimport seeding\nseeding.set_seed({seed})"
# One statement, slow enough to be written to disk, so a restart can restore it.
DRAW = "import time\nx = (time.sleep(0.35), float(np.random.normal()))[1]\nprint('VALUE %r' % x)"


def _value(nb_runner) -> float:
    for line in nb_runner.get_output(3).splitlines():
        if line.startswith("VALUE "):
            return float(line.split()[1])
    raise AssertionError(nb_runner.get_output(3))


def _plain(seed: int) -> float:
    return float(np.random.RandomState(seed).normal())


def test_each_seed_gets_its_own_draw_and_a_repeated_seed_restores(nb_runner):
    (nb_runner.work_dir / "seeding.py").write_text(SEEDING, encoding="utf-8")
    nb_runner.create_notebook([C_ON, SEED_CELL.format(seed=1), DRAW])
    nb_runner.start_kernel()
    for run, seed in enumerate((1, 2, 1)):
        nb_runner.set_cell_source(2, SEED_CELL.format(seed=seed))
        if run:
            nb_runner.restart()
        nb_runner.run_all()
        assert _value(nb_runner) == _plain(seed), f"run {run + 1}: seed {seed} was served another seed's draw"
    assert shows_cached(nb_runner.get_output(3)), "the same seed after a restart recomputed instead of restoring"
