"""An edited seed cell that was not re-run still decides a later draw.

Edit ``np.random.seed(0)`` to ``seed(1)``, run only a later draw: the draw must
give what the cells give top to bottom, as if the seed cell had been run. A
bare ``seed()`` binds no variable, so lineage alone cannot see the edit; cash
spots the seed cell whose source has not run this session and runs the random
chain from it first.

The simple two-cell case is in ``test_rng_reseed_reconstruction.py``. Here: a
draw between the seed and the target, a seed cell that also binds names, and a
restart, where every upstream cell counts as not run yet.

Oracle = the same cell sources run in order without cash.
"""

import re

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"
SEED0 = "import numpy as np\nnp.random.seed(0)"
SEED1 = "import numpy as np\nnp.random.seed(1)"
SLEEP = f"import time\ntime.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
WARMUP = f"a = np.random.rand(1000)\n{SLEEP}"
DRAW = f"x = np.random.rand(1000)\n{SLEEP}\nprint('X0', repr(float(x[0])))"


def _x0(text):
    m = re.search(r"X0\s+([0-9.eE+-]+)", text)
    return float(m.group(1)) if m else None


def _oracle(cells):
    """Run cell sources in order in a fresh namespace without cash: the truth."""
    buf: list[str] = []
    ns = {"print": lambda *a, **k: buf.append(" ".join(str(x) for x in a))}
    for src in cells:
        exec(compile(src, "<oracle>", "exec"), ns)
    return _x0("\n".join(buf))


@pytest.mark.timeout(240)
def test_a_draw_between_the_seed_and_the_target_moves_the_stream_under_the_new_seed(nb_runner):
    correct = _oracle([SEED1, WARMUP, DRAW])
    nb_runner.create_notebook([C_ON, SEED0, WARMUP, DRAW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _x0(nb_runner.get_output(4)) == pytest.approx(_oracle([SEED0, WARMUP, DRAW]), abs=1e-12)
    nb_runner.set_cell_source(2, SEED1)  # edited, saved, not run
    nb_runner.run_cell(4)
    assert _x0(nb_runner.get_output(4)) == pytest.approx(correct, abs=1e-12)


@pytest.mark.timeout(240)
def test_a_seed_cell_that_also_binds_names_still_counts(nb_runner):
    seed0 = "import numpy as np\nN = 3\nnp.random.seed(0)"
    seed1 = "import numpy as np\nN = 3\nnp.random.seed(1)"
    correct = _oracle([seed1, DRAW])
    nb_runner.create_notebook([C_ON, seed0, DRAW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, seed1)
    nb_runner.run_cell(3)
    assert _x0(nb_runner.get_output(3)) == pytest.approx(correct, abs=1e-12)


@pytest.mark.timeout(240)
def test_after_a_restart_an_edited_seed_still_reaches_the_draw(nb_runner):
    correct = _oracle([SEED1, WARMUP, DRAW])
    nb_runner.create_notebook([C_ON, SEED0, WARMUP, DRAW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.set_cell_source(2, SEED1)
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)
    assert _x0(nb_runner.get_output(4)) == pytest.approx(correct, abs=1e-12)
