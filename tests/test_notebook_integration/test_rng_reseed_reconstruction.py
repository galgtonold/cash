"""CAS-225 / ADR-017: editing a bare seed() cell without re-running it must
still give the draw its correct top-to-bottom value.

nb_runner reaches this bug (it writes a real .ipynb and `set_cell_source`
persists an edit without running the cell), so it is testable without the real
Jupyter driver. Oracle = the same cell sources run in order with no cash — the
correct clean-run value.
"""
import re

import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"
C_SEED0 = "import numpy as np\nnp.random.seed(0)"
C_SEED1 = "import numpy as np\nnp.random.seed(1)"
C_DRAW = "x = np.random.rand(200000)\nprint('X0', repr(float(x[0])))"
C_STD_SEED0 = "import random\nrandom.seed(0)"
C_STD_SEED1 = "import random\nrandom.seed(1)"
C_STD_DRAW = "y = [random.random() for _ in range(50000)]\nprint('X0', repr(y[0]))"


def _draw(runner, cell_num):
    m = re.search(r"X0\s+([0-9.eE+-]+)", runner.get_output(cell_num))
    return float(m.group(1)) if m else None


def _oracle(cells):
    """Run cell sources in order in a fresh namespace, no cash = the truth."""
    buf: list[str] = []
    ns = {'print': lambda *a, **k: buf.append(" ".join(str(x) for x in a))}
    for src in cells:
        exec(compile(src, "<oracle>", "exec"), ns)
    m = re.search(r"X0\s+([0-9.eE+-]+)", "\n".join(buf))
    return float(m.group(1)) if m else None


@pytest.mark.timeout(180)
def test_numpy_cross_cell_reseed_edit_without_rerun(nb_runner):
    correct = _oracle([C_SEED1, C_DRAW])          # seed(1) top to bottom
    nb_runner.create_notebook([C_ON, C_SEED0, C_DRAW])
    nb_runner.start_kernel()
    nb_runner.run_all()                            # cold: draw cached under seed(0)
    nb_runner.set_cell_source(2, C_SEED1)          # edit seed cell, persist, do NOT run it
    nb_runner.run_cell(3)                          # run only the draw
    got = _draw(nb_runner, 3)
    assert got == pytest.approx(correct, abs=1e-9), (
        f"draw did not reflect the edited seed: got {got}, "
        f"clean-run value is {correct}"
    )


@pytest.mark.timeout(180)
def test_stdlib_cross_cell_reseed_edit_without_rerun(nb_runner):
    correct = _oracle([C_STD_SEED1, C_STD_DRAW])
    nb_runner.create_notebook([C_ON, C_STD_SEED0, C_STD_DRAW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, C_STD_SEED1)
    nb_runner.run_cell(3)
    got = _draw(nb_runner, 3)
    assert got == pytest.approx(correct, abs=1e-9), (
        f"stdlib draw did not reflect the edited seed: got {got}, clean-run is {correct}"
    )


@pytest.mark.timeout(180)
def test_reseed_rerun_still_correct_and_warm_draw_unaffected(nb_runner):
    """Guard: the fix must not break the cases that already work.

    (a) A clean warm re-run (no edit) restores the draw — caching still works.
    (b) Editing AND re-running the seed cell gives the new value (CAS-223 path).
    """
    seed0 = _oracle([C_SEED0, C_DRAW])
    seed1 = _oracle([C_SEED1, C_DRAW])
    nb_runner.create_notebook([C_ON, C_SEED0, C_DRAW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _draw(nb_runner, 3) == pytest.approx(seed0, abs=1e-9)
    nb_runner.run_all()                            # warm: must still equal seed0 (cached)
    assert _draw(nb_runner, 3) == pytest.approx(seed0, abs=1e-9), "warm re-run changed the draw"
    nb_runner.set_cell_source(2, C_SEED1)
    nb_runner.run_cell(2)                          # re-run the seed cell too
    nb_runner.run_cell(3)
    assert _draw(nb_runner, 3) == pytest.approx(seed1, abs=1e-9), "reseed+rerun did not update"


@pytest.mark.timeout(180)
def test_editing_a_downstream_seed_cell_does_not_reach_an_upstream_draw(nb_runner):
    """The fix must only re-run UPSTREAM seed cells, never a downstream one.

    Draw at cell 3 with a downstream seed at cell 4. Editing cell 4 (which is
    after the draw) without running it must leave the cell-3 draw exactly as it
    was — the fix must not pull the edited downstream seed into the draw's
    upstream. (Asserted as stability across the edit, independent of the
    separate CAS-223 epoch-position behaviour.)
    """
    seed999 = _oracle(["import numpy as np\nnp.random.seed(999)", C_DRAW])
    nb_runner.create_notebook([C_ON, C_SEED0, C_DRAW, C_SEED1])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(4, "import numpy as np\nnp.random.seed(999)")  # edit DOWNSTREAM seed
    nb_runner.run_cell(3)                           # re-run only the upstream draw
    after = _draw(nb_runner, 3)
    # The one thing the fix must guarantee: it re-runs only UPSTREAM seed cells.
    # If it wrongly pulled in the edited downstream seed(999), the draw would be
    # seed999's value. (The draw's exact value is otherwise governed by the
    # separate CAS-223 global-epoch behaviour, which this test does not pin.)
    assert after != pytest.approx(seed999, abs=1e-9), (
        "the upstream draw took the DOWNSTREAM seed's value — the fix over-reached"
    )
