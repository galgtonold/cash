"""A draw re-executed because an ORDINARY input changed must still land at its
top-to-bottom stream position (ADR-017).

The RNG state is a side-effect dependency a draw consumes, but it binds no
variable, so nothing in the lineage graph links a draw back to its ``seed()``.
When reconstruction re-executes a draw because a *deterministic* upstream value
changed (not the seed), the unchanged seed is not re-run, so the draw would
re-draw from wherever the live stream was last left -- yielding a value that
matches NEITHER a cache-off run NOR a clean top-to-bottom run (a silent wrong
value). ``UpstreamChecker._prepend_rng_chain_for_reexecuted_draws`` re-runs the
seed (and any draws ahead of the re-executed one) so the draw lands correctly.

Oracle = the notebook's current sources run top-to-bottom with no cash.
"""
import re

import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"


def _num(runner, n, tag):
    m = re.search(tag + r"\s+([0-9.eE+-]+)", runner.get_output(n))
    return float(m.group(1)) if m else None


@pytest.mark.timeout(180)
def test_deterministic_edit_in_seeded_draw_cell(nb_runner):
    """Edit a deterministic constant in a seed+draw cell, re-run downstream only."""
    data = "import numpy as np\nnp.random.seed(42)\nMULT = 100.0\narr = np.random.rand(3) * MULT"
    nb_runner.create_notebook([C_ON, data, "total = float(arr.sum())\nprint('TOTAL', round(total, 6))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, data.replace("MULT = 100.0", "MULT = 200.0"))
    nb_runner.run_cell(3)  # data cell NOT re-run

    import numpy as np
    np.random.seed(42)
    oracle = float((np.random.rand(3) * 200.0).sum())
    got = _num(nb_runner, 3, "TOTAL")
    assert got == pytest.approx(oracle, abs=1e-6), (
        f"re-executed draw ignored its seed: cash={got} oracle={oracle}"
    )


@pytest.mark.timeout(180)
def test_unchanged_draw_before_a_reexecuted_draw(nb_runner):
    """A draw ahead of the re-executed one must re-run to advance the stream, so
    the re-executed draw keeps its position (not reset to 0)."""
    data = (
        "import numpy as np\nnp.random.seed(0)\n"
        "SCALE = 10.0\n"
        "x = np.random.rand(2)\n"          # unchanged draw, ahead of y
        "y = np.random.rand(2) * SCALE"    # re-executes when SCALE changes
    )
    nb_runner.create_notebook([C_ON, data, "print('SX', round(float(x.sum()), 6), 'SY', round(float(y.sum()), 6))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, data.replace("SCALE = 10.0", "SCALE = 20.0"))
    nb_runner.run_cell(3)

    import numpy as np
    np.random.seed(0)
    ox = float(np.random.rand(2).sum())
    oy = float((np.random.rand(2) * 20.0).sum())
    assert _num(nb_runner, 3, "SX") == pytest.approx(ox, abs=1e-6)
    assert _num(nb_runner, 3, "SY") == pytest.approx(oy, abs=1e-6), (
        "re-executed draw was reset to position 0 instead of continuing after the "
        "earlier unchanged draw"
    )


@pytest.mark.timeout(180)
def test_later_draw_position_preserved(nb_runner):
    """A draw AFTER the re-executed one whose value is unaffected by the edit
    (same number of values consumed) must keep its cached value."""
    data = (
        "import numpy as np\nnp.random.seed(7)\n"
        "K = 3.0\n"
        "a = np.random.rand(4) * K\n"       # re-executes when K changes (draw count unchanged)
        "b = np.random.rand(4)"             # later draw, position unaffected
    )
    nb_runner.create_notebook([C_ON, data, "print('SA', round(float(a.sum()), 6), 'SB', round(float(b.sum()), 6))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, data.replace("K = 3.0", "K = 5.0"))
    nb_runner.run_cell(3)

    import numpy as np
    np.random.seed(7)
    oa = float((np.random.rand(4) * 5.0).sum())
    ob = float(np.random.rand(4).sum())
    assert _num(nb_runner, 3, "SA") == pytest.approx(oa, abs=1e-6)
    assert _num(nb_runner, 3, "SB") == pytest.approx(ob, abs=1e-6)


@pytest.mark.timeout(180)
def test_warm_rerun_is_untouched(nb_runner):
    """No edit: re-running the consumer must not disturb a correct seeded draw."""
    data = "import numpy as np\nnp.random.seed(1)\nv = np.random.rand(3)"
    nb_runner.create_notebook([C_ON, data, "print('SV', round(float(v.sum()), 6))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = _num(nb_runner, 3, "SV")
    nb_runner.run_cell(3)
    import numpy as np
    np.random.seed(1)
    oracle = float(np.random.rand(3).sum())
    assert _num(nb_runner, 3, "SV") == pytest.approx(first, abs=1e-9) == pytest.approx(oracle, abs=1e-6)
