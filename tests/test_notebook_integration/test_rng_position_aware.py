"""CAS-226 + CAS-227 (ADR-018): the global RNG is restored to its position-correct
state before a re-executed draw.

Both are symptoms of the RNG being tracked by a time-ordered side-channel rather
than cash's position-aware reconstruction. The fix records each random cell's
post-state and, before a drawing cell re-executes, restores the state of the
nearest upstream random cell — so the draw continues from the seed and stream
position it holds top-to-bottom, not from wherever the live state was last left.

Testable in plain pytest: nb_runner writes a real .ipynb and persists edits
without running the cell. Oracle = the same cell sources run in order, no cash.
"""
import re

import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"
C_SEED0 = "import numpy as np\nnp.random.seed(0)"
C_SEED1 = "import numpy as np\nnp.random.seed(1)"
C_DRAW_A = "a = np.random.rand(1)\nprint('A', a[0])"
C_DRAW_B = "b = np.random.rand(1)\nprint('B', b[0])"

# seed(0) stream: p0=0.5488135039273248  p1=0.7151893663724195  p2=0.6027633760716439
SEED0_P0 = 0.5488135039273248
SEED0_P1 = 0.7151893663724195


def _v(runner, n, tag):
    m = re.search(tag + r"\s+([0-9.eE+-]+)", runner.get_output(n))
    return float(m.group(1)) if m else None


@pytest.mark.timeout(180)
def test_cas227_edited_draw_rerun_uses_position_state(nb_runner):
    """Editing the 2nd draw and re-running it must give position-1's value."""
    nb_runner.create_notebook([C_ON, C_SEED0, C_DRAW_A, C_DRAW_B])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _v(nb_runner, 4, "B") == pytest.approx(SEED0_P1, abs=1e-9)
    nb_runner.set_cell_source(4, "b = np.random.rand(1)\nprint('BEDIT', b[0])")
    nb_runner.run_cell(4)
    got = _v(nb_runner, 4, "BEDIT")
    assert got == pytest.approx(SEED0_P1, abs=1e-9), (
        f"re-executed draw used the wrong stream position: got {got}, want p1 {SEED0_P1}"
    )


@pytest.mark.timeout(180)
def test_cas226_draw_above_a_later_seed_keys_on_its_own_seed(nb_runner):
    """A draw governed by an upstream seed, re-run out of order with a LATER seed
    present, must give the upstream seed's value — not the later seed's."""
    nb_runner.create_notebook([C_ON, C_SEED0, C_DRAW_A, C_SEED1])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _v(nb_runner, 3, "A") == pytest.approx(SEED0_P0, abs=1e-9)
    nb_runner.run_cell(3)   # re-run the draw out of order (global state is now at seed(1))
    got = _v(nb_runner, 3, "A")
    assert got == pytest.approx(SEED0_P0, abs=1e-9), (
        f"draw took a downstream seed's value: got {got}, want seed(0) p0 {SEED0_P0}"
    )


@pytest.mark.timeout(180)
def test_warm_rerun_all_is_unaffected(nb_runner):
    """Guard: a plain warm Run-All still restores every draw to its cached value."""
    nb_runner.create_notebook([C_ON, C_SEED0, C_DRAW_A, C_DRAW_B])
    nb_runner.start_kernel()
    nb_runner.run_all()
    a1, b1 = _v(nb_runner, 3, "A"), _v(nb_runner, 4, "B")
    nb_runner.run_all()
    assert _v(nb_runner, 3, "A") == pytest.approx(a1, abs=1e-9)
    assert _v(nb_runner, 4, "B") == pytest.approx(b1, abs=1e-9)
    assert a1 == pytest.approx(SEED0_P0, abs=1e-9)
    assert b1 == pytest.approx(SEED0_P1, abs=1e-9)


@pytest.mark.timeout(180)
def test_non_random_cell_rerun_is_untouched(nb_runner):
    """Guard: the mechanism only fires for drawing cells."""
    nb_runner.create_notebook([
        C_ON,
        C_SEED0,
        C_DRAW_A,
        "s = a[0] * 2\nprint('S', s)",   # pure, no RNG
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    s1 = _v(nb_runner, 4, "S")
    nb_runner.run_cell(4)
    assert _v(nb_runner, 4, "S") == pytest.approx(s1, abs=1e-9)
