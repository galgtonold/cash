"""A SEEDED expensive random draw must restore from disk across a restart.

The seed-change work (CAS-234) established that changing a seed invalidates,
and that an unseeded draw is frozen. The other half of the contract is that a
*seeded* draw persisted to disk is genuinely RESTORED after a kernel restart --
not recomputed. Verified here directly, because the seed work touches exactly
this path and a regression would be a silent loss of the headline restart
speedup for seeded work.

The draw must exceed the ~0.1s persistence floor in a SINGLE statement, or it
stays RAM-only and the restart recomputes it (correctly, but the test would
prove nothing). An SVD over a seeded random matrix clears the floor and draws
from the global RNG in one statement.
"""
import pytest
from conftest import shows_cached

pytestmark = [pytest.mark.restore, pytest.mark.libraries]

C_ON = "import cash\n%cash_on\n%cash_badge print"

DRAW = (
    "import numpy as np\n"
    "np.random.seed(2024)\n"
    "val = float(np.linalg.svd(np.random.rand(900, 900), compute_uv=False).sum())\n"
    "print('VAL %.8f' % val)"
)


def _val(nb_runner, cell_num=2):
    for line in nb_runner.get_output(cell_num).splitlines():
        if line.startswith("VAL "):
            return float(line.split()[1])
    raise AssertionError(f"no VAL line in cell {cell_num}:\n{nb_runner.get_output(cell_num)}")


@pytest.mark.timeout(180)
def test_seeded_draw_is_restored_from_disk_after_restart(nb_runner):
    nb_runner.create_notebook([C_ON, DRAW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _val(nb_runner)

    # It must have reached disk, or the restart cannot restore it.
    assert "RAM+DISK" in nb_runner.get_raw_output(2), (
        "seeded draw did not persist to disk; the restart test would be vacuous"
    )

    nb_runner.restart()
    nb_runner.run_all()

    assert _val(nb_runner) == pytest.approx(cold), "seeded value changed across restart"
    raw = nb_runner.get_raw_output(2)
    assert shows_cached(raw), (
        "seeded draw was recomputed, not restored from disk, after a restart -- "
        f"the restart speedup is lost:\n{raw}"
    )
