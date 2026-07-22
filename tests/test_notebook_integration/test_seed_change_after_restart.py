"""Changing a seed must not serve the previous seed's result after a restart.

The failing shape, found by an adversarial tester sweep against the 0.1.0
wheel and reproduced against a real Jupyter server:

    SEED = 12345 -> accuracy 0.94140000000000001
    edit to 999, re-run -> 0.94140000000000001   <- the 12345 answer
    fresh cache at 999  -> 0.94189999999999996   <- ground truth

Why a kernel RESTART is essential to this test, and why a unit test cannot
replace it: a draw hidden inside a called function is only discovered while
the statement runs, so the statement's cache key is built before cash knows
about it. The entry written on that first run therefore carries no seed
epoch, and every later run rebuilds exactly that key and matches it. In one
long-lived kernel the in-memory ledger papers over this; a restart empties
the ledger, the epoch-free key is what gets rebuilt, and the stale value
comes back. Restart-and-run-all is the headline workflow, so this is the
path that matters.

The draw is deliberately hidden behind a helper (no ``np.random`` appears in
the consuming statement) because that is what makes it invisible to static
analysis -- the same shape as an sklearn ``fit()`` with no ``random_state``.
"""
import pytest

pytestmark = [pytest.mark.restore, pytest.mark.upstream]

C_ON = "import cash\n%cash_on\n%cash_badge print"

# The sleep is load-bearing. Cross-process persistence has a ~0.1s compute
# floor, so a microsecond draw is never written to disk and the restart simply
# recomputes -- the test then passes whether or not the bug exists. Without it
# this file passed against the unfixed source.
SETUP = (
    "import numpy as np\n"
    "import time\n"
    "def make_value():\n"
    "    time.sleep(0.35)\n"
    "    return float(np.random.rand(200).sum())  # hidden from the AST"
)

SEED_CELL = "SEED = 12345\nnp.random.seed(SEED)"

CONSUMER = "value = make_value()\nprint('VALUE %.17f' % value)"


def _value(nb_runner, cell_num=4):
    out = nb_runner.get_output(cell_num)
    for line in out.splitlines():
        if line.startswith("VALUE "):
            return float(line.split()[1])
    raise AssertionError(f"no VALUE line in cell {cell_num} output:\n{out}")


@pytest.mark.timeout(300)
def test_seed_edit_after_restart_does_not_serve_the_old_seeds_value(nb_runner):
    nb_runner.create_notebook([C_ON, SETUP, SEED_CELL, CONSUMER])
    nb_runner.start_kernel()
    nb_runner.run_all()
    under_12345 = _value(nb_runner)

    # Restart, so the runtime-observed "this statement draws" verdict is gone
    # -- exactly the state in which the epoch-free key used to be rebuilt.
    nb_runner.restart()
    nb_runner.run_all()
    assert _value(nb_runner) == pytest.approx(under_12345), (
        "same seed after a restart should reproduce the same value"
    )

    # Now the user changes the seed and re-runs.
    nb_runner.set_cell_source(3, SEED_CELL.replace("SEED = 12345", "SEED = 999"))
    nb_runner.restart()
    nb_runner.run_all()
    under_999 = _value(nb_runner)

    assert under_999 != pytest.approx(under_12345), (
        "the seed changed but cash served the value computed under the previous "
        "seed -- silently, with a RESTORED badge"
    )

    # And it must be what seed 999 genuinely produces.
    expected = float(__import__("numpy").random.RandomState(999).rand(200).sum())
    assert under_999 == pytest.approx(expected), (
        f"recomputed, but not on seed 999's stream: got {under_999!r}, "
        f"expected {expected!r}"
    )
