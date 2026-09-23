"""CACHE-LOOP-GROWTH fires only on growth, and its advice fits the code.

Seen twice in one user's notebook: the warning told them to "move `# @cash:persist` off
the loop" in a notebook with no annotation at all, and it fired on a sweep that
REBINDS a same-sized dict per window (``sc = {mid: ... for mid in clean}``) --
nothing grows, each pass stores a different result. The guard summed every
stored size per statement, so ``N x size`` passed its ratio for any loop of
more than four passes.
"""

import pytest

pytest.importorskip("numpy")

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.timeout(600)]

SETUP = (
    "import time\n"
    "import numpy as np\n"
    "def slow(x):\n"
    # Past the 0.1 s persistence floor AND under the bytes-per-compute-second
    # ceiling: the guard counts disk writes only, and `value_policy` refuses a
    # value over 8 MiB that holds more than 128 MiB of cache per second of
    # compute. The accumulator reaches 96 MB by its last pass, so at 0.15 s it
    # crossed that at the third iteration and stopped reaching disk: the
    # amplification this file is about never accumulated and
    # CACHE-LOOP-GROWTH never fired. The pathology was prevented rather than
    # reported -- right for a user, useless for a test of the report. 1.0 s
    # keeps every pass on the right side of the ceiling (96 MB needs 0.75 s),
    # so the warning is reachable and this file measures it again.
    "    time.sleep(1.0)\n"
    "    return x"
)


def test_a_loop_that_rebinds_a_same_sized_value_does_not_warn(nb_runner):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            SETUP,
            "for w in range(7):\n    sc = slow(np.random.default_rng(w).random(2_000_000))",
            "print('SUM', round(float(sc.sum()), 3))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "SUM" in nb_runner.get_output(4)
    assert "CACHE-LOOP-GROWTH" not in nb_runner.get_raw_output(3), nb_runner.get_raw_output(3)


def test_the_advice_for_an_unannotated_loop_does_not_mention_the_annotation(nb_runner):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            SETUP,
            "acc = np.zeros(0)\n"
            "for i in range(12):\n"
            "    acc = slow(np.concatenate([acc, np.full(1_000_000, float(i))]))",
            "print('LEN', len(acc))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(3)
    assert "CACHE-LOOP-GROWTH" in raw, raw
    assert "@cash:persist" not in raw, raw
