"""The loop fast path: taken when it is safe, and only then.

A cheap loop with many iterations runs as one unit, because decomposing it
per iteration costs ~8 ms per body statement per iteration. Doing that means
evaluating the header twice, so the question of WHEN it is allowed is the
whole feature. It was once wrong in both directions.

Measured on a real notebook's cell, 627 iterations of four cheap numpy statements:

    range(0, len(frame), STEP)   first 2.52 s   re-run 4.51 s   decomposed
    range(0, NROWS, STEP)        first 0.06 s   re-run 0.05 s   one unit

-- `len` was not on the list of builtins trusted in a header. The user saw
16.9 s cached against 1.0 s uncached, a re-run slower than the first, and
called it BLOCKING. And the other way: `for x in sorted(g):` over a 400-item
generator ran zero times, because the check for one-shot iterators looked
only at the header's RESULT, and `sorted` returns a list.

The unit twin is ``tests/test_notebook/test_a_loop_header_is_safe_to_evaluate_twice.py``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = """import numpy as np
import pandas as pd
rng = np.random.default_rng(0)
frame = pd.DataFrame(rng.standard_normal((3131, 400)))
flags = rng.random(400) < 0.15
STEP = 5
TOPN = 50
"""

# The reported loop cell, unchanged.
LOOP = """M = frame.to_numpy()
share = []
for t in range(0, len(frame), STEP):
    s = M[t]
    valid = np.flatnonzero(np.isfinite(s))
    order = valid[np.argsort(s[valid])]
    share.append(float(flags[order[:TOPN]].mean()))
share = np.array(share)
print("iterations=" + str(len(share)))
"""


def test_a_loop_bounded_by_len_runs_as_one_unit(nb_runner):
    """The reported cell. Structural, not timed: which path did it take?"""
    nb_runner.create_notebook(["import cash\n%cash_on\n%cash_badge print", SETUP, LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    raw = nb_runner.get_raw_output(3)
    assert "iterations=627" in nb_runner.get_output(3), raw
    assert "LOOP x627" not in raw, (
        "the loop was decomposed into 627 iterations of four statements each; "
        "`len` in its header is not a side effect:\n" + raw
    )


@pytest.mark.parametrize("wrapper", ["sorted", "list"])
def test_a_generator_inside_the_header_is_not_drained_twice(nb_runner, wrapper):
    """The wrong answer, and the reason the fix is not just 'add len'."""
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            "g = (i for i in range(400))",
            "OUT = []\nfor x in " + wrapper + "(g):\n    OUT.append(x * 2)\nprint('N', len(OUT))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "N 400" in nb_runner.get_output(3), (
        "for x in %s(g) ran over a drained generator; plain Python gives "
        "400:\n%s" % (wrapper, nb_runner.get_raw_output(3))
    )
