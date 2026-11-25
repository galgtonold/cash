"""After a restart, a seed set through a function of the notebook is put
back before a cell below draws.

``set_seed(5)`` moved the streams last, and the new kernel holds neither
the function nor a recorded position. The random-stream plan runs the call
again when the function only seeds, with its ``def`` and the imports its
body reads ahead of it. A function that does more is not run again for the
stream alone: cash says the stream was not put back.

The planner is a function of the notebook's cells, the planned statements,
the live namespace and what the kernel recorded, so these tests drive it
without a kernel.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

from cash.notebook.upstream import UpstreamChecker
from cash.notebook.upstream._types import InputHashes, TraceEntry
from cash.tracking.randomness import rng_virtual_var

NP = rng_virtual_var("numpy.random")
IMPORT = "import numpy as np, random"
DEF = "def set_seed(s):\n    random.seed(s)\n    np.random.seed(s)"
DRAW = "r = np.random.rand(3)"


def _plan(cells, statements, live=None):
    shell = MagicMock()
    shell.user_ns = dict(live or {})
    checker = UpstreamChecker(shell=shell)
    checker.tracking_state.simulated_lineage[NP] = "lineage-after-the-seed"
    trace = [TraceEntry("set_seed(5)", set(), {"set_seed"}, InputHashes({}), {}, False, cell=0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, added = checker.rng.with_rng_chain(DRAW, [*cells, DRAW], len(cells), statements, trace, {})
    return out, added, [getattr(w.message, "code", None) for w in caught], checker


def test_the_call_runs_again_with_its_def_and_imports_first():
    cells = [f"{IMPORT}\n{DEF}\nset_seed(5)", "n = 3"]
    out, added, codes, _ = _plan(cells, ["n = 3"])
    assert out == [IMPORT, DEF, "set_seed(5)", "n = 3"]
    assert "set_seed(5)" in added
    assert codes == []


def test_a_function_that_does_more_is_not_run_again_and_cash_says_so():
    noisy = DEF.replace("    random.seed(s)", "    print('seeding', s)\n    random.seed(s)")
    cells = [f"{IMPORT}\n{noisy}\nset_seed(5)", "n = 3"]
    out, added, codes, checker = _plan(cells, ["n = 3"])
    assert out == ["n = 3"]
    assert added == set()
    assert codes == ["RANDOM-STREAM-UNRESTORED"]
    assert "numpy.random" in checker.tracking_state.rng_unplaced


def test_a_function_defined_twice_is_not_defined_again():
    """Which ``def`` the call ran with is not the planner's to choose."""
    cells = [f"{IMPORT}\n{DEF}\nset_seed(5)", DEF.replace("np.random.seed(s)", "np.random.seed(s + 1)"), "n = 3"]
    out, _, codes, _ = _plan(cells, ["n = 3"])
    assert out == ["n = 3"]
    assert codes == ["RANDOM-STREAM-UNRESTORED"]
