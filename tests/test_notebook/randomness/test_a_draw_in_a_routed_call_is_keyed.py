"""A draw hidden in a call routed through the call cache reaches the key.

The runtime records a statement it sees draw without spelling it
(``X = make_data(3)``) under the statement's text, and its key and the
simulation look it up so. It was recorded under the text that ran, which a
call routed through the call cache rewrites: the key never read the RNG
variable, so after a restart a new seed was served the value the old seed
drew.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash.notebook.lineage_formula import key_hidden_reads
from tests._cell_driver import run_cash_cell

SEED = "import numpy as np\nnp.random.seed(42)"
HELPER = "import time\ndef make_data(n):\n    time.sleep(0.2)  # slow enough to store\n    return np.random.normal(size=n)"
DRAW = "X = make_data(3)"


@pytest.fixture
def run(cash_magics, mock_shell, clean_backend):
    def _run(code):
        run_cash_cell(cash_magics, code)
        return mock_shell.user_ns

    _run.state = cash_magics._cell_executor.tracking_state
    return _run


def test_the_key_reads_the_stream_the_routed_call_drew_from(run):
    run(SEED)
    run(HELPER)
    run(DRAW)
    assert "__cash_rng__numpy.random" in key_hidden_reads(DRAW, run.state)


def test_after_a_restart_a_new_seed_is_not_served_the_old_draw(run):
    for _ in range(2):
        run(SEED)
        run(HELPER)
        run(DRAW)
    # A restart forgets which statements draw unseen.
    run.state.observed_rng_statement_draws.clear()
    run("np.random.seed(7)")
    got = list(run(DRAW)["X"])
    np.random.seed(7)
    assert got == list(np.random.normal(size=3))
