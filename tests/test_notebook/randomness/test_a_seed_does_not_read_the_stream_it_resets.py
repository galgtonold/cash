"""A bare seed call does not read the stream it resets.

``np.random.seed(42)`` moves numpy's stream, and the runtime records a
statement that moves a stream without spelling a draw as a hidden draw
(``model.fit()``). The seed was recorded so: from then on its key read the
stream it inherits -- whatever the last draw of the previous Run All left --
so the seed's key, and the stream every draw below it reads, changed on each
Run All, and no seeded draw was served again in the same kernel.
"""

from __future__ import annotations

import pytest

from cash.tracking.randomness import seed_epochs
from tests._cell_driver import run_cash_cell

SEED = "import numpy as np\nnp.random.seed(42)"
DRAW = "noise = np.random.normal(size=3)"


@pytest.fixture
def run(cash_magics, mock_shell, clean_backend):
    def _run(code):
        run_cash_cell(cash_magics, code)
        return mock_shell.user_ns

    _run.state = cash_magics._cell_executor.tracking_state
    return _run


def test_the_seed_keys_alike_on_every_run_all(run):
    epochs = []
    values = []
    for _ in range(3):
        run(SEED)
        epochs.append(seed_epochs()["numpy.random"])
        values.append(list(run(DRAW)["noise"]))
    assert epochs[0] == epochs[1] == epochs[2], "the seed's key moved with the stream it inherits"
    assert values[0] == values[1] == values[2]
    assert not run.state.observed_rng_statement_draws, "the seed was recorded as a draw"


def test_a_seed_whose_argument_may_draw_is_left_as_observed(run):
    run("import numpy as np\ndef make_seed():\n    return int(np.random.randint(100))")
    run("np.random.seed(make_seed())")
    assert any("numpy.random" in m for m in run.state.observed_rng_statement_draws.values())
