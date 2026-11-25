"""The lineage a rewound stream is given stays out of the cells above it.

Before a draw cell re-run on its own, the upstream check puts the global
stream back and gives its RNG variable the lineage it has there in a run
from the top. The simulation's snapshots then take that lineage, but only
those from the cell of the last statement above that moved the stream: the
seed's cell above it holds the stream at another lineage. With the name
given to every snapshot, the next check read the stream before the first
draw at the lineage after it, and the first draw, re-run, drew the second
draw's number.
"""

from __future__ import annotations

import random

from cash.tracking.randomness import rng_virtual_var
from tests._cell_driver import run_cash_cell

CELLS = ["import random\nrandom.seed(11)", "a = random.randint(0, 10**6)", "b = random.randint(0, 10**6)"]


def test_the_upper_draw_run_after_the_lower_one_keeps_its_number(cash_magics, mock_shell):
    state = random.getstate()
    try:
        for cell in CELLS:
            run_cash_cell(cash_magics, cell, cells=CELLS)
        ns = mock_shell.user_ns
        random.seed(11)
        expected = (random.randint(0, 10**6), random.randint(0, 10**6))
        assert (ns["a"], ns["b"]) == expected

        run_cash_cell(cash_magics, CELLS[2], cells=CELLS)
        run_cash_cell(cash_magics, CELLS[1], cells=CELLS)
        assert (ns["a"], ns["b"]) == expected
        run_cash_cell(cash_magics, CELLS[2], cells=CELLS)
        assert (ns["a"], ns["b"]) == expected
    finally:
        random.setstate(state)


def test_the_seed_cell_snapshot_keeps_the_lineage_the_seed_gives(cash_magics):
    state = random.getstate()
    try:
        var = rng_virtual_var("random")
        lineage = cash_magics.tracking_state.variable_lineage
        run_cash_cell(cash_magics, CELLS[0], cells=CELLS)
        after_seed = lineage[var]
        run_cash_cell(cash_magics, CELLS[1], cells=CELLS)
        after_first = lineage[var]
        run_cash_cell(cash_magics, CELLS[2], cells=CELLS)
        assert after_seed != after_first

        run_cash_cell(cash_magics, CELLS[2], cells=CELLS)  # puts the stream back before it
        # The first draw, run again, reads the stream as the seed left it,
        # and so leaves it at the lineage it left it at before.
        run_cash_cell(cash_magics, CELLS[1], cells=CELLS)
        assert lineage[var] == after_first
        assert {var: after_seed} in cash_magics.tracking_state.rng_read_at_last_draw.values()
    finally:
        random.setstate(state)
