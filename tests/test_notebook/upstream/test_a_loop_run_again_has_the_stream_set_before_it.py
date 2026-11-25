"""A loop or branch the upstream check runs again draws from where a run
from the top has the stream there.

The simulation models a control structure as one statement, and its trace
entry carries the lineage each global stream it draws from has as it is
entered. The random-stream plan reads that as it reads a plain draw's: it
runs the bare seed above again, or puts back the position recorded under
that lineage. Without it the plan took the loop for a statement that reads
no seeded stream and left the stream where it stood.
"""

from __future__ import annotations

import random
import warnings
from unittest.mock import MagicMock

from cash.notebook.upstream import UpstreamChecker
from cash.notebook.upstream._types import TraceEntry
from cash.tracking.randomness import rng_virtual_var
from tests._cell_driver import run_cash_cell

RANDOM = rng_virtual_var("random")
LOOP = "for i in items:\n    draws.append(random.random())"
CELLS = ["import random\nrandom.seed(3)\nitems = [1, 2, 3]", "draws = []\n" + LOOP, "r = list(draws)"]


def _plan(trace):
    shell = MagicMock()
    shell.user_ns = {"random": random, "items": [1, 2, 3], "draws": []}
    checker = UpstreamChecker(shell=shell)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, added = checker.rng.with_rng_chain(CELLS[2], CELLS, 2, [LOOP], trace, {})
    return out, added, [getattr(w.message, "code", None) for w in caught]


def _trace(rng_reads):
    return [
        TraceEntry("random.seed(3)", set(), set(), {}, {}, False, cell=0),
        TraceEntry(LOOP, {"draws", "i"}, {"items", "draws"}, {}, {}, False, cell=1, rng_reads=rng_reads),
    ]


def test_the_bare_seed_above_a_loop_that_reads_the_seeded_stream_runs_first():
    out, added, codes = _plan(_trace({RANDOM: "lineage-after-the-seed"}))
    assert out == ["random.seed(3)", LOOP]
    assert added == {"random.seed(3)"}
    assert codes == []


def test_a_loop_that_reads_the_stream_unseeded_is_left_alone():
    """Positive control for the read: an unseeded stream has no place a run
    from the top fixes, so nothing is run for it."""
    out, added, _ = _plan(_trace({RANDOM: ""}))
    assert out == [LOOP]
    assert added == set()


def test_a_loop_run_again_after_an_edit_above_draws_as_a_run_from_the_top(cash_magics, mock_shell):
    state = random.getstate()
    try:
        for cell in CELLS:
            run_cash_cell(cash_magics, cell, cells=CELLS)
        edited = [CELLS[0].replace("[1, 2, 3]", "[1, 2, 3, 4]"), *CELLS[1:]]
        run_cash_cell(cash_magics, edited[2], cells=edited)

        random.seed(3)
        assert mock_shell.user_ns["r"] == [random.random() for _ in range(4)]
    finally:
        random.setstate(state)
