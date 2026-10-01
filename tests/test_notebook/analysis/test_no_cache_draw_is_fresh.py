"""A ``# @cash:no-cache`` draw from an unseeded stream is new on every run.

Cash rewinds the ``random``, ``numpy.random`` and ``torch`` streams so a
re-run draw repeats the value a top-to-bottom run gave (the frozen draw), and a
cache hit puts the stream back where the cached statement left it. The
documented way out is ``no-cache``: the statement runs for real every time and
draws fresh. Each test runs the notebook top to bottom twice, as Run All does.
"""

from __future__ import annotations

import random

from cash.notebook.upstream.rng_rewind import RngRewind
from cash.tracking.randomness import capture_rng_state, restore_rng_state
from tests._cell_driver import run_cash_cell


def _run_all(magics, cells: list[str]) -> None:
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


def _two_runs(magics, cells: list[str], name: str) -> tuple[float, float]:
    _run_all(magics, cells)
    first = magics.shell.user_ns[name]
    _run_all(magics, cells)
    return first, magics.shell.user_ns[name]


class TestTheDirectiveTurnsTheRewindOff:
    """Where the directive sits on the statement does not matter."""

    def test_on_a_line_of_its_own(self):
        assert RngRewind._opts_out_of_rng_rewind("# @cash:no-cache\nr = random.random()")

    def test_at_the_end_of_the_code_line(self):
        assert RngRewind._opts_out_of_rng_rewind("r = random.random()  # @cash:no-cache")

    def test_not_for_other_directives(self):
        assert not RngRewind._opts_out_of_rng_rewind("r = random.random()  # @cash:allow-random")

    def test_not_for_the_text_in_a_string(self):
        assert not RngRewind._opts_out_of_rng_rewind("r = random.random()\nnote = 'no-cache'")

    def test_in_a_cell_with_a_magic(self):
        assert RngRewind._opts_out_of_rng_rewind("%time r = 1\nr = random.random()  # @cash:no-cache")


def test_a_trailing_directive_draws_fresh(cash_magics):
    cells = ["import random", "r = random.random()  # @cash:no-cache"]
    first, second = _two_runs(cash_magics, cells, "r")
    assert first != second, "a trailing # @cash:no-cache must switch the rewind off like an own-line one"


def test_an_own_line_directive_draws_fresh(cash_magics):
    cells = ["import random", "# @cash:no-cache\nr = random.random()"]
    first, second = _two_runs(cash_magics, cells, "r")
    assert first != second


def test_a_draw_below_a_frozen_draw_is_fresh(cash_magics):
    """The frozen draw above rewinds the stream to where it started, so the
    no-cache draw below it began from the same position on every run."""
    cells = ["import random", "a = random.random()", "# @cash:no-cache\nr = random.random()"]
    _run_all(cash_magics, cells)
    first_a, first_r = cash_magics.shell.user_ns["a"], cash_magics.shell.user_ns["r"]
    _run_all(cash_magics, cells)
    assert cash_magics.shell.user_ns["a"] == first_a, "the draw above must stay frozen"
    assert cash_magics.shell.user_ns["r"] != first_r, "the no-cache draw below it must be drawn again"


def test_a_draw_after_a_replayed_statement_in_its_cell_is_fresh(cash_magics):
    """A cache hit puts the stream back where the cached statement left it."""
    cash_magics.cash_persist("on")  # store the cheap statement, so the second run replays it
    cells = ["import random", "x = 1\n# @cash:no-cache\nr = random.random()"]
    first, second = _two_runs(cash_magics, cells, "r")
    assert first != second


def test_every_run_draws_a_new_value(cash_magics):
    """Not just the second run: the stream a no-cache draw resumes must move on."""
    cells = ["import random", "a = random.random()", "# @cash:no-cache\nr = random.random()"]
    seen = []
    for _ in range(4):
        _run_all(cash_magics, cells)
        seen.append(cash_magics.shell.user_ns["r"])
    assert len(set(seen)) == len(seen), seen


def test_a_frozen_draw_below_stays_frozen(cash_magics):
    cells = [
        "import random",
        "a = random.random()",
        "# @cash:no-cache\nr = random.random()",
        "b = random.random()",
    ]
    _run_all(cash_magics, cells)
    first = cash_magics.shell.user_ns["a"], cash_magics.shell.user_ns["b"]
    _run_all(cash_magics, cells)
    assert (cash_magics.shell.user_ns["a"], cash_magics.shell.user_ns["b"]) == first


def test_a_seeded_draw_still_repeats(cash_magics):
    """A seeded stream is reproducible, so the no-cache draw gives what a
    clean top-to-bottom run gives: the same value each time."""
    cells = [
        "import random\nrandom.seed(0)",
        "a = random.random()",
        "# @cash:no-cache\nr = random.random()",
    ]
    first, second = _two_runs(cash_magics, cells, "r")
    random.seed(0)
    random.random()
    assert first == second == random.random()


def test_a_restore_records_the_earliest_position_it_moved_away_from():
    random.seed(1)
    live = capture_rng_state()
    random.seed(2)
    recorded = capture_rng_state()
    random.seed(1)

    displaced: dict = {}
    restore_rng_state(recorded, displaced)
    random.random()
    restore_rng_state(recorded, displaced)

    assert displaced["random"] == live["random"]
