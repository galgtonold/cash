"""A ``# @cash:no-cache`` draw from an unseeded stream is new on every run.

Cash rewinds the ``random``, ``numpy.random`` and ``torch`` streams so a
re-run draw repeats the value a top-to-bottom run gave (the frozen draw), and a
cache hit puts the stream back where the cached statement left it. The
documented way out is ``no-cache``: the statement runs for real every time and
draws fresh. Each test runs the notebook top to bottom twice, as Run All does.
"""

from __future__ import annotations

from cash.notebook.upstream.rng_rewind import RngRewind
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
