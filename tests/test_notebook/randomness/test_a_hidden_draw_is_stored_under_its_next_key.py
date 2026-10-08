"""A draw hidden in a helper is stored under the key the next run gives it.

``X = make_data(3)``, where ``make_data`` draws from numpy's seeded stream,
spells no draw: its first key does not read numpy's RNG variable. The run
sees it draw, so the next run's key reads the variable. The value was not
stored on the first run (under the first key a restart would serve it
across a new seed), so a second Run All in the same kernel ran the slow
draw again. It is now stored under the key the next run builds, when that
key is known when the value is stored.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from cash.notebook.statement.run import StatementRun

from tests._cell_driver import run_cash_cell

SEED = "import numpy as np\nnp.random.seed(42)"
HELPER = "import time\ndef make_data(n):\n    time.sleep(0.2)  # slow enough to store\n    return np.random.normal(size=n)"


@pytest.fixture
def run(cash_magics, mock_shell, clean_backend):
    def _run(code):
        run_cash_cell(cash_magics, code)
        return mock_shell.user_ns

    def statuses():
        return [str(m.get("status")) for m in cash_magics.cash_status("dict")["last_cell"]["statements"]]

    _run.statuses = statuses
    _run.state = cash_magics._cell_executor.tracking_state
    return _run


def _plain(seed: int, n: int = 3) -> list[float]:
    np.random.seed(seed)
    return list(np.random.normal(size=n))


def test_the_second_run_all_restores_the_hidden_draw(run):
    run(SEED)
    run(HELPER)
    first = list(run("X = make_data(3)")["X"])
    run(SEED)
    second = list(run("X = make_data(3)")["X"])
    assert run.statuses() == ["RESTORED"], f"the draw ran again on the second Run All ({run.statuses()})"
    assert first == second == _plain(42)


def test_after_a_restart_a_new_seed_is_not_served_the_old_draw(run):
    run(SEED)
    run(HELPER)
    run("X = make_data(3)")
    # A restart forgets which statements draw unseen: the key is the first one again.
    run.state.observed_rng_statement_draws.clear()
    run("np.random.seed(7)")
    got = list(run("X = make_data(3)")["X"])
    assert got == _plain(7)


def test_a_key_whose_inputs_changed_while_it_ran_is_not_moved(run, cash_magics):
    # The first key, built again after the run, differs (an input changed as
    # it ran): the next run's key is not known, so the write is skipped once.
    run(SEED)
    run(HELPER)
    processor = cash_magics._statement_processor
    processor._randomness.draw_newly_seen = True
    processor._randomness.newly_seen_draws = {"numpy.random"}
    stmt = StatementRun(code="X = make_data(3)", tree=ast.parse("X = make_data(3)"), cache_key="stmt:as-it-ran")
    processor._key_a_newly_seen_draw(stmt)
    assert stmt.cache_key == "stmt:as-it-ran"
    assert processor._randomness.draw_newly_seen, "the write would not be skipped"


def test_a_statement_that_seeds_and_draws_hidden_gives_the_plain_values(run):
    run(SEED)
    run(HELPER)
    values = [list(run("y = (np.random.seed(5), make_data(3))[1]")["y"]) for _ in range(3)]
    assert values == [_plain(5)] * 3
