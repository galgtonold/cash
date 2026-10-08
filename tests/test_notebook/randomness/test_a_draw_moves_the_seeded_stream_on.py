"""A draw from a seeded global stream moves the stream's lineage on.

``np.random.seed(42)`` gives numpy's stream a lineage, and a draw reads it.
When only the seed wrote it, a change in the draws above a draw (a cell that
draws 5 numbers instead of 3, a draw cell inserted, removed or moved) left the
draw below keyed as before, and the cache served the numbers it drew at the
old stream position. Each draw now leaves the stream at a lineage derived
from its own key, so the draws below key anew. A bare re-seed still keys the
same on every Run All, so an unchanged notebook is served from the cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash.tracking.randomness import advanced_rng_lineage, drawn_rng_vars, rng_virtual_var
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

NP = rng_virtual_var("numpy.random")
SETUP = f"import numpy as np, time\ndef slow(a):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return a"
SEED = "np.random.seed(42)"


@pytest.fixture
def run(cash_magics, mock_shell, clean_backend):
    def _run(*cells):
        statuses = []
        for code in cells:
            run_cash_cell(cash_magics, code)
            statuses += [str(m.get("status")) for m in cash_magics.cash_status("dict")["last_cell"]["statements"]]
        return statuses

    _run.ns = mock_shell.user_ns
    _run.lineage = cash_magics._cell_executor.tracking_state.variable_lineage
    return _run


def _plain(*sizes):
    np.random.seed(42)
    return [np.random.rand(n).tolist() for n in sizes]


def test_a_changed_draw_count_above_gives_the_draw_below_new_numbers(run):
    run(SETUP, SEED, "X = slow(np.random.rand(3))", "Y = slow(np.random.rand(2))")
    run(SEED, "X = slow(np.random.rand(5))", "Y = slow(np.random.rand(2))")
    assert run.ns["Y"].tolist() == _plain(5, 2)[1]


def test_an_inserted_draw_above_gives_the_draw_below_new_numbers(run):
    run(SETUP, SEED, "Y = slow(np.random.rand(2))")
    run(SEED, "X = np.random.rand(3)", "Y = slow(np.random.rand(2))")
    assert run.ns["Y"].tolist() == _plain(3, 2)[1]


def test_swapped_draws_get_the_numbers_of_their_new_place(run):
    run(SETUP, SEED, "X = slow(np.random.rand(3))", "Y = slow(np.random.rand(2))")
    run(SEED, "Y = slow(np.random.rand(2))", "X = slow(np.random.rand(3))")
    y, x = _plain(2, 3)
    assert (run.ns["X"].tolist(), run.ns["Y"].tolist()) == (x, y)


def test_a_draw_in_a_helper_moves_the_stream(run):
    helper = f"def make(n):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return np.random.rand(n)"
    run(SETUP, helper, SEED, "X = make(3)", "Y = make(2)")
    run(SEED, "X = make(3)", "Y = make(2)")
    run(SEED, "X = make(5)", "Y = make(2)")
    assert run.ns["Y"].tolist() == _plain(5, 2)[1]


def test_an_unchanged_notebook_is_served_from_the_cache_after_a_reseed(run):
    """The seed keys the same on every run, and so does each draw after it."""
    run(SETUP, SEED, "X = slow(np.random.rand(3))", "Y = slow(np.random.rand(2))")
    assert run(SEED, "X = slow(np.random.rand(3))", "Y = slow(np.random.rand(2))")[1:] == ["RESTORED", "RESTORED"]
    assert [run.ns["X"].tolist(), run.ns["Y"].tolist()] == _plain(3, 2)


def test_a_draw_run_again_without_a_reseed_draws_new_numbers(run):
    """As in a plain kernel: the stream has moved on since the first run."""
    run(SETUP, SEED, "X = slow(np.random.rand(3))")
    first = run.ns["X"].tolist()
    assert run("X = slow(np.random.rand(3))") == ["COMPUTED"]
    assert run.ns["X"].tolist() != first


def test_a_restored_draw_leaves_the_stream_where_the_run_did(run):
    run(SETUP, SEED, "X = slow(np.random.rand(3))")
    after_run = run.lineage[NP]
    run(SEED, "X = slow(np.random.rand(3))")
    assert run.lineage[NP] == after_run


def test_an_unseeded_stream_takes_no_lineage(run):
    run(SETUP, "X = slow(np.random.rand(3))", "Y = slow(np.random.rand(2))")
    assert NP not in run.lineage


def test_a_def_holding_a_draw_does_not_move_the_stream(run):
    run(SETUP, SEED)
    seeded = run.lineage[NP]
    run("def make(n):\n    return np.random.rand(n)")
    assert run.lineage[NP] == seeded


def test_only_a_seeded_stream_a_statement_draws_from_moves():
    lineage = {NP: "seeded"}
    assert drawn_rng_vars({NP, "x"}, "x = np.random.rand(2)", lineage) == {NP}
    assert drawn_rng_vars({NP}, "x = np.random.rand(2)", {}) == set()
    assert drawn_rng_vars({NP}, "x = (np.random.seed(1), np.random.rand(2))", lineage) == set()
    assert drawn_rng_vars({NP}, "def f():\n    return np.random.rand()", lineage) == set()


def test_the_lineage_after_a_draw_depends_on_the_draw_and_the_stream():
    assert advanced_rng_lineage("stmt:a", NP) == advanced_rng_lineage("stmt:a", NP)
    assert advanced_rng_lineage("stmt:a", NP) != advanced_rng_lineage("stmt:b", NP)
    assert advanced_rng_lineage("stmt:a", NP) != advanced_rng_lineage("stmt:a", rng_virtual_var("random"))
