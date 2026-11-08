"""A served plot run whose statements cannot all be accounted for runs instead.

``settle_served_plot_run`` puts each statement's random streams where its
run left them and moves the RNG ledgers on. When one statement fails there,
the streams, the RNG variables' lineages and the seed ledger go back to where
they stood, and the caller is told to run the plot: nothing is half served.
"""

from __future__ import annotations

import random

from cash.tracking.randomness import capture_rng_state, rng_virtual_var


def test_a_failure_part_way_puts_the_streams_and_ledgers_back(statement_processor, monkeypatch):
    random.seed(11)
    before = capture_rng_state()["random"]
    var = rng_virtual_var("random")
    statement_processor.tracking_state.lineage.record(var, "seeded-above")
    real = statement_processor.settle_served_plot_statement
    calls = []

    def settle(code, occurrence, rng_state, reseeded):
        calls.append(code)
        if len(calls) == 2:
            raise RuntimeError("cannot key it")
        real(code, occurrence, rng_state, reseeded)

    monkeypatch.setattr(statement_processor, "settle_served_plot_statement", settle)
    random.seed(99)
    moved = {"random": capture_rng_state()["random"]}
    random.setstate(before)

    done = statement_processor.settle_served_plot_run(
        [("x = 1", 0, moved, frozenset({"random"})), ("y = 2", 0, None, frozenset())]
    )

    assert done is False
    assert calls == ["x = 1", "y = 2"]
    assert capture_rng_state()["random"] == before, "the first statement's stream was left where its run left it"
    assert statement_processor.tracking_state.variable_lineage.get(var) == "seeded-above"


def test_a_run_that_settles_moves_the_streams(statement_processor):
    random.seed(11)
    random.seed(99)
    moved = {"random": capture_rng_state()["random"]}
    random.seed(11)

    assert statement_processor.settle_served_plot_run([("x = 1", 0, moved, frozenset())]) is True
    assert capture_rng_state()["random"] == moved["random"]
