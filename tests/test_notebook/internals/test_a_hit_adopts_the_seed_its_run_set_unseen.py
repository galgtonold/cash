"""A hit leaves a stream its run seeded unseen with the run's lineage.

E16's ``test_suite`` seeds ``random`` inside a function: the run's seed is
adopted before the next statement, from where the stream stands
(`adopt_unseen_seeds`). A hit put the stream back where the run left it,
but no seed was set, so nothing was adopted: the next draw keyed on
another lineage and every other benchmark ran again after a restart.
"""

from __future__ import annotations

import random

from cash.tracking.randomness.lineage import adopted_seed_lineage, rng_virtual_var, stream_position, watch_seeds

VAR = rng_virtual_var("random")


def _seed_unseen(n: int) -> None:
    random.seed(n)
    random.random()


def test_after_a_hit_the_stream_has_the_lineage_it_has_after_the_run(cash_magics):
    randomness = cash_magics._statement_processor._randomness
    lineage = cash_magics.tracking_state.variable_lineage
    watch_seeds()
    randomness.adopt_unseen_seeds()

    # The run: seeded inside a function, adopted before the next statement.
    _seed_unseen(7)
    after_run_state = random.getstate()
    randomness.adopt_unseen_seeds()
    after_run = lineage[VAR]
    assert after_run == adopted_seed_lineage("random", stream_position("random"))

    # Elsewhere the stream moves on and its lineage with it.
    random.seed(99)
    randomness.adopt_unseen_seeds()
    assert lineage[VAR] != after_run

    # The hit: the stream put back where the run left it, no seed set.
    random.setstate(after_run_state)
    randomness.adopt_seeds_of_a_hit(["random"], "suite = test_suite(5)")
    randomness.adopt_unseen_seeds()
    assert lineage[VAR] == after_run


def test_a_seed_the_statement_spells_is_not_adopted(cash_magics):
    randomness = cash_magics._statement_processor._randomness
    lineage = cash_magics.tracking_state.variable_lineage
    watch_seeds()
    randomness.adopt_unseen_seeds()
    before = lineage.get(VAR)
    randomness.adopt_seeds_of_a_hit(["random"], "random.seed(3)")
    randomness.adopt_unseen_seeds()
    assert lineage.get(VAR) == before
