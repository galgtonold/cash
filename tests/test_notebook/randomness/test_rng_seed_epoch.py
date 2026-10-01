"""re-seeding must invalidate the draws that follow it.

`np.random.seed(0)` -> `seed(1)` left every downstream draw unchanged: the draw
has a stable source and no tracked inputs, so its cache key never moved and cash
replayed the previous seed's numbers. The docs recommend seeding as *the* fix
for reproducibility, so following the advice produced provably wrong values.

The repair has two halves, and the first alone does nothing:

1. **Key the draw on a seed EPOCH** -- the seeding statement's own cache key.
2. **Stop replaying stale RNG state.** Restoring a cached statement replays its
   post-execution RNG state; after a re-seed that rewinds the generator to the
   cold run's state, so the draw recomputes (its key changed) and still yields
   the old seed's numbers because it draws from the old seed's state.

That second half is why keying on the LIVE RNG state cannot work: the live state
is a function of the cache, not of the current seed.
"""

from __future__ import annotations

from cash.notebook.statement.restore import rng_replay_is_current
from cash.tracking.randomness import (
    get_drawing_rng_modules,
    get_seeding_rng_modules,
)


class TestModuleDetection:
    """Seeding and drawing must resolve to the SAME module name or nothing matches."""

    def test_numpy_seed_and_draw_agree(self):
        assert get_seeding_rng_modules("np.random.seed(0)") == {"numpy.random"}
        assert get_drawing_rng_modules("a = np.random.rand(3)") == {"numpy.random"}

    def test_stdlib_seed_and_draw_agree(self):
        assert get_seeding_rng_modules("random.seed(0)") == {"random"}
        assert get_drawing_rng_modules("x = random.random()") == {"random"}

    def test_a_seed_is_not_a_draw(self):
        """Only a draw's result depends on the state, so only a draw is keyed."""
        assert get_drawing_rng_modules("np.random.seed(0)") == set()

    def test_a_draw_is_not_a_seed(self):
        assert get_seeding_rng_modules("a = np.random.rand(3)") == set()

    def test_syntax_error_is_not_fatal(self):
        assert get_seeding_rng_modules("def (") == set()
        assert get_drawing_rng_modules("def (") == set()


class TestRngReplayGate:
    """The second half: a stale RNG state must not clobber a fresh seed."""

    def test_matching_epoch_still_replays(self):
        """Within one seeding regime, replay is what keeps the stream coherent."""
        payload = {"rng_epochs": {"numpy.random": "stmt:aaa"}}
        assert rng_replay_is_current(payload, {"numpy.random": "stmt:aaa"}) is True

    def test_changed_epoch_suppresses_replay(self):
        """The regression: after a re-seed, replay would discard the new seed."""
        payload = {"rng_epochs": {"numpy.random": "stmt:aaa"}}
        assert rng_replay_is_current(payload, {"numpy.random": "stmt:bbb"}) is False

    def test_unknown_module_does_not_suppress(self):
        """No epoch for a module means nothing is known to have changed."""
        assert rng_replay_is_current({"rng_epochs": {"numpy.random": "stmt:aaa"}}, {}) is True

    def test_any_changed_module_suppresses(self):
        payload = {"rng_epochs": {"numpy.random": "stmt:aaa", "random": "stmt:zzz"}}
        assert rng_replay_is_current(payload, {"numpy.random": "stmt:aaa", "random": "stmt:CHANGED"}) is False


def test_a_hit_is_judged_by_the_seed_in_force_when_it_is_served(statement_processor):
    """A seed processed AFTER an entry was cached stops that entry's RNG state
    from replaying: the hit is judged by the seed ledger as it is when the hit
    is served. Judged by an older copy, the hit rewinds the stream to where the
    old seed left it, and the next draw repeats the old seed's numbers."""
    import random

    from cash.notebook.cache_status import CacheStatus

    processor = statement_processor
    processor.cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    processor.process_statement("import random")
    processor.process_statement("random.seed(0)")
    assert processor.process_statement("y = 1 + 1")["status"] == CacheStatus.COMPUTED

    processor.process_statement("random.seed(1)")
    assert processor.process_statement("y = 1 + 1")["status"] == CacheStatus.RESTORED

    drawn = random.random()
    random.seed(1)
    assert drawn == random.random()
