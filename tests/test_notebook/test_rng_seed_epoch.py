"""CAS-223: re-seeding must invalidate the draws that follow it.

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

import pytest

from cash.notebook.randomness import (
    get_drawing_rng_modules,
    get_seeding_rng_modules,
)
from cash.notebook.statement.restore import StatementRestorer


class TestModuleDetection:
    """Seeding and drawing must resolve to the SAME module name or nothing matches."""

    def test_numpy_seed_and_draw_agree(self):
        assert get_seeding_rng_modules("np.random.seed(0)") == {'numpy.random'}
        assert get_drawing_rng_modules("a = np.random.rand(3)") == {'numpy.random'}

    def test_stdlib_seed_and_draw_agree(self):
        assert get_seeding_rng_modules("random.seed(0)") == {'random'}
        assert get_drawing_rng_modules("x = random.random()") == {'random'}

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

    def _restorer(self, epochs):
        return StatementRestorer(
            shell=object(), file_deps=object(), rng_seed_epochs=epochs,
        )

    def test_matching_epoch_still_replays(self):
        """Within one seeding regime, replay is what keeps the stream coherent."""
        r = self._restorer({'numpy.random': 'stmt:aaa'})
        assert r._rng_replay_is_current({'rng_epochs': {'numpy.random': 'stmt:aaa'}}) is True

    def test_changed_epoch_suppresses_replay(self):
        """The regression: after a re-seed, replay would discard the new seed."""
        r = self._restorer({'numpy.random': 'stmt:bbb'})
        assert r._rng_replay_is_current({'rng_epochs': {'numpy.random': 'stmt:aaa'}}) is False

    def test_legacy_entry_without_epochs_still_replays(self):
        """Entries cached before CAS-223 keep their previous behaviour."""
        r = self._restorer({'numpy.random': 'stmt:bbb'})
        assert r._rng_replay_is_current({'rng_state': {'x': 1}}) is True

    def test_unknown_module_does_not_suppress(self):
        """No epoch for a module means nothing is known to have changed."""
        r = self._restorer({})
        assert r._rng_replay_is_current({'rng_epochs': {'numpy.random': 'stmt:aaa'}}) is True

    def test_any_changed_module_suppresses(self):
        r = self._restorer({'numpy.random': 'stmt:aaa', 'random': 'stmt:CHANGED'})
        payload = {'rng_epochs': {'numpy.random': 'stmt:aaa', 'random': 'stmt:zzz'}}
        assert r._rng_replay_is_current(payload) is False


def test_default_ledger_is_isolated():
    """Two restorers built without a ledger must not share one dict."""
    a = StatementRestorer(shell=object(), file_deps=object())
    b = StatementRestorer(shell=object(), file_deps=object())
    a._rng_seed_epochs['numpy.random'] = 'stmt:aaa'
    assert b._rng_seed_epochs == {}


@pytest.mark.parametrize('seeded_first', [True, False])
def test_epoch_ledger_is_shared_with_the_processor(seeded_first):
    """The restorer must observe seeds executed AFTER it was constructed.

    The ledger is passed by reference precisely so a seed statement processed
    later is visible at restore time; copying it would silently reinstate the
    bug for every statement cached before the re-seed.
    """
    ledger: dict[str, str] = {}
    r = StatementRestorer(shell=object(), file_deps=object(), rng_seed_epochs=ledger)
    if seeded_first:
        ledger['numpy.random'] = 'stmt:aaa'
    ledger['numpy.random'] = 'stmt:bbb'
    assert r._rng_replay_is_current({'rng_epochs': {'numpy.random': 'stmt:aaa'}}) is False
