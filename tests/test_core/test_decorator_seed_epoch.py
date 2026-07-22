"""A cached function that draws from the global RNG must see a seed change.

Change ``np.random.seed(12345)`` to ``seed(999)``, re-run, and the model
trained under the OLD seed came straight back -- silently, with a green
RESTORED badge -- because the decorator keys on ``(source + arguments)`` and
the global stream is neither. "Change the seed and re-run" is the canonical ML
action, and it happened on the exact idiom ``cash.help()`` rule 4 recommends.

Confirmed against a real Jupyter server before the fix:

    SEED=12345, fresh cache -> 0.94140000000000001
    SEED=999,   warm cache  -> 0.94140000000000001   <- the 12345 answer
    SEED=999,   fresh cache -> 0.94189999999999996   <- ground truth

The counterpart guard matters just as much: an UNSEEDED draw must still be
cached and replayed. Freezing an unseeded value is the documented contract and
the reason caching an expensive sample is worth anything.
"""
from __future__ import annotations

import pytest

import cash

np = pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def _isolated_epochs():
    """Point the shared seed ledger at a dict this test owns."""
    from cash.notebook.randomness import publish_seed_epochs
    epochs: dict[str, str] = {}
    publish_seed_epochs(epochs)
    yield epochs
    publish_seed_epochs(None)


def _drawing_fn():
    calls = []

    @cash.cache
    def draw(n):
        calls.append(n)
        return float(np.random.rand(n).sum())

    return draw, calls


def test_seed_change_invalidates_a_drawing_function(_isolated_epochs):
    draw, calls = _drawing_fn()

    np.random.seed(12345)
    _isolated_epochs["numpy.random"] = "epoch-of-seed-12345"
    first = draw(3)
    assert len(calls) == 1

    # Same seed, same epoch -> still a hit, and the frozen value is replayed.
    np.random.seed(12345)
    assert draw(3) == first
    assert len(calls) == 2, (
        "the call that revealed the draw stored an epoch-free key; exactly one "
        "extra recompute is expected before the key stabilises"
    )

    assert draw(3) == first
    assert len(calls) == 2, "key did not stabilise: every call is recomputing"

    # A DIFFERENT seed must not serve the previous seed's value.
    np.random.seed(999)
    _isolated_epochs["numpy.random"] = "epoch-of-seed-999"
    under_999 = draw(3)
    assert len(calls) == 3, "seed change did not invalidate the cached value"

    # And it must equal what the function genuinely computes under seed 999.
    np.random.seed(999)
    oracle = float(np.random.rand(3).sum())
    assert under_999 == pytest.approx(oracle), (
        "recomputed, but not with the new seed's stream"
    )


def test_unseeded_draw_is_still_cached_and_replayed(_isolated_epochs):
    """The freeze contract: no seed anywhere means the value is replayed.

    This is the over-invalidation guard. Folding RNG state (rather than the
    epoch) into the key would make an unseeded draw miss forever, which would
    defeat the point of caching an expensive sample.
    """
    draw, calls = _drawing_fn()

    first = draw(4)
    assert draw(4) == first
    assert draw(4) == first
    assert len(calls) <= 2, "an unseeded draw should settle into hits, not recompute"


def test_non_drawing_function_key_is_unchanged(_isolated_epochs):
    """Regression guard: a function that never draws must not gain key churn."""
    calls = []

    @cash.cache
    def pure(x):
        calls.append(x)
        return x * 2

    assert pure(21) == 42
    _isolated_epochs["numpy.random"] = "some-new-epoch"
    assert pure(21) == 42
    assert len(calls) == 1, "a seed change invalidated a function that never draws"


def test_epoch_component_is_empty_without_a_seed():
    """The fold must be inert until something is actually seeded."""
    from cash.notebook.randomness import publish_seed_epochs, seed_epoch_component

    publish_seed_epochs({})
    try:
        assert seed_epoch_component({"numpy.random"}) == ""
        publish_seed_epochs({"numpy.random": "abc"})
        assert seed_epoch_component({"numpy.random"}) == ":rng:numpy.random:abc"
        assert seed_epoch_component({"torch"}) == "", "unrelated module leaked in"
        assert seed_epoch_component(set()) == ""
    finally:
        publish_seed_epochs(None)


def test_no_entry_is_left_under_the_epoch_free_key(_isolated_epochs):
    """The call that DISCOVERS a draw must not leave a cacheable entry.

    This is the trap the fix closes, and it only shows up across a restart.
    The key is built before the draw is known, so the entry written on that
    first call carries no seed epoch. Every later run rebuilds exactly that
    key and matches it -- and a restart guarantees the rebuild, because the
    in-memory "this function draws" verdict is gone.

    Clearing ``_rng_drawing_funcs`` below is precisely what a kernel restart
    does. Without the fix the second phase HITS the epoch-free entry and
    returns the old seed's value.
    """
    inst = cash.cache.__self__
    draw, calls = _drawing_fn()

    np.random.seed(4242)
    _isolated_epochs["numpy.random"] = "epoch-A"
    draw(3)
    assert len(calls) >= 1

    # Simulate a restart: the observed-draw verdict is forgotten, the cache
    # is not. Then change the seed, as a user would.
    inst._rng_drawing_funcs.clear()
    _isolated_epochs["numpy.random"] = "epoch-B"
    np.random.seed(777)
    before = len(calls)
    under_b = draw(3)
    assert len(calls) > before, (
        "served an entry stored under the epoch-free key: the seed changed but "
        "the previous seed's value came back"
    )

    np.random.seed(777)
    assert under_b == pytest.approx(float(np.random.rand(3).sum())), (
        "recomputed, but not on the new seed's stream"
    )
