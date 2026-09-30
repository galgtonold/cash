"""A seed the caller sets reaches the key of a cached function that draws.

``np.random.seed(seed)`` at module level, or in an outer function before it
calls a cached ``draw()``, never reached ``draw``'s key outside a notebook:
only the notebook published a seed ledger, so every seed was served the
first seed's draw, and the only message said the draw was unseeded.
"""

from __future__ import annotations

import random

import pytest

np = pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def _no_notebook_ledger():
    from cash.tracking.randomness import publish_seed_epochs

    publish_seed_epochs(None)
    yield
    publish_seed_epochs(None)


@pytest.fixture
def inst(tmp_path):
    from cash import Cash
    from cash.backends import FileBackend

    return Cash(backend=FileBackend(cache_dir=str(tmp_path / "c")), register_magic=False)


def _plain(seed):
    np.random.seed(seed)
    return float(np.random.rand(3).sum())


def test_a_seed_set_by_an_outer_function_reaches_the_inner_key(inst):
    @inst.cache(allow_random=True)
    def draw(n):
        return float(np.random.rand(n).sum())

    @inst.cache
    def sample(seed, n):
        np.random.seed(seed)
        return draw(n)

    for _ in range(2):
        assert [sample(s, 3) for s in (0, 1, 2)] == [_plain(s) for s in (0, 1, 2)]


def test_a_module_level_seed_reaches_the_key_and_each_draw_differs(inst):
    @inst.cache(allow_random=True)
    def draw(n):
        return float(np.random.rand(n).sum())

    for seed in (0, 1, 0, 1):
        np.random.seed(seed)
        got = (draw(3), draw(3))
        np.random.seed(seed)
        assert got == (float(np.random.rand(3).sum()), float(np.random.rand(3).sum()))


def test_a_seed_of_the_random_module_reaches_the_key(inst):
    @inst.cache(allow_random=True)
    def pick():
        return random.random()

    for seed in (0, 1, 0, 1):
        random.seed(seed)
        got = pick()
        random.seed(seed)
        assert got == random.random()


def test_a_hit_under_a_seed_is_served_from_the_cache(inst):
    calls = []

    @inst.cache(allow_random=True)
    def draw(n):
        calls.append(n)
        return float(np.random.rand(n).sum())

    for _ in range(3):
        np.random.seed(7)
        draw(3)
    # One call reveals the draw (stored under no key), one stores, one hits.
    assert len(calls) == 2


def test_an_unseeded_draw_is_still_frozen(inst):
    """Control: with no seed the first value is replayed, as documented."""

    @inst.cache(allow_random=True)
    def draw(n):
        return float(np.random.rand(n).sum())

    np.random.seed(None)
    first = draw(3)
    assert draw(3) == first
    assert draw(3) == first


def test_the_warning_says_a_callers_seed_is_keyed(inst):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @inst.cache
        def draw(n):
            return float(np.random.rand(n).sum())

    text = " ".join(str(w.message) for w in caught)
    assert "RANDOM-UNSEEDED" in text or "Unseeded randomness" in text
    assert "np.random.seed() after this decoration" in text
