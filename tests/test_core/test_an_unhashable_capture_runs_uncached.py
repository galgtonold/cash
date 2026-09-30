"""A value a closure reads that cannot be hashed runs the call uncached.

A cached function built by a factory, or a helper it calls, reads a
captured object -- a config holding a lock, a client holding a socket. Its
value could not be hashed, and it was left out of the key without a word:
changing the setting it carried served the old result.
"""

from __future__ import annotations

import threading
import time
import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


class _Settings:
    def __init__(self, weight):
        self.weight = weight
        self.lock = threading.Lock()  # what makes it unhashable


def _cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)


def _make_scorer(c, settings, runs):
    @c.cache
    def score(v):
        runs.append(1)
        time.sleep(0.12)
        return v * settings.weight

    return score


def _make_weigher(settings):
    def weigh(v):
        return v * settings.weight

    return weigh


WEIGH = _make_weigher(_Settings(2))


def _codes(record):
    return {str(w.message).split("]", 1)[0].lstrip("[") for w in record}


def test_a_cached_closure_over_an_unhashable_value_is_not_served_stale(tmp_path):
    settings = _Settings(2)
    runs: list = []
    score = _make_scorer(_cash(tmp_path), settings, runs)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert score(1) == 2
        settings.weight = 3
        assert score(1) == 3, "the old weight's result was served"
    assert "KEY-UNHASHABLE-CAPTURE" in _codes(record)


def test_a_helper_closure_over_an_unhashable_value_is_not_served_stale(tmp_path):
    global WEIGH
    c = _cash(tmp_path)

    @c.cache
    def use(v):
        time.sleep(0.12)
        return WEIGH(v)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert use(1) == 2
        WEIGH = _make_weigher(_Settings(5))
        try:
            assert use(1) == 5, "the old helper's result was served"
        finally:
            WEIGH = _make_weigher(_Settings(2))
    assert "KEY-UNHASHABLE-CAPTURE" in _codes(record)


class _Plain:
    def __init__(self, weight):
        self.weight = weight


def test_a_closure_over_a_hashable_value_still_caches(tmp_path):
    """Control: the same factory over a value that hashes is cached."""
    runs: list = []
    score = _make_scorer(_cash(tmp_path), _Plain(2), runs)
    score(1)
    score(1)
    assert len(runs) == 1
