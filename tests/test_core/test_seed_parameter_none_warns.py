"""A seed PARAMETER that is None in this call is unseeded, and says so.

CAS-116, round-17 tester r17s4 (F4):

    @cash.cache
    def simulate(params, seed=None):
        rng = np.random.default_rng(seed)
        ...
    reps = [simulate(p) for _ in range(4)]   # 4 identical values, std exactly 0

Freezing an unseeded draw is the documented contract -- the first call's
value is what later calls get -- but the contract includes RANDOM-UNSEEDED
saying so. The detector read `default_rng(seed)` as seeded because it has an
argument; whether it is depends on what the caller passed, which only the
call knows.
"""
from __future__ import annotations

import warnings

import pytest

from cash import Cash

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core


def _decorate(tmp_path, **opts):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @c.cache(**opts)
    def simulate(n, seed=None):
        rng = np.random.default_rng(seed)
        return float(rng.normal(size=n).mean())

    return simulate


def _unseeded(rec):
    return [w for w in rec if "RANDOM-UNSEEDED" in str(w.message)]


def test_a_seed_left_at_its_none_default_warns(tmp_path):
    """THE BUG: silence, while the replicates were identical."""
    simulate = _decorate(tmp_path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        reps = [simulate(100) for _ in range(3)]

    assert len(set(reps)) == 1, "the contract: an unseeded draw is frozen"
    found = _unseeded(rec)
    assert found, "an unseeded draw was frozen without a word"
    assert "'seed'" in str(found[0].message)


def test_none_passed_explicitly_warns_too(tmp_path):
    simulate = _decorate(tmp_path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        simulate(100, seed=None)
    assert _unseeded(rec)


def test_a_real_seed_is_silent(tmp_path):
    """The control: seeded is seeded, and must not be flagged."""
    simulate = _decorate(tmp_path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        simulate(100, seed=1)
        simulate(100, seed=2)
    assert not _unseeded(rec)


def test_allow_random_silences_it(tmp_path):
    """The opt-out the fix line names still works."""
    simulate = _decorate(tmp_path, allow_random=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        simulate(100)
    assert not _unseeded(rec)
