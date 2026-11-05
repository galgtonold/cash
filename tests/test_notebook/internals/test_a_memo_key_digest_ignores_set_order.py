"""A helper memo's key is digested the same whatever order a frozenset in
it iterates in.

A frozenset's order is its hash table's, and a pickle rebuilds it
otherwise. A statement that filled ``benchmark(tour_a, cities)`` was never
served after a restart without the C extension (which puts a restored
set's order back): the keys put back from the entry digested as other
keys. An equal key built in another order read as one the memo lacks.
"""

from __future__ import annotations

import pickle

from cash.notebook.helper_memos import _change_key_digest

# Five floats whose frozenset iterates otherwise once unpickled.
CITIES = frozenset([0.351846, 0.34347, 0.453671, 0.430186, 0.512444])


def tour_a(c):
    return sorted(c)


def test_a_frozenset_key_digests_alike_after_a_pickle():
    back = pickle.loads(pickle.dumps(CITIES))
    assert back == CITIES
    assert list(back) != list(CITIES)  # the positive control: another order
    assert _change_key_digest(back) == _change_key_digest(CITIES)
    key = (tour_a, (CITIES, frozenset({1, 2})))
    again = pickle.loads(pickle.dumps(key))
    assert _change_key_digest(again) == _change_key_digest(key)


def test_other_keys_digest_apart():
    assert _change_key_digest(CITIES) != _change_key_digest(CITIES | {0.5})
    assert _change_key_digest((tour_a, CITIES)) != _change_key_digest((sorted, CITIES))
    assert _change_key_digest(frozenset({1})) != _change_key_digest((1,))
    # Keys a memo takes as one still digest alike.
    assert _change_key_digest((1, frozenset({2}))) == _change_key_digest((1.0, frozenset({2.0})))
