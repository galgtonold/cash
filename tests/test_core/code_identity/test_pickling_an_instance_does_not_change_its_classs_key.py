"""Pickling an instance does not change the key of its class.

The first pickle of an instance (protocol 2 and up) caches the class's slot
names on the class itself, as ``__slotnames__`` (`copyreg._slotnames`). The
class's code surface folded that attribute in, so a call keyed before any
instance was pickled and the same call keyed after had different keys: a
later process that had not pickled one yet -- its value restored from disk
straight to the caller -- missed an entry it should have hit.
"""

from __future__ import annotations

import pickle

from cash import Cash
from cash.backends import InMemoryBackend


class _Model:
    def __init__(self, n: int) -> None:
        self.weights = list(range(n))


def _score(cache: Cash, runs: list):
    @cache.cache
    def score(model):
        runs.append(1)
        return sum(model.weights)

    return score


def test_a_call_keyed_before_and_after_a_pickle_of_its_argument_hits():
    """Two `Cash` instances over one backend, as two processes over one
    disk: the first keys the call before any instance was pickled."""
    backend, runs = InMemoryBackend(), []
    if "__slotnames__" in vars(_Model):
        del _Model.__slotnames__
    assert _score(Cash(backend=backend, register_magic=False), runs)(_Model(10)) == 45
    pickle.dumps(_Model(1), protocol=pickle.HIGHEST_PROTOCOL)
    assert "__slotnames__" in vars(_Model)
    assert _score(Cash(backend=backend, register_magic=False), runs)(_Model(10)) == 45
    assert len(runs) == 1
