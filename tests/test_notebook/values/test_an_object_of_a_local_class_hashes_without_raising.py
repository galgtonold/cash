"""An instance of a class defined inside a function is hashed by identity.

Pickling it fails, and on Python 3.13 the failure is an ``AttributeError``
("Can't get local object"), which the hash's last fallback did not catch: the
notebook raised from inside cash instead of keying the value on its identity,
and `mutation_fingerprint` raised instead of answering None.
"""

from __future__ import annotations

from cash.mutation_fingerprint import mutation_fingerprint
from cash.value_hash import compute_hash, is_identity_fallback_hash


def _local_instance():
    class Local:
        def __init__(self):
            self.n = 1

    return Local()


def test_the_hash_falls_back_to_identity():
    value = _local_instance()
    assert is_identity_fallback_hash(value, compute_hash(value))


def test_the_fingerprint_says_it_cannot_observe_it():
    assert mutation_fingerprint(_local_instance()) is None
