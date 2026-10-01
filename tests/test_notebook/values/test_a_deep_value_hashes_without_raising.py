"""A value nested deeper than pickle follows is hashed by identity, never raises.

The notebook hashes every value it keys or checks for change. Its value hashes
are documented never to fail: a value with no content hash falls back to its
identity, and `mutation_fingerprint` answers None for a value it cannot observe.
A linked list of a few hundred objects, or a deeply nested list, made pickle
raise RecursionError, which escaped the hash and ended the cell with an
internal error.
"""

from __future__ import annotations

import pytest

from cash.mutation_fingerprint import mutation_fingerprint
from cash.value_hash import compute_hash, is_identity_fallback_hash


class Node:
    def __init__(self, nxt):
        self.nxt = nxt


def _chain(n: int) -> Node | None:
    head = None
    for _ in range(n):
        head = Node(head)
    return head


def _nested(n: int) -> list:
    value: list = []
    for _ in range(n):
        value = [value]
    return value


DEEP = [pytest.param(_chain(3000), id="linked list"), pytest.param(_nested(5000), id="nested list")]


@pytest.mark.parametrize("value", DEEP)
def test_compute_hash_falls_back_to_identity(value):
    assert is_identity_fallback_hash(value, compute_hash(value))


@pytest.mark.parametrize("value", DEEP)
def test_mutation_fingerprint_reports_it_unobservable(value):
    assert mutation_fingerprint(value) is None
