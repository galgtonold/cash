"""A value nested deeper than pickle follows is hashed by identity, never raises.

The notebook hashes every value it keys or checks for change. Its value hashes
are documented never to fail: a value with no content hash falls back to its
identity, and `mutation_fingerprint` answers None for a value it cannot observe.
A linked list or nested list deeper than pickle follows makes pickle raise
RecursionError, which must not escape the hash and end the cell with an
internal error.

How deep pickle follows depends on the Python: to the recursion limit on
3.10 and 3.11, a few thousand levels of its own C limit on 3.12 and 3.13,
as far as the C stack allows on 3.14. A value pickle does follow is hashed
by its content, whole. The values here are deeper than any of them.
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


DEPTH = 100_000
DEEP = [pytest.param(_chain(DEPTH), id="linked list"), pytest.param(_nested(DEPTH), id="nested list")]


@pytest.mark.parametrize("value", DEEP)
def test_compute_hash_falls_back_to_identity(value):
    assert is_identity_fallback_hash(value, compute_hash(value))


@pytest.mark.parametrize("value", DEEP)
def test_mutation_fingerprint_reports_it_unobservable(value):
    assert mutation_fingerprint(value) is None
