"""Many small sets are recorded and laid out quickly, in the stored order.

Each set's elements were sorted by hash with radix tables of 2 MiB, zeroed
for every set: 200 us a set, 2.4 s for the 11,264 small frozensets a
Held-Karp memo of 12 cities keys on, and a hit slower than the run. A set
of up to 64 elements is sorted in place instead, into the same order.
"""

from __future__ import annotations

import pickle
import random
import time

import pytest

from cash.backends import set_order

pytestmark = pytest.mark.skipif(set_order._KERNELS is None, reason="no C extension for this Python")


def _small_sets(count: int) -> list[frozenset]:
    r = random.Random(5)
    points = [complex(r.randint(0, 300), r.randint(0, 200)) for _ in range(16)]
    return [frozenset(r.sample(points, r.randint(2, 14))) for _ in range(count)]


def test_sets_on_both_sides_of_the_in_place_sort_come_back_in_the_stored_order():
    r = random.Random(11)
    value = [set(r.sample(range(1 << 40), n)) for n in (2, 3, 8, 40, 63, 64, 65, 66, 200, 1000)]
    for s in value:  # dummies in the table too
        s.discard(next(iter(s)))
    metadata: dict = {}
    set_order.record(value, metadata)
    back = pickle.loads(pickle.dumps(value))
    set_order.restore(back, metadata)
    assert back == value
    assert [list(s) for s in back] == [list(s) for s in value]


def test_twenty_thousand_small_frozensets_keep_their_order():
    value = {"keys": [(1j, s, 2j) for s in _small_sets(20_000)]}
    metadata: dict = {}
    set_order.record(value, metadata)
    back = pickle.loads(pickle.dumps(value))
    set_order.restore(back, metadata)
    assert [list(k[1]) for k in back["keys"]] == [list(k[1]) for k in value["keys"]]


@pytest.mark.perf
def test_twenty_thousand_small_frozensets_are_recorded_and_laid_out_within_a_second():
    value = {"keys": [(1j, s, 2j) for s in _small_sets(20_000)]}
    metadata: dict = {}
    start = time.perf_counter()
    set_order.record(value, metadata)
    recorded = time.perf_counter() - start
    back = pickle.loads(pickle.dumps(value))
    start = time.perf_counter()
    set_order.restore(back, metadata)
    laid_out = time.perf_counter() - start
    # About 4 s each way with the radix tables, 10 to 40 ms without.
    assert recorded + laid_out < 1.0, (recorded, laid_out)
