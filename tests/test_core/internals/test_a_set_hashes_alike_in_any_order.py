"""A set's digest does not depend on the order it iterates in.

A set iterates in its hash table's slot order, which depends on the order
its elements went in and, for strings, on ``PYTHONHASHSEED``. Pickled, a set
is written in that order, so two equal sets digested apart: a frozenset a
``functools.cache`` memo answered with, put back from a pickle, held the
same elements in another order, read as changed, and the cell below it ran
again. The value hash (`compute_hash`, `unshared_hash` and the digest a
store's pickle gives) now writes a set's elements sorted by their content.
"""

from __future__ import annotations

import json
import pickle
import types

import pytest

from cash import _native, kept_state, stored_pickle
from cash.value_hash import compute_hash, is_identity_fallback_hash, unshared_hash
from tests._scripts import run_python

#: 1 and 2**61 have one hash (CPython hashes ints modulo 2**61 - 1), so the
#: one put in second takes another slot: the order is the insertion order.
A, B = 1, 2**61


def _two_orders(kind=frozenset):
    first, second = kind([A, B, 7]), kind([B, A, 7])
    assert first == second
    assert list(first) != list(second), "the two iterate in different orders"
    return first, second


class Holder:
    def __init__(self, items):
        self.items = items


def _wrapped(kind):
    first, second = _two_orders(kind)
    yield "alone", first, second
    yield "in a tuple", (1, first, "x"), (1, second, "x")
    yield "in a list", [first, [first]], [second, [second]]
    yield "a dict value", {"k": first}, {"k": second}
    yield "in an object", Holder(first), Holder(second)


@pytest.mark.parametrize("kind", [frozenset, set])
def test_equal_sets_in_two_orders_hash_alike(kind):
    for where, first, second in _wrapped(kind):
        assert compute_hash(first) == compute_hash(second), where
        assert unshared_hash(first) == unshared_hash(second), where


def test_a_frozenset_of_frozensets_in_two_orders_hashes_alike():
    one, other = _two_orders()
    first = frozenset([one, frozenset({"a", "b"}), frozenset({B, A})])
    second = frozenset([frozenset({"b", "a"}), frozenset({A, B}), other])
    assert first == second
    assert compute_hash(first) == compute_hash(second)
    assert compute_hash({(3,): first}) == compute_hash({(3,): second})


def test_a_set_put_back_from_its_pickle_hashes_as_before():
    """A memo's answer as a store gives it back: rebuilt by pickle."""
    complexes = frozenset(complex(i * 37 % 300, i * 11 % 200) for i in range(40))
    # Emptied down to three items, a set keeps its big table: 1 sorts before
    # 8 there, and after 8 in the small table a rebuild makes.
    shrunk = set(range(20))
    for x in range(20):
        if x not in (1, 8, 9):
            shrunk.discard(x)
    memo = {(12,): complexes, (5,): frozenset({A, B, 3}), (7,): shrunk}
    restored = pickle.loads(pickle.dumps(memo))
    assert restored == memo
    assert list(restored[(7,)]) != list(shrunk), "the rebuilt set iterates in another order"
    assert compute_hash(restored) == compute_hash(memo)
    assert unshared_hash(restored) == unshared_hash(memo)


def test_a_stores_pickle_digests_a_set_in_any_order():
    first, second = _two_orders()
    one = stored_pickle.pickle_value([first], keep=True)
    other = stored_pickle.pickle_value([second], keep=True)
    assert one.digest == other.digest == compute_hash([first])
    # The store keeps the set as pickle writes it, and gives it back equal.
    assert pickle.loads(one.stream, buffers=one.buffers) == [first]


def test_different_sets_still_hash_apart():
    digests = {
        compute_hash(frozenset({1, 2})),
        compute_hash(frozenset({1, 3})),
        compute_hash({1, 2}),
        compute_hash((1, 2)),
        compute_hash(frozenset({(1, 2), (2, 1)})),
        compute_hash(frozenset({(1, 2), (2, 2)})),
        compute_hash(frozenset({"1", "2"})),
        compute_hash(frozenset({1.0, 2.5})),
    }
    assert len(digests) == 8


def test_one_set_twice_hashes_apart_from_two_equal_sets():
    """Sharing still counts for a set that can change: ``[s, s]`` is not
    ``[s, set(s)]``, whose second can be changed alone."""
    shared = {A, B, 3}
    assert compute_hash([shared, shared]) != compute_hash([shared, set(shared)])
    assert compute_hash([shared, shared]) == compute_hash([set(shared)] * 2)


def test_a_set_holding_what_cannot_be_pickled_keeps_the_identity_hash():
    class Refuses:
        def __reduce__(self):
            raise TypeError("no")

    value = frozenset({Refuses(), Refuses(), 1})
    assert is_identity_fallback_hash(value, compute_hash(value))


def test_a_set_of_strings_hashes_alike_under_any_hash_seed(tmp_path):
    source = (
        "import json\n"
        "from cash.value_hash import compute_hash, unshared_hash\n"
        "words = frozenset(f'w{i}' for i in range(200))\n"
        "value = {'words': words, 'nested': (set(words), frozenset({words, frozenset({'a', 'b'})}))}\n"
        "print(json.dumps([list(words)[:5], compute_hash(value), unshared_hash(value)]))\n"
    )
    seen = []
    for seed in ("1", "2"):
        out = run_python("-c", source, cwd=tmp_path, env={"PYTHONHASHSEED": seed}).stdout
        seen.append(json.loads(out.strip().splitlines()[-1]))
    assert seen[0][0] != seen[1][0], "the seeds iterate the strings in different orders"
    assert seen[0][1:] == seen[1][1:]


@pytest.mark.skipif(getattr(_native.kernels, "sorted_set_id", None) is None, reason=f"no C hook: {_native.REASON}")
@pytest.mark.parametrize("digest", [compute_hash, unshared_hash])
def test_the_c_key_hooks_sort_sets_as_the_python_ones_do(monkeypatch, digest):
    first, second = _two_orders()
    shared = {A, B, 9}
    value = [first, shared, shared, Holder(frozenset({"x", "y", first})), {b"a", b"b"}, {"s", 1, 2.5, None}]
    other = [second, {B, A, 9}, {B, A, 9}, Holder(frozenset({"y", "x", second})), {b"b", b"a"}, {None, 2.5, 1, "s"}]
    fast = digest(value)
    with monkeypatch.context() as m:
        m.setattr(kept_state, "_native", types.SimpleNamespace(kernels=None))
        slow = digest(value)
    assert fast == slow
    assert digest(other) != fast, "one set held twice is not two equal sets"
    other[2] = other[1]
    assert digest(other) == fast


def test_a_value_without_sets_digests_as_before():
    """Only a set of two or more items is written apart: every other
    value's bytes, and so its digest, are what they were."""
    for value in ([1, "a", (2.5, None)], {"k": frozenset({1})}, [set()], {1: b"\x8f\x94\x91\x94"}):
        assert stored_pickle.object_digest_and_size(value)[0] == compute_hash(value)
        assert b"__cash_set__" not in kept_state.dumps(value, protocol=5, keyed=True)
