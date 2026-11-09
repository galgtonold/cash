"""A set-of-numbers argument iterating in another order keys apart.

``{1, 9}`` and ``{9, 1}`` are equal, but iterate ``[1, 9]`` and ``[9, 1]``:
1 and 9 collide in a small set's table and sit in the order they went in.
A function returning ``list(s)`` or ``next(iter(s))`` was served the other
set's answer. A set of numbers iterates alike after the same history in
every process, so its order goes into the key where it is not the one a
set built from its items in sorted order has. Sets that iterate as such a
set does still share an entry, and a set of strings keys alike in any
order and any process (``test_set_arg_determinism``).
"""

from __future__ import annotations

from cash import kept_state


def test_a_set_of_another_order_is_called_again(cash_instance):
    @cash_instance.cache
    def first(s):
        return next(iter(s))

    a, b = {1, 9}, {9, 1}
    assert first(a) == 1
    assert first(b) == 9


def test_a_set_of_the_same_order_still_hits(cash_instance):
    calls = []

    @cash_instance.cache
    def ordered(s):
        calls.append(1)
        return list(s)

    assert ordered({1, 2, 3}) == [1, 2, 3]
    assert ordered({3, 2, 1}) == [1, 2, 3]
    assert len(calls) == 1


def test_a_set_inside_a_value_keys_by_its_order(cash_instance):
    @cash_instance.cache
    def head(d):
        return list(d["tags"])

    assert head({"tags": {1, 9}}) == [1, 9]
    assert head({"tags": {9, 1}}) == [9, 1]


def test_a_set_of_floats_or_tuples_keys_by_its_order(cash_instance):
    @cash_instance.cache
    def first(s):
        return next(iter(s))

    assert first({1.0, 9.0}) == 1.0
    assert first({9.0, 1.0}) == 9.0
    assert first({(1, 2**61), (2**61, 1)}) == next(iter({(1, 2**61), (2**61, 1)}))


def test_a_set_of_strings_still_keys_alike_in_two_orders(cash_instance):
    calls = []

    @cash_instance.cache
    def size(s):
        calls.append(1)
        return len(s)

    words = [f"w{i}" for i in range(50)]
    assert size(set(words)) == size(set(reversed(words))) == 50
    assert len(calls) == 1


def test_the_order_is_what_the_set_iterates():
    assert kept_state.set_walk_order([1, 9], (1, 9)) is None
    assert kept_state.set_walk_order(list({9, 1}), (1, 9)) == (1, 0)
    assert kept_state.set_walk_order(["b", "a"], ("a", "b")) is None, "strings key order-free"
    assert kept_state.set_walk_order([9.0, float("nan")], (9.0, float("nan"))) is None
