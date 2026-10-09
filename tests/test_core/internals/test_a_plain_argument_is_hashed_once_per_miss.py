"""A miss on plain data hashes its arguments once, for the key.

The check that the body left its arguments alone hashed them again after
the body: a second full hash of a list of ten million ints, a second of a
3.5 s first call whose work took 0.09 s. Plain lists and tuples are compared
by the identities of what they hold (`_plain_data.identity_snapshot`), and
their leaves cannot change in place, so when no identity moved and every
other argument cannot change at all, the second hash proves nothing more.
"""

from __future__ import annotations

import pytest

from cash.decorator.purity_checks import PurityChecks
from tests._work_counts import hashed_bytes, method_calls

pytestmark = pytest.mark.core


class Tagged(int):
    """An int that can carry attributes."""


@pytest.mark.parametrize(
    "rows",
    [
        list(range(200_000)),
        tuple(range(200_000)),
        [(i, str(i), float(i)) for i in range(50_000)],
        [[i, i + 1] for i in range(50_000)],
    ],
    ids=["list of ints", "tuple of ints", "list of tuples", "list of lists"],
)
def test_a_miss_on_plain_data_does_not_hash_it_again_after_the_body(disk_cash, rows):
    f = disk_cash.cache(lambda rows, scale=1: len(rows) * scale)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f(rows, scale=2)
    assert rehash.calls == 0


def test_a_miss_hashes_a_big_plain_list_about_once(disk_cash):
    rows = list(range(300_000))
    f = disk_cash.cache(lambda rows: len(rows))
    f([1])  # set-up work of a first call, not of this argument
    with hashed_bytes() as hashed:
        f(rows)
    # One hash of ~2.7 MB of pickled ints for the key; two was the old count.
    one = len(__import__("pickle").dumps(rows, protocol=5))
    assert hashed.bytes < 1.5 * one, (hashed.bytes, one)


@pytest.mark.parametrize(
    "arg",
    [{"a": [1, 2]}, bytearray(b"abc"), [bytearray(b"abc")], [{"a": 1}]],
    ids=["dict", "bytearray", "list of bytearrays", "list of dicts"],
)
def test_an_argument_identities_cannot_vouch_for_is_hashed_again(disk_cash, arg):
    """Control: a dict or a bytearray can change with every identity in place."""
    f = disk_cash.cache(lambda x: 1)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f(arg)
    assert rehash.calls == 1


def test_an_int_subclass_beside_a_plain_list_is_hashed_again(disk_cash):
    """Control: a subclass of int can carry state that changes."""
    f = disk_cash.cache(lambda rows, n: len(rows) + n)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f([1, 2, 3], Tagged(4))
    assert rehash.calls == 1
