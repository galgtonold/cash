"""A set of ids or names keys at about the cost of a list of them, and as
it always did: by its items whatever order they went in, plus the order a
set of numbers iterates in.

To give a set one order its items were each pickled on their own, through
the Python-level pickler, on every key build, hits included: 10-15 us an
item, a second for 100,000 ids. A set of strings, of ints, of bytes, of
floats, or of tuples of those sorts by value. The work is counted here
(pickles of single items), not timed.
"""

from __future__ import annotations

import json
import random

import pytest

from cash import _plain_data
from cash.canonical_form import canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT
from tests._scripts import run_python
from tests._work_counts import method_calls

pytestmark = pytest.mark.core

N = 5_000


def _shuffled(items, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


SETS = {
    "ints": lambda seed: set(_shuffled(range(N), seed)),
    "strings": lambda seed: set(_shuffled((f"id-{i}" for i in range(N)), seed)),
    "bytes": lambda seed: set(_shuffled((b"k%d" % i for i in range(N)), seed)),
    "floats": lambda seed: set(_shuffled((i / 7 for i in range(N)), seed)),
    "pairs": lambda seed: set(_shuffled(((i % 50, f"n{i}") for i in range(N)), seed)),
    "frozen strings": lambda seed: frozenset(_shuffled((f"id-{i}" for i in range(N)), seed)),
}


def _key(value):
    return canonical_bytes(value, BUILTIN_CONTENT)


@pytest.mark.parametrize("kind", list(SETS))
def test_no_item_is_pickled_to_place_it(kind):
    value = SETS[kind](1)
    with method_calls(_plain_data, "content_dumps") as dumps:
        _key(value)
    # The form is written once; it was once more per item (5,000).
    assert dumps.calls <= 3


def test_other_items_are_still_placed_by_their_content():
    """The positive control of the count above, and the fallback: a set
    mixing kinds, or holding a NaN, goes item by item as before."""
    mixed = {"one", b"two", "three", b"four", ("a", b"b"), ("c",)}
    with method_calls(_plain_data, "content_dumps") as dumps:
        _key(mixed)
    assert dumps.calls >= len(mixed)
    assert _key(mixed) == _key(set(_shuffled(mixed, 3)))
    assert _key(mixed) != _key(mixed | {"five"})
    nan = float("nan")
    assert _key({1.5, nan}) == _key({nan, 1.5})
    assert _key({1.5, nan}) != _key({2.5, nan})


@pytest.mark.parametrize("kind", ["strings", "bytes", "frozen strings"])
def test_items_whose_order_the_process_picks_key_order_free(kind):
    assert _key(SETS[kind](1)) == _key(SETS[kind](2))


@pytest.mark.parametrize("kind", ["ints", "floats", "pairs"])
def test_equal_sets_of_numbers_built_alike_key_alike(kind):
    assert SETS[kind](1) == SETS[kind](2)
    assert _key(SETS[kind](1)) == _key(SETS[kind](1))


def test_the_order_a_set_of_numbers_iterates_in_stays_in_the_key():
    assert list({1, 9}) != list({9, 1})  # they collide in the table: first in, first out
    assert _key({1, 9}) != _key({9, 1})
    assert _key({(1, 0), (9, 0)}) == _key({(1, 0), (9, 0)})
    grown = set(range(64))
    for i in range(8, 64):
        grown.discard(i)
    assert grown == set(range(8))
    assert (_key(grown) == _key(set(range(8)))) == (list(grown) == list(range(8)))


def test_sets_differing_in_one_item_or_in_kind_key_apart():
    assert _key({1, 2, 3}) != _key({1, 2, 4})
    assert _key({"a", "b"}) != _key({"a", "c"})
    assert _key({1, 2}) != _key({1.0, 2.0})
    assert _key({"a"}) != _key({b"a"})
    assert _key({(1, "a")}) != _key({(1, "b")})
    assert _key({1, 2}) != _key(frozenset({1, 2}))
    assert _key({(1, 2)}) != _key({(1, 2, 3)})
    assert _key({("ab",)}) != _key({("a", "b")})


SCRIPT = """
import hashlib, json
from cash.canonical_form import canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT

values = {
    "ints": set(range(0, 3000, 7)) | {10**30, -5},
    "collided ints": {9, 1, 17},
    "strings": {f"name-{i}" for i in range(500)},
    "bytes": {b"k%d" % i for i in range(500)},
    "floats": {i / 3 for i in range(500)} | {float("inf"), -0.0},
    "pairs": {(i % 9, f"n{i}", i / 2) for i in range(500)},
    "frozen": frozenset(f"x{i}" for i in range(400)),
    "nested": {"keep": {3, 1, 2}, "names": [{"b", "a"}, frozenset({(1, 2), (0, 5)})]},
    "mixed": {1, "one", 2.5, None, (1, "a")},
}
print(json.dumps({k: hashlib.sha256(canonical_bytes(v, BUILTIN_CONTENT)).hexdigest() for k, v in values.items()}))
"""


def test_the_key_is_the_same_in_every_process_with_or_without_the_c_extension(tmp_path):
    def digests(**env):
        return json.loads(run_python("-c", SCRIPT, cwd=tmp_path, env=env).stdout.strip().splitlines()[-1])

    first = digests(PYTHONHASHSEED="0")
    assert len(first) == 9
    assert digests(PYTHONHASHSEED="12345") == first
    assert digests(PYTHONHASHSEED="7", CASH_PURE_PYTHON="1") == first
