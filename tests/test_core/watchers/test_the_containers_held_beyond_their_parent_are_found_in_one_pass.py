"""The containers held beyond their parent are found in one pass at C speed.

After a loop parsed 75,000 log rows into 1.7M ``(action, time)`` pairs, the
share check asked which containers in the list something besides their
parent holds (`held_beyond_parents`): the last row's list, still named by
the loop's variable. The look had walked the list at C speed, and the
answer walked it again, a level at a time in Python: 0.45 s of each run
of the cell. One C pass now answers it.

The answer names exactly the containers whose reference count says
something besides their parent holds them, or their parent more than once,
less the tuples holding leaves alone. The Python walk, the fallback, may
name more (a container below one that is held twice), never fewer.
"""

from __future__ import annotations

import datetime
import random
import sys

import pytest

from cash import _native
from cash import _plain_data as pd
from cash.value_types import LEAF_TYPES

pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")

NODES = (dict, list, tuple)


def _value(rng: random.Random, depth: int, pool: list):
    if pool and rng.random() < 0.1:
        return rng.choice(pool)  # held twice in the tree, or by the pool too
    kind = rng.random()
    if depth == 0 or kind < 0.3:
        return rng.choice([1, "s", 2.5, None, True, datetime.datetime(2020, 1, 1), ()])
    n = rng.randint(0, 4)
    if kind < 0.5:
        made = [_value(rng, depth - 1, pool) for _ in range(n)]
    elif kind < 0.75:
        made = tuple(_value(rng, depth - 1, pool) for _ in range(n))
    else:
        made = {f"k{i}": _value(rng, depth - 1, pool) for i in range(n)}
    if rng.random() < 0.2:
        pool.append(made)
    return made


def _held_beyond(value) -> set[int]:
    """The ids of the containers below *value* held by more than one
    reference, read off their counts: each is held here by the dict, and
    by the call's argument."""
    below: dict[int, object] = {}
    stack = [value]
    while stack:
        node = stack.pop()
        for item in node.values() if type(node) is dict else node:
            if type(item) in NODES and id(item) not in below:
                below[id(item)] = item
                stack.append(item)
    del stack, node, item
    out = set()
    for key in list(below):
        if sys.getrefcount(below[key]) > 3:
            held = below[key]
            if not (type(held) is tuple and all(type(x) in LEAF_TYPES for x in held)):
                out.add(key)
            del held
    return out


def _ids(found):
    return None if found is None else {id(x) for x in found}


@pytest.mark.parametrize("seed", range(200))
def test_the_answer_names_what_the_counts_say(seed, monkeypatch):
    rng = random.Random(9000 + seed)
    pool: list = []
    value = [_value(rng, rng.randint(1, 5), pool) for _ in range(rng.randint(1, 6))]
    held_by_name = value[0]  # a name holding a row, as a loop's variable does
    expected = _held_beyond(value)

    c = _ids(pd.held_beyond_parents(value, LEAF_TYPES, 10_000))
    monkeypatch.setattr(pd, "_K", None)
    python = _ids(pd.held_beyond_parents(value, LEAF_TYPES, 10_000))

    assert c == expected
    assert python >= expected
    del held_by_name, pool


def test_more_than_the_limit_is_no_answer():
    rows = [[i] for i in range(10)]
    names = rows[:3]
    assert pd.held_beyond_parents(rows, LEAF_TYPES, 2) is None
    assert len(pd.held_beyond_parents(rows, LEAF_TYPES, 3)) == 3
    del names


def test_data_that_is_no_tree_is_no_answer():
    assert pd.held_beyond_parents([[1], [object()]], LEAF_TYPES, 10) is None
    assert pd.held_beyond_parents([{(1,): 1}], LEAF_TYPES, 10) is None


def test_parsed_rows_are_not_walked_again_in_python(monkeypatch):
    when = datetime.datetime(2020, 1, 1)
    data = [[uid, [(f"a{k}", when) for k in range(5)]] for uid in range(2000)]
    actions = data[-1][1]

    def walked(*args, **kwargs):
        raise AssertionError("walked a level at a time in Python")

    monkeypatch.setattr(pd, "tree_levels", walked)
    found = pd.held_beyond_parents(data, LEAF_TYPES, 10)
    assert found is not None
    assert [x is actions for x in found] == [True]
