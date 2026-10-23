"""A list an object in the key also holds keys apart from an equal copy.

An object small enough to be pickled whole is left to the key pickle,
whose marks for an object met twice saw only that pickle: ``a`` beside
``Rec(a)`` -- in two arguments, or in one list or dict -- keyed like an
equal copy of ``a`` beside it, and after the shared call ``f([1, 2, 3],
r)`` was served its result. The key pickle now marks a container the rest
of the key holds as held there: the list the key walk met, or a plain
argument keyed by its own digest.
"""

from __future__ import annotations

import argparse

import pytest

from cash import Cash

pytestmark = pytest.mark.filterwarnings("ignore::cash.exceptions.CashWarning")


class Rec:
    def __init__(self, x):
        self.x = x


def _two(a, r):
    return (a, r), lambda args: args[0] is args[1].x


def _in_list(a, r):
    return ([a, r],), lambda args: args[0][0] is args[0][1].x


def _in_dict(a, r):
    return ({"a": a, "r": r},), lambda args: args[0]["a"] is args[0]["r"].x


def _many(a, r):
    rows = [Rec(i) for i in range(5000)]
    rows[0] = r
    return (a, rows), lambda args: args[0] is args[1][0].x


@pytest.mark.parametrize("backend", ["disk", "memory"])
@pytest.mark.parametrize("holder", [Rec, lambda a: argparse.Namespace(x=a)], ids=["an object", "a namespace"])
@pytest.mark.parametrize(
    "shape", [_two, _in_list, _in_dict, _many], ids=["two arguments", "one list", "one dict", "a list of records"]
)
def test_the_call_with_a_copy_is_not_served_the_shared_ones_result(tmp_path, backend, holder, shape):
    if backend == "memory":
        cash = Cash(backend="memory", register_magic=False)
    else:
        cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    a = [1, 2, 3]
    r = holder(a)
    shared, test = shape(a, r)
    apart, _ = shape([1, 2, 3], r)
    f = cash.cache(lambda *args: test(args))
    for _ in range(8):
        assert f(*shared) is True
    for _ in range(8):
        assert f(*apart) is False
    assert f(*shared) is True


@pytest.mark.parametrize("shape", [_two, _in_list], ids=["two arguments", "one list"])
def test_an_object_pickled_flat_marks_the_list_too(tmp_path, monkeypatch, shape):
    """Where the pickler would run out of C stack first (Windows on 3.10),
    a deep object is pickled as a flat list of steps (`kept_state.flatten`):
    a list held elsewhere in the key is kept a leaf there, so the key
    pickle marks it as held, as it does for an object pickled as it is."""
    from cash import kept_state

    monkeypatch.setattr(kept_state, "FLAT_FIRST", True)
    monkeypatch.setattr(kept_state, "_SAFE_DEPTH", 0)
    cash = Cash(backend="memory", register_magic=False)
    a = [1, 2, 3]
    r = Rec(a)
    shared, test = shape(a, r)
    apart, _ = shape([1, 2, 3], r)
    f = cash.cache(lambda *args: test(args))
    for _ in range(8):
        assert f(*shared) is True
    for _ in range(8):
        assert f(*apart) is False
    assert f(*shared) is True
