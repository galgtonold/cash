"""A class change after the fast hit path watched a call ends its fast hits.

A write to a base's dict, a middle class rebased or given an attribute move
a watched dict or type. An ABC registering a class (``isinstance``
changes) or a class defined below one (``__subclasses__`` changes) moves
neither: the memo checks those classes on every hit (`hit_memo.family_print`).
"""

import abc

import pytest

from cash import Cash
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


class Shape(abc.ABC):
    pass


class Circle:
    pass


class Base:
    setting = "base"


class Mid(Base):
    pass


class Leaf(Mid):
    pass


class Other:
    setting = "other"


class Plugin:
    pass


def is_shape(x):
    RUNS["n"] += 1
    return (x, isinstance(Circle(), Shape))


def leaf_setting(x):
    RUNS["n"] += 1
    return (x, Leaf.setting)


def plugin_count(x):
    RUNS["n"] += 1
    return (x, len(Plugin.__subclasses__()))


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def served_fast(f, *args):
    before_hits, before_runs = hit_memo.status()["fast_hits"], RUNS["n"]
    result = f(*args)
    return result, hit_memo.status()["fast_hits"] > before_hits and RUNS["n"] == before_runs


def warm(f, *args):
    for _ in range(6):
        result, fast = served_fast(f, *args)
        if fast:
            return result
    raise AssertionError(f"{f.__name__}{args} was never served from the memo")


def test_abc_register(c):
    f = c.cache(is_shape)
    assert warm(f, 1) == (1, False)
    Shape.register(Circle)
    assert f(1) == (1, True)


def test_base_dict_written_after_the_leaf_was_watched(c):
    f = c.cache(leaf_setting)
    assert warm(f, 1) == (1, "base")
    Base.setting = "changed"
    try:
        assert f(1) == (1, "changed")
    finally:
        Base.setting = "base"


def test_mid_rebased_after_the_leaf_was_watched(c):
    f = c.cache(leaf_setting)
    assert warm(f, 1)[1] == "base"
    Mid.__bases__ = (Other,)
    try:
        assert f(1) == (1, "other")
    finally:
        Mid.__bases__ = (Base,)


def test_setting_added_on_mid(c):
    f = c.cache(leaf_setting)
    assert warm(f, 1)[1] == "base"
    type.__setattr__(Mid, "setting", "mid")
    try:
        assert f(1) == (1, "mid")
    finally:
        del Mid.setting


def test_a_new_subclass(c):
    f = c.cache(plugin_count)
    n = warm(f, 1)[1]

    class P(Plugin):
        pass

    assert f(1) == (1, n + 1)
