"""What a helper's globals fold put into a key is remembered while nothing
it reads can have moved, and folded again as soon as something does.

A cached function whose argument the fast path cannot serve (an instance)
builds its key on every call, and the key folds the data globals of every
helper it reaches: tens of milliseconds a call for a library function with
a hundred helpers, for values that did not change. With the C extension,
the helper's world is watched the way the fast path watches a cached
function's, and the fold is reused while it holds. Anything the fold reads
moving -- a global rebound, a dict it holds changed in place, a module
attribute, a class attribute, a closure cell, the helper's code -- folds it
again, so the key moves exactly as it did without the memo.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import textwrap
import types

import pytest

from cash import Cash
from cash.decorator import helper_fold_memo, hit_memo, state_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the helper fold memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}

SCALE = 3
TABLE = {"k": 1}

settings = types.ModuleType("cash_test_remembered_fold_settings")
settings.RATE = 10
sys.modules[settings.__name__] = settings


class Weights:
    factor = 100


#: A class reached through data: only the helper's globals fold reads it.
KINDS = {"weights": Weights}


def _make_offset():
    offset = 1000

    def offset_of(x):
        return x + offset + SCALE * 0

    def set_offset(v):
        nonlocal offset
        offset = v

    return offset_of, set_offset


offset_of, set_offset = _make_offset()


def _helper(x):
    return x * SCALE + TABLE["k"] + settings.RATE + KINDS["weights"].factor


def _env_helper(x):
    return x + len(os.environ.get("CASH_TEST_REMEMBERED_FOLD", ""))


LIMIT = 5


def _clip(v):
    return min(v, LIMIT)


#: Functions held as data: the helper's fold keys what their code reads.
STEPS = {"clip": _clip}


def _held_helper(x, post=lambda v: v + SCALE):
    return post(STEPS["clip"](x))


def _scaled(x):
    return x


_scaled.k = 2


def _attr_helper(x):
    return x * _scaled.k


Encoder = json.JSONEncoder


def _library_class_helper(x):
    return x + len(Encoder.item_separator)


class Box:
    def __init__(self, v):
        self.v = v


def through_helper(box):
    RUNS["n"] += 1
    return _helper(box.v)


def through_held(box):
    RUNS["n"] += 1
    return _held_helper(box.v)


def through_attr(box):
    RUNS["n"] += 1
    return _attr_helper(box.v)


def through_library_class(box):
    RUNS["n"] += 1
    return _library_class_helper(box.v)


#: The user module the `lib` fixture writes and imports.
remembered_fold_lib = None


def _via_module(x):
    return remembered_fold_lib.cap(x)


def through_module(box):
    RUNS["n"] += 1
    return _via_module(box.v)


LIB = """
LIMITS = {"hi": 7}


def cap(x):
    return min(x, LIMITS["hi"])
"""


@pytest.fixture
def lib(tmp_path, monkeypatch):
    (tmp_path / "remembered_fold_lib.py").write_text(textwrap.dedent(LIB), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module("remembered_fold_lib")
    monkeypatch.setitem(globals(), "remembered_fold_lib", module)
    yield module
    sys.modules.pop("remembered_fold_lib", None)


def through_closure(box):
    RUNS["n"] += 1
    return offset_of(box.v)


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _warm(f, *args):
    """Call until the memo has the helper's fold; return the result."""
    for _ in range(3):
        result = f(*args)
    return result


def _hit(f, *args):
    runs = RUNS["n"]
    result = f(*args)
    assert RUNS["n"] == runs, "the call ran again"
    return result


def _ran(f, *args):
    runs = RUNS["n"]
    result = f(*args)
    assert RUNS["n"] == runs + 1, "a stale entry was served"
    return result


def test_the_fold_is_served_from_the_memo(c, monkeypatch):
    """With the function's whole state segment remembered
    (`state_memo`), the helper's fold is not asked for at all; this is
    about the call that still folds it."""
    monkeypatch.setattr(state_memo, "ENABLED", False)
    f = c.cache(through_helper)
    assert _warm(f, Box(1)) == 1 * 3 + 1 + 10 + 100
    before = helper_fold_memo.STATS["served"]
    _hit(f, Box(1))
    assert helper_fold_memo.STATS["served"] > before, "the helper's fold was not remembered"


def test_a_rebound_global_is_folded_again(c, monkeypatch):
    f = c.cache(through_helper)
    _warm(f, Box(1))
    monkeypatch.setitem(globals(), "SCALE", 4)
    assert _ran(f, Box(1)) == 1 * 4 + 1 + 10 + 100
    assert _hit(f, Box(1)) == 115


def test_a_dict_changed_in_place_is_folded_again(c, monkeypatch):
    f = c.cache(through_helper)
    _warm(f, Box(1))
    monkeypatch.setitem(TABLE, "k", 2)
    assert _ran(f, Box(1)) == 3 + 2 + 10 + 100


def test_a_module_attribute_is_folded_again(c, monkeypatch):
    f = c.cache(through_helper)
    _warm(f, Box(1))
    monkeypatch.setattr(settings, "RATE", 20)
    assert _ran(f, Box(1)) == 3 + 1 + 20 + 100


def test_a_class_attribute_is_folded_again(c, monkeypatch):
    f = c.cache(through_helper)
    _warm(f, Box(1))
    monkeypatch.setattr(Weights, "factor", 200)
    assert _ran(f, Box(1)) == 3 + 1 + 10 + 200


def test_a_closure_cell_is_folded_again(c):
    f = c.cache(through_closure)
    assert _warm(f, Box(1)) == 1001
    try:
        set_offset(2000)
        assert _ran(f, Box(1)) == 2001
    finally:
        set_offset(1000)


def test_the_helpers_code_is_folded_again(c, monkeypatch):
    f = c.cache(through_helper)
    _warm(f, Box(1))

    def replacement(x):
        return x * SCALE + TABLE["k"] + settings.RATE

    monkeypatch.setattr(_helper, "__code__", replacement.__code__)
    assert _ran(f, Box(1)) == 3 + 1 + 10


def test_an_environment_read_is_never_remembered(c, monkeypatch):
    def through_env(box):
        RUNS["n"] += 1
        return _env_helper(box.v)

    through_env.__module__ = __name__
    monkeypatch.setitem(globals(), "through_env", through_env)
    f = c.cache(through_env)
    monkeypatch.setenv("CASH_TEST_REMEMBERED_FOLD", "ab")
    assert _warm(f, Box(1)) == 3
    monkeypatch.setenv("CASH_TEST_REMEMBERED_FOLD", "abcd")
    assert _ran(f, Box(1)) == 5


def test_a_global_a_held_function_reads_is_folded_again(c, monkeypatch):
    f = c.cache(through_held)
    assert _warm(f, Box(9)) == 5 + 3
    monkeypatch.setitem(globals(), "LIMIT", 6)
    assert _ran(f, Box(9)) == 6 + 3


def test_a_global_a_default_function_reads_is_folded_again(c, monkeypatch):
    f = c.cache(through_held)
    assert _warm(f, Box(9)) == 5 + 3
    monkeypatch.setitem(globals(), "SCALE", 4)
    assert _ran(f, Box(9)) == 5 + 4


def test_a_global_a_module_function_reads_is_folded_again(c, lib, monkeypatch):
    f = c.cache(through_module)
    assert _warm(f, Box(9)) == 7
    monkeypatch.setitem(lib.LIMITS, "hi", 8)
    assert _ran(f, Box(9)) == 8


def test_an_attribute_stored_on_a_function_is_folded_again(c, monkeypatch):
    f = c.cache(through_attr)
    assert _warm(f, Box(3)) == 6
    monkeypatch.setattr(_scaled, "k", 5)
    assert _ran(f, Box(3)) == 15


def test_a_library_class_attribute_is_folded_again(c, monkeypatch):
    f = c.cache(through_library_class)
    assert _warm(f, Box(3)) == 3 + len(json.JSONEncoder.item_separator)
    monkeypatch.setattr(json.JSONEncoder, "item_separator", ";;;;")
    assert _ran(f, Box(3)) == 7
