"""Without the C kernels the stream walk finds what it finds with them.

The pure walk tells the objects it met by address over 16, the classes
and functions of a level by the objects themselves, and splits a level by
type in a pass per type. A stream is found however often an object, its
class or a function is met, behind a cycle, or where two parts share it.
"""

from __future__ import annotations

import itertools
import types

import pytest

from cash import _native, value_watch
from cash.notebook import call_key


class Holder:
    def __init__(self, i):
        self.i = i
        self.tags = ["a"]


def _closure(value):
    def fn():
        return value

    return fn


def _shared_then_stream():
    shared = [Holder(i) for i in range(50)]
    it = iter(range(3))
    tail = Holder(0)
    tail.it = it
    return [shared, shared, [*shared, tail]]


def _cycle_with_stream():
    a = {"name": "a"}
    b = {"next": a, "it": iter("xy")}
    a["next"] = b
    return [a, a, b]


def _self_closure():
    def fn():
        return fn

    return fn


def _class_with_stream():
    kind = type("WithIt", (), {"it": iter(range(2))})
    return [kind() for _ in range(30)]


def _fn_holding_fn():
    inner = _closure(iter(range(2)))
    return [inner, inner, types.SimpleNamespace(f=inner)]


SHAPES = {
    "clean_objects": lambda: [Holder(i) for i in range(2_000)],
    "clean_records": lambda: [{"id": i, "tags": ["a"]} for i in range(2_000)],
    "the_same_object_many_times": lambda: [Holder(0)] * 500,
    "one_class_many_objects_then_a_stream": lambda: [Holder(i) for i in range(500)] + [iter(())],
    "a_stream_behind_a_shared_list": _shared_then_stream,
    "a_stream_in_a_cycle": _cycle_with_stream,
    "a_stream_on_a_class": _class_with_stream,
    "a_stream_in_a_held_function": _fn_holding_fn,
    "mixed_types_and_a_stream": lambda: [1, "a", Holder(1), [2], {"k": (3,)}, frozenset({4}), itertools.count()],
    "mixed_types_clean": lambda: [1, "a", Holder(1), [2], {"k": (3,)}, frozenset({4})] * 40,
}


def _verdict(fn, monkeypatch, pure):
    with monkeypatch.context() as m:
        m.setattr(value_watch, "WATCHING", False)
        if pure:
            m.setattr(_native, "kernels", None)
        return call_key.holds_a_stream(fn)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_pure_walk_gives_the_kernels_verdict(shape, monkeypatch):
    fn = _closure(SHAPES[shape]())
    expected = "stream" in shape and "clean" not in shape
    assert _verdict(fn, monkeypatch, pure=True) is expected
    if _native.kernels is not None:
        assert _verdict(fn, monkeypatch, pure=False) is expected


def test_a_function_in_its_own_closure_is_walked_once(monkeypatch):
    assert _verdict(_self_closure(), monkeypatch, pure=True) is False


def test_each_object_is_met_once_and_a_repeat_is_dropped():
    first, second = Holder(1), Holder(2)
    seen: set[int] = set()
    assert call_key._new_in([first, second, first], seen) == [first, second]
    third = Holder(3)
    assert call_key._new_in([second, third], seen) == [third]
    assert call_key._new_in([Holder, Holder, Holder], seen) == [Holder]
    assert call_key._new_in([Holder], seen) == []


class _Shell:
    """The IPython shell's stand-in: it holds the whole session, past the
    walk's bounds."""

    def __init__(self):
        self.user_ns = {f"v{i}": (lambda i=i: i) for i in range(12_000)}


@pytest.mark.parametrize("pure", [True, False], ids=["pure", "kernels"])
def test_the_shell_is_left_out_with_and_without_the_kernels(monkeypatch, pure):
    """``def f(x, ip=get_ipython())`` holds no stream. Without the kernels
    the walk met objects by address over 16 and skipped the shell by its
    address: never matched, it was walked past the bounds, and a call
    reaching it was never stored."""
    if not pure and _native.kernels is None:
        pytest.skip("the C kernels are not built")
    shell = _Shell()
    monkeypatch.setattr(value_watch, "shells", lambda: [shell])

    def f(x, ip=shell):
        return x

    def mk(sh, it):
        def h():
            return sh, next(it)

        return h

    assert _verdict(f, monkeypatch, pure) is False
    assert _verdict(mk(shell, itertools.count()), monkeypatch, pure) is True
