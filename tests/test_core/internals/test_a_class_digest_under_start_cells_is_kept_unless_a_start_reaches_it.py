"""A class digest taken while a call's start cells are set is remembered
when no function it goes through started with cells of its own.

While a miss's trace is built, the cells each function started with stand
in for its cells (`call_starts`), so a class digest there may be that
call's and not the class's; such digests were never served from the class
memo nor kept in it, and every miss digested each class the trace names
again: about 70 digests of an lmfit fit. A digest that goes through no
function whose start cells stand in is the class's own, and is now served
and kept; one that does is still taken anew. A start whose cells all hold
classes (a method calling ``super()``) is read as it is now, as with no
start.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash.decorator import class_memo
from cash.tracking import trace_items

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not class_memo.ENABLED, reason="the class memo needs the C extension's world watch"),
]

_MODULE = """
TABLE = [1, 2, 3]


def _make_hook(step):
    def hook(x):
        return x + step

    return hook


class Base:
    def run(self, x):
        return x


class Plain(Base):
    def run(self, x):
        return super().run(x) + TABLE[0]


class Hooked:
    hook = staticmethod(_make_hook(2))

    def run(self, x):
        return self.hook(x) * TABLE[1]


def other(x):
    def inner():
        return x

    return inner
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"trace_digest_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def digests(mod, monkeypatch):
    """The classes of *mod* digested anew, by name."""
    taken = []
    real = trace_items._class_digest_now

    def counting(cls, *args, **kwargs):
        if getattr(cls, "__module__", None) == mod.__name__:
            taken.append(cls.__name__)
        return real(cls, *args, **kwargs)

    monkeypatch.setattr(trace_items, "_class_digest_now", counting)
    return taken


def _starts(code, cells):
    return {code: trace_items._StartCells(cells, tuple(None for _ in cells))}


def test_a_digest_through_no_start_is_kept_while_starts_are_set(mod, digests):
    plain = trace_items.remembered_class_digest(mod.Plain)
    unrelated = mod.other(1).__code__
    with trace_items.call_starts(_starts(unrelated, ("x",))):
        assert trace_items.remembered_class_digest(mod.Plain) == plain
        assert trace_items.remembered_class_digest(mod.Plain) == plain
    assert digests == ["Plain"], "a digest no start reaches was taken again under start cells"


def test_a_digest_taken_under_starts_is_served_after_them(mod, digests):
    unrelated = mod.other(1).__code__
    with trace_items.call_starts(_starts(unrelated, ("x",))):
        under = trace_items.remembered_class_digest(mod.Hooked)
    assert trace_items.remembered_class_digest(mod.Hooked) == under
    assert digests == ["Hooked"]


def test_a_digest_through_a_function_whose_start_stands_in_is_taken_anew(mod, digests):
    plain = trace_items.remembered_class_digest(mod.Hooked)
    code = mod.Hooked.hook.__code__
    other = trace_items.value_digest(5)
    with trace_items.call_starts(_starts(code, (other,))):
        started = trace_items.remembered_class_digest(mod.Hooked)
        assert trace_items.remembered_class_digest(mod.Hooked) == started
    assert started != plain, "the start cells did not stand in for the hook's cells"
    assert trace_items.remembered_class_digest(mod.Hooked) == plain, "the call's digest was kept as the class's"
    assert digests == ["Hooked", "Hooked", "Hooked"]


def test_a_start_whose_cells_all_hold_classes_is_read_as_with_none(mod, digests):
    plain = trace_items.remembered_class_digest(mod.Plain)
    code = mod.Plain.run.__code__
    assert code.co_freevars == ("__class__",)
    with trace_items.call_starts(_starts(code, (trace_items._AS_IT_IS,))):
        assert trace_items.remembered_class_digest(mod.Plain) == plain
    assert digests == ["Plain"]


def test_a_class_digested_inside_another_passes_its_start_up(mod, digests):
    """The holder's digest goes through the hook's function by way of the
    class it holds, remembered or not: it is not kept under that start."""

    class Holder:
        inner = mod.Hooked

    Holder.__module__ = mod.__name__
    Holder.__qualname__ = "Holder"
    trace_items.remembered_class_digest(mod.Hooked)
    plain = trace_items.remembered_class_digest(Holder)
    code = mod.Hooked.hook.__code__
    with trace_items.call_starts(_starts(code, (trace_items.value_digest(5),))):
        assert trace_items.remembered_class_digest(Holder) != plain
