"""How shadow mode records a unit (``sys.monitoring``) and digests what it
ran: the markers that make a trace unusable, and digests that stay equal
for what a re-run would not see differ."""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from cash.notebook import shadow
from cash.tracking import trace_items as items

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring")


@pytest.fixture
def on():
    before = shadow.SHADOW
    shadow.disable()
    sh = shadow.enable()
    assert sh is not None
    ns: dict = {"__name__": "__main__"}
    sh.notebook = ns
    sh.recorder.notebook_ns = ns
    yield sh, ns
    shadow.disable()
    if before is not None:
        shadow.enable()


def _define(ns: dict, src: str) -> None:
    exec(compile(src, "<cell>", "exec"), ns)


def _record(sh, fn):
    unit = sh.begin("stmt")
    try:
        fn()
    finally:
        sh.end(unit)
    return unit


def test_it_records_the_users_code_by_qualified_name(on):
    sh, ns = on
    _define(ns, "K = 2\ndef helper(x):\n    return x * K\nclass A:\n    def m(self):\n        return helper(1)")
    unit = _record(sh, lambda: ns["A"]().m())
    assert {"A.m", "helper"} <= {c.co_qualname for c in unit.codes}
    built, markers, _ = sh.build(unit)
    kinds = {i[:3] for i in built}
    assert ("code", "__main__", "A.m") in kinds
    assert ("class", "__main__", "A") in kinds
    assert ("global", "__main__", "K") in kinds
    assert not markers


def test_a_nested_unit_adds_what_it_saw_to_the_one_around_it(on):
    sh, ns = on
    _define(ns, "def a():\n    return 1\ndef b():\n    return 2")
    outer = sh.begin("stmt")
    ns["a"]()
    inner = sh.begin("call")
    ns["a"]()
    ns["b"]()
    sh.end(inner)
    sh.end(outer)
    assert {c.co_qualname for c in inner.codes} >= {"a", "b"}
    assert {c.co_qualname for c in outer.codes} >= {"a", "b"}


def test_a_generator_started_before_the_unit_marks_it(on):
    sh, ns = on
    _define(ns, "def gen():\n    yield 1\n    yield 2")
    g = ns["gen"]()
    next(g)
    unit = _record(sh, lambda: next(g))
    assert "generator_resumed" in unit.markers


def test_a_thread_of_the_users_still_running_marks_it(on):
    sh, ns = on
    stop = threading.Event()
    _define(ns, "def work(stop):\n    stop.wait(5)")
    holder = []

    def start():
        t = threading.Thread(target=ns["work"], args=(stop,))
        t.start()
        holder.append(t)

    unit = _record(sh, start)
    stop.set()
    holder[0].join()
    assert "thread_alive" in unit.markers


def test_a_thread_joined_inside_the_unit_does_not(on):
    sh, ns = on
    _define(ns, "def work():\n    return 1")

    def run():
        t = threading.Thread(target=ns["work"])
        t.start()
        t.join()

    unit = _record(sh, run)
    assert "thread_alive" not in unit.markers
    assert "work" in {c.co_qualname for c in unit.codes}


def test_a_lookup_by_string_marks_the_trace(on):
    sh, ns = on
    _define(ns, "def f():\n    return 1\ndef g():\n    return globals()['f']()")
    unit = _record(sh, lambda: ns["g"]())
    _, markers, _ = sh.build(unit)
    assert "string_lookup" in markers


def test_losing_the_tool_id_marks_the_unit(on):
    sh, _ = on
    unit = sh.begin("stmt")
    # The id that sees each code first: with its events off, nothing new is seen.
    tool = sh.recorder.disc
    sys.monitoring.set_events(tool, 0)
    sh.end(unit)
    assert "tool_conflict" in unit.markers
    assert sys.monitoring.get_tool(tool) == sh.recorder.tool_name
    sh.end(sh.begin("stmt"))  # the next unit arms them again
    assert sys.monitoring.get_events(tool) == sys.monitoring.events.PY_START | sys.monitoring.events.PY_RESUME


def test_worker_processes_mark_the_unit(on):
    import multiprocessing

    sh, _ = on
    # The platform's own start method: Windows has no fork.
    ctx = multiprocessing.get_context()

    def run():
        p = ctx.Process(target=time.sleep, args=(0,))
        p.start()
        p.join()

    unit = _record(sh, run)
    assert "process_pool" in unit.markers


def test_a_closure_is_checked_through_the_global_holding_it(on):
    sh, ns = on
    _define(ns, "def maker(n):\n    def inner(x):\n        return x + n\n    return inner\nadd = maker(3)")
    unit = _record(sh, lambda: ns["add"](1))
    built, _, _ = sh.build(unit)
    held = [i for i in built if i[0] == "held"]
    assert held and held[0][2] == "add"
    memo: dict = {}
    assert sh.verify({"items": built}, memo)["changed"] == []
    _define(ns, "add = maker(4)")
    assert [c[:3] for c in sh.verify({"items": built}, {})["changed"]] == [["held", "__main__", "add"]]


def test_a_docstring_is_no_part_of_a_code_digest():
    ns: dict = {}
    exec("def f(x):\n    'one'\n    return x", ns)
    exec("def g(x):\n    'two'\n    return x", ns)
    exec("def h(x):\n    return x", ns)
    f, g, h = ns["f"].__code__, ns["g"].__code__, ns["h"].__code__
    assert items.code_digest(f) == items.code_digest(g)
    if sys.version_info < (3, 14):
        # From 3.14 a function reserves the docstring slot only when it has
        # one, so adding or removing one shifts the constants, as the key
        # sees it too (`source_norm.code_consts_without_docstring`).
        assert items.code_digest(f) == items.code_digest(h.replace(co_name="f", co_qualname="f"))


def test_equal_values_digest_equal_and_a_big_one_is_opaque():
    assert items.value_digest({"a": [1, 2.5, "x"], "b": {3, 1}}) == items.value_digest(
        {"a": [1, 2.5, "x"], "b": {1, 3}}
    )
    assert items.value_digest({"a": 1}) != items.value_digest({"a": 2})
    assert items.value_digest(b"x" * (items.VALUE_CAP + 1)).startswith("opaque:")


def test_resolve_finds_methods_and_nested_code():
    mod = types.ModuleType("_shadow_resolve_mod")
    exec(
        "class C:\n    @staticmethod\n    def s():\n        return 1\n    @property\n    def p(self):\n        return 2\n"
        "def outer():\n    def inner():\n        return 3\n    return inner",
        mod.__dict__,
    )
    sys.modules[mod.__name__] = mod
    try:
        for qualname in ("C.s", "C.p", "outer.<locals>.inner"):
            code, _, _ = items.resolve(mod.__name__, qualname, {}, "__main__")
            assert code is not None and code.co_qualname == qualname
    finally:
        del sys.modules[mod.__name__]
