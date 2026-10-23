"""A draw from an iterator the notebook set on a library's module runs
every time.

``textwrap.cnt = itertools.count()``, then ``a = next(textwrap.cnt)`` and
``b = next(textwrap.cnt)``: once the value set on ``textwrap`` was folded
into the readers' keys, a run again served ``a`` without drawing, and ``b``,
keyed on the counter where the first draw had not moved it, drew 0 where
plain Python draws 1 (Python 3.12, where the counter pickles). The read of
a stream set on a library's module is a draw, as one from a variable is:
also when the stream is held inside what was set (an object, a list), is a
random generator, or is read through a notebook function. A cell reading
what the draws bound found the stream moved on since the draws were keyed,
and drew again to rebuild them: the stream is keyed by which object it is.
"""

from __future__ import annotations

import pytest

from cash.notebook import callee_reach
from tests._cell_driver import run_cash_cell

#: ``(cell setting textwrap.cnt, a draw from it, a cell defining what they use)``.
SHAPES = {
    "a_counter": ("textwrap.cnt = itertools.count()", "next(textwrap.cnt)", ""),
    "a_list_iterator": ("textwrap.cnt = iter(range(100))", "next(textwrap.cnt)", ""),
    "an_iterator_in_an_object": (
        "textwrap.cnt = Box(itertools.count())",
        "next(textwrap.cnt.it)",
        "class Box:\n    def __init__(self, it):\n        self.it = it",
    ),
    "a_method_drawing": (
        "textwrap.cnt = Box(itertools.count())",
        "textwrap.cnt.draw()",
        "class Box:\n    def __init__(self, it):\n        self.it = it\n    def draw(self):\n        return next(self.it)",
    ),
    "an_iterator_in_a_list": ("textwrap.cnt = [itertools.count()]", "next(textwrap.cnt[0])", ""),
    "a_numpy_generator": ("textwrap.cnt = np.random.default_rng(0)", "int(textwrap.cnt.integers(1000))", ""),
    "a_random_generator": ("textwrap.cnt = random.Random(0)", "textwrap.cnt.random()", ""),
    "an_iterator_after_many_numbers": ("textwrap.cnt = [0] * 1500 + [itertools.count()]", "next(textwrap.cnt[-1])", ""),
    "an_iterator_after_many_lists": (
        "textwrap.cnt = [[0] for _ in range(6000)] + [itertools.count()]",
        "next(textwrap.cnt[-1])",
        "",
    ),
    "an_iterator_six_lists_deep": (
        "textwrap.cnt = [[[[[[itertools.count()]]]]]]",
        "next(textwrap.cnt[0][0][0][0][0][0])",
        "",
    ),
    "an_iterator_past_the_walk": (
        "textwrap.cnt = Nest(40, itertools.count())",
        "next(textwrap.cnt.it)",
        "class Nest:\n    def __init__(self, n, it):\n        self.inner = Nest(n - 1, it) if n else None\n        self.it = it",
    ),
    "a_notebook_function_drawing": (
        "textwrap.cnt = itertools.count()",
        "d()",
        "def d():\n    return next(textwrap.cnt)",
    ),
}


@pytest.mark.parametrize("every_call_stored", [False, True])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_each_run_draws_as_plain_python_does(cash_magics, cash_instance, shape, every_call_stored, monkeypatch):
    """A cheap draw, a slow one, and a cell reading both: the reader does
    not draw again to rebuild what the draws bound. With every call stored,
    a call of a notebook function that draws is not served either."""
    np = pytest.importorskip("numpy")
    import textwrap

    if every_call_stored:
        monkeypatch.setattr(cash_instance.config, "call_cost_floor_seconds", 0.0)
        monkeypatch.setattr(cash_instance.config, "min_execution_time_to_cache_seconds", 0.0)

    setter, draw, defines = SHAPES[shape]
    monkeypatch.setattr(callee_reach, "_SET_ON_LIBRARIES", {})
    cells = [
        cell
        for cell in (
            "import time, itertools, random\nimport numpy as np\nimport textwrap",
            defines,
            setter,
            f"a = (time.sleep(0.05), {draw})[1]",
            f"b = (time.sleep(0.3), {draw})[1]",
            "seen = (a, b)",
        )
        if cell
    ]
    plain: dict = {"np": np}
    for cell in cells:
        exec(cell, plain)
    textwrap.__dict__.pop("cnt", None)
    ns = cash_magics.shell.user_ns
    seen = []
    try:
        for _ in range(3):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
            seen.append(ns["seen"])
    finally:
        textwrap.__dict__.pop("cnt", None)
    assert seen == [plain["seen"]] * 3


def test_the_draw_is_named(monkeypatch):
    import itertools
    import textwrap

    monkeypatch.setattr(callee_reach, "_SET_ON_LIBRARIES", {"textwrap": {"cnt"}})
    monkeypatch.setattr(textwrap, "cnt", itertools.count(), raising=False)
    ns = {"textwrap": textwrap, "tw": textwrap}
    assert callee_reach.library_streams_read("a = next(textwrap.cnt)", ns) == ["textwrap.cnt"]
    assert callee_reach.library_streams_read("a = next(getattr(tw, 'cnt'))", ns) == ["textwrap.cnt"]
    assert callee_reach.library_streams_read("from textwrap import cnt\na = next(cnt)", ns) == ["textwrap.cnt"]
    assert callee_reach.library_streams_read("a = textwrap.dedent('x')", ns) == []


@pytest.mark.parametrize(
    "code",
    [
        "importlib.import_module('textwrap').cnt = itertools.count()",
        "__import__('textwrap').cnt = itertools.count()",
        "vars(textwrap).update(cnt=itertools.count())",
        "textwrap.__dict__.setdefault('cnt', itertools.count())",
        "m = sys.modules['textwrap']; m.cnt = itertools.count()",
        "import textwrap as m; m.cnt = itertools.count()",
    ],
)
def test_each_spelling_of_a_set_is_recorded(monkeypatch, code):
    import importlib
    import sys
    import textwrap

    monkeypatch.setattr(callee_reach, "_SET_ON_LIBRARIES", {})
    callee_reach.module_state_writes(code, {"textwrap": textwrap, "sys": sys, "importlib": importlib})
    assert callee_reach._SET_ON_LIBRARIES == {"textwrap": {"cnt"}}


@pytest.mark.parametrize(
    "code",
    [
        "a = next(sys.modules['textwrap'].cnt)",
        "a = next(importlib.import_module('textwrap').cnt)",
        "a = next(vars(textwrap)['cnt'])",
        "a = next(textwrap.__dict__['cnt'])",
    ],
)
def test_each_spelling_of_a_read_is_a_draw(monkeypatch, code):
    import importlib
    import itertools
    import sys
    import textwrap

    monkeypatch.setattr(callee_reach, "_SET_ON_LIBRARIES", {"textwrap": {"cnt"}})
    monkeypatch.setattr(textwrap, "cnt", itertools.count(), raising=False)
    ns = {"textwrap": textwrap, "sys": sys, "importlib": importlib}
    assert callee_reach.library_streams_read(code, ns) == ["textwrap.cnt"]


@pytest.mark.parametrize(
    ("value", "held", "warned"),
    [
        ("[0] * 1500 + [itertools.count()]", True, False),
        ("[[[[[[itertools.count()]]]]]]", True, False),
        ("[[i, i + 1] for i in range(1500)]", False, False),
        ("[[0] for _ in range(6000)] + [itertools.count()]", True, True),
        ("[list(range(3)) for _ in range(6000)]", True, True),
        ("functools.reduce(lambda inner, _: [inner], range(40), [0])", True, True),
    ],
)
def test_a_value_the_walk_cannot_finish_is_read_as_a_draw_with_a_warning(monkeypatch, value, held, warned):
    """``[0] * 1500 + [count()]`` or a ``count()`` six lists deep: past its
    bound the walk answered "no draw", the value was keyed by its state, and
    the draws came out wrong with nothing said. Past the bound it now
    answers "may draw", and says so once."""
    import functools
    import itertools
    import textwrap
    import warnings

    monkeypatch.setattr(callee_reach, "_SET_ON_LIBRARIES", {"textwrap": {"cnt"}})
    monkeypatch.setattr(callee_reach, "_UNCHECKED_WARNED", set())
    monkeypatch.setattr(textwrap, "cnt", eval(value, {"itertools": itertools, "functools": functools}), raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert callee_reach.library_draw_source("textwrap.cnt", textwrap.cnt) is held
        callee_reach.library_draw_source("textwrap.cnt", textwrap.cnt)
    codes = [getattr(w.message, "code", None) for w in caught]
    assert codes == (["KEY-LIBRARY-VALUE-UNCHECKED"] if warned else []), codes
