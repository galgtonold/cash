"""A function of the user's that ran in a statement on a path cash does not
read from its text, and changes something the statement does not name, keeps
the statement from being stored (on Python 3.12+ from the trace of the run;
without one, from what its objects could run).

``Vec.__add__`` appends to ``calls``. Run by ``s + 1`` on an object-dtype
column of ``Vec``, by ``d[k][0] + 1``, by a closure, a classmethod, a
comprehension, ``sum(lst, Vec())`` or ``partial(operator.add, v)``: each was
served after a restart with ``calls == []``, where a run appends.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest

from cash.decorator import code_trace
from cash.notebook import shadow
from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

#: Without a trace (Python 3.10 and 3.11, ``CASH_TRACE=0``) the objects are
#: read instead (`writers_without_a_trace`): the same tests hold.
NEEDS_THE_TRACE = pytest.mark.skipif(not code_trace.on(), reason="reads the trace of the run (3.12+)")

SLEEP = f"(time.sleep({F}), 0)[1]"


@pytest.fixture
def main_namespace(mock_shell, monkeypatch):
    """The notebook's namespace is ``__main__``'s, as in IPython. A class
    of an object served from memory is looked up by its name there, as a
    read from disk does: with a namespace of its own, an object made before
    the class's cell ran again came back of the old class (3.10 under load:
    ``v = Vec()`` slow enough to be stored, then ``v + 1`` ran the old
    class's ``__add__``)."""
    main = types.ModuleType("__main__")
    main.__dict__.update(mock_shell.user_ns)
    mock_shell.user_ns = mock_shell.user_global_ns = main.__dict__
    monkeypatch.setitem(sys.modules, "__main__", main)


HEAD = """import time, types, functools, operator
calls = []
class Vec:
    def __add__(self, x):
        calls.append(1)
        return 0
"""
NAMES = (
    "GEN",
    "H",
    "STATE",
    "St",
    "heapq",
    "Two",
    "calls",
    "Vec",
    "v",
    "d",
    "k",
    "i",
    "lst",
    "np",
    "pd",
    "arr",
    "s",
    "df",
    "K",
    "g",
    "mk",
    "p",
    "w",
    "time",
)

CASES = {
    "numpy_elementwise": (
        "import numpy as np\narr = np.empty(1, dtype=object)\narr[0] = Vec()",
        f"w = (arr + 1, {SLEEP})",
    ),
    "pandas_elementwise": (
        "import pandas as pd\ns = pd.Series([Vec()], dtype=object)",
        f"w = (s + 1, {SLEEP})",
    ),
    "two_steps_down": ("d = {'a': [Vec()]}\nk = 'a'", f"w = (d[k][0] + 1, {SLEEP})"),
    "closure": (
        "v = Vec()\ndef mk():\n    x = v\n    def g(a):\n        return x + 1\n    return g\ng = mk()",
        f"w = g({SLEEP})",
    ),
    "classmethod": (
        "v = Vec()\nclass K:\n    @classmethod\n    def m(cls, a):\n        return v + 1",
        f"w = K.m({SLEEP})",
    ),
    "comprehension": ("lst = [Vec()]", f"w = ([x + 1 for x in lst], {SLEEP})"),
    "sum_from_an_object": ("lst = [Vec()]", f"w = (sum(lst, Vec()), {SLEEP})"),
    "partial_of_an_operator": ("v = Vec()\np = functools.partial(operator.add, v)", f"w = p({SLEEP})"),
}


#: Cases whose write the source already shows: ``K.m`` reads ``v`` and runs
#: ``v + 1``, which the callee analysis follows to ``Vec.__add__`` appending
#: to ``calls``. The statement runs each time for that reason ("Callee
#: mutates: calls"), so the unseen-writes warning has nothing to add.
SEEN_IN_THE_SOURCE = frozenset({"classmethod"})


def _restart(magics):
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _codes(magics, cell):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(magics, cell)
    return [c for w in caught if (c := getattr(w.message, "code", None))]


@pytest.mark.parametrize("case", list(CASES))
def test_the_statement_runs_each_time_and_says_why(cash_magics, case):
    setup, statement = CASES[case]
    said = []
    for _ in range(2):
        run_cash_cell(cash_magics, HEAD + setup)
        said += _codes(cash_magics, statement)
        assert cash_magics.shell.user_ns["calls"] == [1]
        _restart(cash_magics)
    assert "NOTEBOOK-UNSEEN-WRITES" in said or case in SEEN_IN_THE_SOURCE


def _status(magics):
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


@NEEDS_THE_TRACE
def test_an_entry_stored_before_is_not_served(cash_magics, monkeypatch):
    """Stored while the check was not there, the entry's trace names
    ``Vec.__add__``: the hit is refused."""
    setup, statement = CASES["pandas_elementwise"]
    monkeypatch.setattr(processor, "unseen_writers", lambda *a: [])
    monkeypatch.setattr(shadow.Shadow, "unseen_writes", lambda *a: False)
    run_cash_cell(cash_magics, HEAD + setup)
    run_cash_cell(cash_magics, statement)
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]  # stored, without the check
    monkeypatch.undo()
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["COMPUTED"]
    assert cash_magics.shell.user_ns["calls"] == [1, 1]


def test_a_change_the_statement_names_is_stored(cash_magics):
    """``calls`` is an output of the statement (``calls += ...`` names it),
    so a served entry gives it back: stored and served."""
    setup, _ = CASES["pandas_elementwise"]
    statement = f"calls += [(s + 1).iloc[0], {SLEEP}]"
    run_cash_cell(cash_magics, HEAD + setup)
    assert not _codes(cash_magics, statement)
    run_cash_cell(cash_magics, "calls = []")
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]
    assert cash_magics.shell.user_ns["calls"] == [1, 0, 0]


def test_a_special_method_changing_nothing_outside_is_stored(cash_magics):
    pure = "import time\nimport pandas as pd\nclass Vec:\n    def __add__(self, x):\n        return 0\ns = pd.Series([Vec()], dtype=object)"
    statement = f"w = (s + 1, {SLEEP})"
    run_cash_cell(cash_magics, pure)
    assert not _codes(cash_magics, statement)
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]


def test_assume_safe_stores_it(cash_magics):
    setup, statement = CASES["pandas_elementwise"]
    run_cash_cell(cash_magics, HEAD + setup)
    assert not _codes(cash_magics, statement + "  # @cash:assume-safe")
    run_cash_cell(cash_magics, statement + "  # @cash:assume-safe")
    assert _status(cash_magics) == ["RESTORED"]


def test_a_generator_expression_is_read_with_the_function_it_is_made_in(cash_magics):
    """``sum(i * i for i in r)`` inside a function that ran: its text is
    the function's, which changes nothing outside."""
    run_cash_cell(cash_magics, "import time\ndef total(n):\n    return sum(i * i for i in range(n))")
    statement = f"w = (total(5), {SLEEP})"
    assert not _codes(cash_magics, statement)
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]


@NEEDS_THE_TRACE
def test_a_generator_expression_run_on_its_own_is_read_from_its_line(tmp_path, monkeypatch):
    """Made in a function that has returned (``mk()`` hands it out), it
    runs without that function: read from its own line."""
    import importlib
    import types

    from cash.notebook.statement.unseen_writes import own_writes

    (tmp_path / "gx_made.py").write_text(
        "calls = []\ndef mk():\n    a = sum(i for i in range(3))\n    return a, (calls.append(1) for _ in [0])\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("gx_made")
    monkeypatch.delitem(sys.modules, "gx_made")
    made = [c for c in module.mk.__code__.co_consts if isinstance(c, types.CodeType)]
    assert sorted(own_writes(c) for c in made) == [frozenset(), frozenset({"calls"})]


MADE_BY_A_LIBRARY = {
    "dataclass": "from dataclasses import dataclass\n@dataclass\nclass P:\n    a: int = 1\n    b: int = 2\n",
    "frozen_dataclass": "from dataclasses import dataclass\n@dataclass(frozen=True, order=True)\nclass P:\n"
    "    a: int = 1\n    b: int = 2\n",
    "namedtuple": "import collections\nP = collections.namedtuple('P', 'a b', defaults=(1, 2))\n",
    "typed_namedtuple": "import typing\nclass P(typing.NamedTuple):\n    a: int = 1\n    b: int = 2\n",
}


@pytest.mark.parametrize("kind", list(MADE_BY_A_LIBRARY))
def test_a_method_a_library_wrote_for_a_class_in_a_module_is_stored(cash_magics, tmp_path, monkeypatch, kind):
    """``@dataclass`` writes ``__init__`` from text of its own (``<string>``):
    for a class in a module of the user's that counted as a function whose
    unreadable text changes anything, and every statement making such an
    object was refused (the parameter sweep of a backtest, all 32 runs)."""
    name = f"made_{kind}"
    (tmp_path / f"{name}.py").write_text(MADE_BY_A_LIBRARY[kind], encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    run_cash_cell(cash_magics, f"import time\nfrom {name} import P")
    statement = f"w = (P(a=3).a, P(3) == P(3), repr(P(3)), {SLEEP})"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        for _ in range(2):
            run_cash_cell(cash_magics, statement)
    sys.modules.pop(name, None)
    assert _status(cash_magics) == ["RESTORED"]
    assert not [w for w in seen if "UNSEEN" in str(w.message)]


@pytest.mark.parametrize("made", ["v = Vec()\n", f"v = (Vec(), {SLEEP})[0]\n"], ids=["made", "stored"])
@pytest.mark.parametrize("how", ["lambda", "def", "staticmethod"])
def test_a_special_method_set_on_the_class_later_and_edited_runs_again(main_namespace, cash_magics, how, made):
    """``Vec.__add__ = lambda self, o: 10`` has no name of its own: an edit
    to 20 went unseen by the trace and the hit served 10. The trace finds it
    again as ``Vec.__add__``; code it cannot find again does not vouch.
    ``v`` served from a stored entry is of the class defined again."""
    setting = {
        "lambda": "Vec.__add__ = lambda self, o: {mark}\n",
        "def": "def _add(self, o):\n    return {mark}\nVec.__add__ = _add\ndel _add\n",
        "staticmethod": "Vec.mk = staticmethod(lambda: {mark})\nVec.__add__ = lambda self, o: Vec.mk()\n",
    }[how]
    seen = []
    for mark in (10, 20):
        for cell in (
            "import time\nclass Vec:\n    pass\n",
            made,
            setting.format(mark=mark),
            f"w = (v + 1, {SLEEP})",
        ):
            run_cash_cell(cash_magics, cell)
        seen.append(cash_magics.shell.user_ns["w"][0])
    assert seen == [10, 20]


WRITES = {
    "next": ("GEN = iter(range(5))", "next(GEN)", "next(GEN)", 1),
    "heappush": ("import heapq\nH = []", "heapq.heappush(H, 1)", "H", [1]),
    "handed_mutator": ("", "list(map(calls.append, [1]))", "calls", [1]),
    "method_on_a_global": (
        "class St:\n    def __init__(self):\n        self.n = 0\n    def bump(self):\n        self.n += 1\nSTATE = St()",
        "STATE.bump()",
        "STATE.n",
        1,
    ),
}


@pytest.mark.parametrize("kind", list(WRITES))
def test_a_special_method_that_changes_a_global_another_way_runs_each_time(cash_magics, kind):
    """``next(GEN)``, ``heapq.heappush(H, 1)``, ``map(calls.append, ...)`` and
    ``STATE.bump()`` in a special method reached two steps down
    (``d[k][0] + 1``) change a global as ``calls.append(1)`` does: a served
    entry would drop the change."""
    setup, body, effect, expected = WRITES[kind]
    head = (
        f"import time\ncalls = []\n{setup}\nclass Two:\n    def __add__(self, x):\n        {body}\n        return 0\n"
    )
    for _ in range(2):
        run_cash_cell(cash_magics, head + "d = {'a': [Two()]}\nk = 'a'")
        run_cash_cell(cash_magics, f"w = (d[k][0] + 1, {SLEEP})")
        assert eval(effect, cash_magics.shell.user_ns) == expected
        _restart(cash_magics)


#: A writer in a module that a later run need not reach: a cached property
#: filled by the first, a hash a tuple keeps from 3.14 on.
SKIPPED_LATER = {
    "cached_property": (
        "import functools\ncalls = []\nclass Vec:\n    @functools.cached_property\n    def val(self):\n"
        "        calls.append(1)\n        return 1\n    def __add__(self, o):\n        return self.val\n",
        "d = {{'a': [{m}.Vec()]}}\nk = 'a'",
        "w = (d[k][0] + 1, {s})",
    ),
    "namedtuple_hash": (
        "from collections import namedtuple\ncalls = []\nclass W:\n    def __hash__(self):\n"
        "        calls.append(1)\n        return 7\nP = namedtuple('P', 'a')\n",
        "p1 = {m}.P({m}.W())",
        "w = (hash(p1), {s})",
    ),
    "tuple_hash": (
        "calls = []\nclass W:\n    def __hash__(self):\n        calls.append(1)\n        return 7\n",
        "p1 = ({m}.W(), 1)",
        "w = (hash(p1), {s})",
    ),
}


@NEEDS_THE_TRACE
@pytest.mark.parametrize("kind", list(SKIPPED_LATER))
def test_a_run_that_skipped_the_writer_is_not_stored(cash_magics, tmp_path, monkeypatch, kind):
    """Refused once for the writer, the statement ran again without it (the
    property filled, the hash kept) and that run was stored, its trace
    holding no writer: after a restart it was served, and the writer that
    runs there did not. It is not stored while the writer is alive."""
    name = f"skipped_{kind}"
    module, setup, statement = SKIPPED_LATER[kind]
    (tmp_path / f"{name}.py").write_text(module, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    setup = f"import time\nimport {name}\n" + setup.format(m=name)
    statement = statement.format(s=SLEEP)
    try:
        for _ in range(2):
            sys.modules.pop(name, None)
            run_cash_cell(cash_magics, setup)
            for _ in range(2):
                run_cash_cell(cash_magics, statement)
                assert _status(cash_magics) == ["COMPUTED"]
            assert len(sys.modules[name].calls) >= 1
            _restart(cash_magics)
    finally:
        sys.modules.pop(name, None)


@NEEDS_THE_TRACE
def test_a_writer_that_is_gone_lets_the_statement_be_stored(cash_magics):
    """Once the class holding the writer is gone (redefined without it,
    nothing holding the old one), the statement is stored again."""
    import gc

    setup, statement = CASES["two_steps_down"]
    run_cash_cell(cash_magics, HEAD + setup)
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["COMPUTED"]
    run_cash_cell(cash_magics, "class Vec:\n    def __add__(self, x):\n        return 0\nd = {'a': [Vec()]}")
    gc.collect()
    for _ in range(2):
        run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]


MULTI_LINE = "import time\ncalls = []\nKEYS = {{\n    'first': lambda r: r[0],\n    'second': lambda r: {body},\n}}\nxs = [(1, 3), (2, 1)]"


@NEEDS_THE_TRACE
@pytest.mark.parametrize("body, stored", [("r[1]", True), ("calls.append(1) or r[1]", False)], ids=["pure", "writes"])
def test_a_lambda_in_a_literal_over_several_lines_is_read(cash_magics, body, stored):
    """``'second': lambda r: r[1],`` on a line of its own (black's dict)
    does not parse alone: it was "unreadable text" and refused every run.
    Read from the cell, a pure one is stored; one that writes is not."""
    run_cash_cell(cash_magics, MULTI_LINE.format(body=body))
    statement = f"w = (sorted(xs, key=KEYS['second'])[0][0], {SLEEP})"
    said = _codes(cash_magics, statement)
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == (["RESTORED"] if stored else ["COMPUTED"])
    assert ("NOTEBOOK-UNSEEN-WRITES" in said) is not stored


SWAPPED = {
    "deque_appendleft": ("DQ = collections.deque()", "DQ.appendleft(x) or x * 10", "DQ"),
    "list_append": ("DQ = []", "DQ.append(x) or x * 10", "DQ"),
    "dict_update": ("DQ = {}", "DQ.update(a=x) or x * 10", "DQ"),
}


#: ``(definition, what gets the new code, its parameters, the call)``.
REACHED = {
    "by_name": ("def inner(x):\n    return x * 2", "inner", "x", "inner(3)"),
    "as_a_method_of_an_object": (
        "class C:\n    def m(self, x):\n        return x * 2\nobj = C()",
        "C.m",
        "self, x",
        "obj.m(3)",
    ),
    # Run through a subclass holding nothing of it.
    "as_a_method_an_object_inherits": (
        "class C:\n    def m(self, x):\n        return x * 2\nclass D(C):\n    pass\nobj = D()",
        "C.m",
        "self, x",
        "obj.m(3)",
    ),
}


@pytest.mark.parametrize("trace", ["on", "off"])
@pytest.mark.parametrize("reached", list(REACHED))
@pytest.mark.parametrize("case", list(SWAPPED))
def test_a_function_given_code_whose_text_cannot_be_read_runs_each_time(cash_magics, monkeypatch, case, trace, reached):
    """``inner.__code__`` swapped for a lambda compiled from a string: its
    text cannot be read, and it appends to ``DQ``. With a trace the run
    names it; without one (``CASH_TRACE=0``) only the function ``inner(3)``
    names was there to read, and nothing did: a run slow enough to be
    stored was served after a restart with ``DQ`` left empty. Called as
    ``obj.m(3)``, the walk opened the object but not its class, and then
    not the class's bases."""
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
        monkeypatch.setattr(shadow, "SHADOW", None)
    elif not code_trace.on():
        pytest.skip("reads the trace of the run (3.12+)")
    setup, body, name = SWAPPED[case]
    definition, target, params, call = REACHED[reached]
    swap = f"{target}.__code__ = eval(compile({(f'lambda {params}: ' + body)!r}, '<variant>', 'eval')).__code__"
    said = []
    for _ in range(2):
        run_cash_cell(cash_magics, f"import collections, time\n{setup}\n{definition}")
        run_cash_cell(cash_magics, swap)
        said += _codes(cash_magics, f"v = ({call}, {SLEEP})")
        assert len(cash_magics.shell.user_ns[name]) == 1
        _restart(cash_magics)
        for gone in ("DQ", "inner", "C", "D", "obj", "collections"):
            cash_magics.shell.user_ns.pop(gone, None)
    assert "NOTEBOOK-UNSEEN-WRITES" in said, said


@pytest.mark.parametrize("trace", ["on", "off"])
def test_a_call_slow_enough_to_be_stored_is_not_when_it_ran_such_a_writer(cash_magics, monkeypatch, trace):
    """``g(a)`` returns ``x + 1`` for a closed-over ``x = Vec()``: the
    statement was refused, but the call, slow enough to be stored on its
    own, was served after a restart with ``calls == []`` (macOS under load
    made the closure case's quick ``g`` that slow)."""
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
        monkeypatch.setattr(shadow, "SHADOW", None)
    setup = (
        "v = Vec()\ndef mk():\n    x = v\n    def g(a):\n"
        f"        time.sleep({F})\n        return x + 1\n    return g\ng = mk()"
    )
    for _ in range(2):
        run_cash_cell(cash_magics, HEAD + setup)
        run_cash_cell(cash_magics, f"w = g({SLEEP})")
        assert cash_magics.shell.user_ns["calls"] == [1]
        _restart(cash_magics)
