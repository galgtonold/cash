"""A big value that reaches a notebook function, however it holds it, keys
the globals the function reads -- also when the function's only holder is
the notebook's namespace, which the value holds.

The walk for held functions stops early when no function with the
notebook's globals exists (`held_functions.notebook_functions_exist`). A
check per function -- "nothing but the namespace refers to ``make``, so no
value can hold it" -- would stop it far more often, and would be wrong:
``box = {"ns": globals()}`` holds every function the namespace does, and so
does a value holding the shell (``get_ipython().user_ns``) or any object of
the kernel that leads to it. ``r = box["ns"]["make"](3)`` must re-key when
``cfg`` changes. Each value here is past the point where the walk asks
(`EXISTENCE_CHECK_AT`, lowered), and a cell defined an unrelated function,
as every real notebook (and the replay harness's kernel) does.
"""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook import held_functions
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)

#: Objects that are not plain data: the walk visits each of them.
PAD = "[types.SimpleNamespace(i=i) for i in range(300)]"

#: ``make`` held by nothing but the namespace, and a function unrelated to it.
NAMESPACE_ONLY = "import types\ndef unrelated():\n    return 1\ndef make(n):\n    return [cfg['w'] * n]\n"

DEFINITIONS = (
    "import functools\nimport types\n"
    "def unrelated():\n    return 1\n"
    "def make(n):\n    return [cfg['w'] * n]\n"
    "def make_method(self, n):\n    return [cfg['w'] * n]\n"
    "def bind(f):\n    def g(n):\n        return f(n)\n    return g\n"
    "class Reg:\n    fn = staticmethod(make)\n"
    "class Cell:\n    __slots__ = ('value', 'parent')\n"
    "    def __init__(self, value, parent):\n        self.value = value\n        self.parent = parent\n"
)

#: id -> (what holds the function, the call through it)
SHAPES = {
    "the namespace in a dict": (f"box = {{'ns': globals(), 'pad': {PAD}}}", "box['ns']['make'](3)"),
    "the namespace as an attribute": (f"box = types.SimpleNamespace(ns=globals(), pad={PAD})", "box.ns['make'](3)"),
    "the shell": (f"box = [get_ipython()] + {PAD}", "box[0].user_ns['make'](3)"),
    "a list": (f"box = [make] + {PAD}", "box[0](3)"),
    "a dict value": (f"box = {{'f': make, 'pad': {PAD}}}", "box['f'](3)"),
    "an instance attribute": (f"box = types.SimpleNamespace(f=make, pad={PAD})", "box.f(3)"),
    "a closure cell": (f"box = [bind(make)] + {PAD}", "box[0](3)"),
    "a partial": (f"box = [functools.partial(make, 3)] + {PAD}", "box[0]()"),
    "a bound method": (f"box = [types.MethodType(make_method, types.SimpleNamespace())] + {PAD}", "box[0](3)"),
    "a class attribute": (f"box = [Reg] + {PAD}", "box[0].fn(3)"),
    "a cell of a workbook-like graph": (
        "sheet = types.SimpleNamespace(cells={})\n"
        "sheet.cells.update({(i, 1): Cell(float(i), sheet) for i in range(600)})\n"
        "sheet.cells[(450, 1)].value = make\n"
        "box = types.SimpleNamespace(sheets=[sheet])",
        "box.sheets[0].cells[(450, 1)].value(3)",
    ),
}


#: The shapes where nothing but the namespace refers to ``make``.
THROUGH_THE_NAMESPACE = ("the namespace in a dict", "the namespace as an attribute", "the shell")


@pytest.fixture(autouse=True)
def _asks_early(monkeypatch):
    monkeypatch.setattr(held_functions, "EXISTENCE_CHECK_AT", 50)


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_an_edited_global_re_keys_the_call(statement_processor, mock_shell, shape):
    hold, call = SHAPES[shape]
    statement = f"r = {call}"
    ns = mock_shell.user_ns
    ns["get_ipython"] = lambda: mock_shell
    _run(statement_processor, "cfg = {'w': 1}")
    _run(statement_processor, NAMESPACE_ONLY if shape in THROUGH_THE_NAMESPACE else DEFINITIONS)
    _run(statement_processor, hold)
    _run(statement_processor, statement)
    assert ns["r"] == [3]
    _run(statement_processor, statement)

    _run(statement_processor, "cfg = {'w': 2}")
    edited = _run(statement_processor, statement)
    assert ns["r"] == [6], shape
    assert edited["status"] == CacheStatus.COMPUTED

    _run(statement_processor, "cfg['w'] = 5")
    _run(statement_processor, statement)
    assert ns["r"] == [15], shape


@pytest.mark.parametrize("shape", ["the namespace in a dict", "a list"])
def test_the_call_is_served_while_nothing_it_reads_changed(statement_processor, mock_shell, shape):
    hold, call = SHAPES[shape]
    statement = f"r = {call}"
    _run(statement_processor, "cfg = {'w': 1}")
    _run(statement_processor, NAMESPACE_ONLY if shape in THROUGH_THE_NAMESPACE else DEFINITIONS)
    _run(statement_processor, hold)
    _run(statement_processor, statement)
    assert _run(statement_processor, statement)["status"] == CacheStatus.RESTORED
