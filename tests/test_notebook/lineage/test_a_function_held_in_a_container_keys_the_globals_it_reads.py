"""A statement that reaches a notebook function through what holds it keys
the globals the function reads as they are at the call.

``fns = [make]`` above ``r = fns[0](3)``, with ``make`` reading ``cfg``: the
statement names ``fns``, a list, and its key followed only what an input
names as a function. With no notebook file to simulate (a headless run),
nothing else moved it when ``cfg`` was edited, and the old ``r`` was served.
The same held for a dict of builders, a partial, a lambda in a dict, a
function kept as another's default, a class attribute, a method of a
notebook class, a library object holding the function, and a function whose
body calls through a list. A value too large to walk runs uncached, with a
reason, rather than be keyed without what it may hold.
"""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook import held_functions
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)

MAKE = "def make(n):\n    return [cfg['w'] * n]"

#: id -> (the definitions, what holds the function, the call through it)
SHAPES = {
    "list": (MAKE, "fns = [make]", "fns[0](3)"),
    "dict of builders": (MAKE, "builders = {'a': make, 'b': make}", "builders['a'](3)"),
    "nested spec": (MAKE, "spec = {'models': [('a', make)]}", "spec['models'][0][1](3)"),
    "partial": ("import functools\n" + MAKE, "p = functools.partial(make, 3)", "p()"),
    "lambda in a dict": ("pass", "d = {'f': lambda n: [cfg['w'] * n]}", "d['f'](3)"),
    "default argument": (MAKE, "def run(f=make):\n    return f(3)", "run()"),
    "closure": (MAKE + "\ndef bind(f):\n    def g(n):\n        return f(n)\n    return g", "h = bind(make)", "h(3)"),
    "class attribute": (MAKE, "class Reg:\n    fn = staticmethod(make)", "Reg.fn(3)"),
    "method": ("class M:\n    def build(self, n):\n        return [cfg['w'] * n]", "pass", "M().build(3)"),
    "library object": ("import types\n" + MAKE, "box = types.SimpleNamespace(fns={'a': make})", "box.fns['a'](3)"),
    "body calls through a list": (MAKE, "fns = [make]\ndef run_all():\n    return [f(3) for f in fns][0]", "run_all()"),
}


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_an_edited_global_re_keys_the_call(statement_processor, mock_shell, shape):
    definitions, hold, call = SHAPES[shape]
    statement = f"r = {call}"
    ns = mock_shell.user_ns
    _run(statement_processor, "cfg = {'w': 1}")
    _run(statement_processor, definitions)
    _run(statement_processor, hold)
    _run(statement_processor, statement)
    assert ns["r"] == [3]
    assert _run(statement_processor, statement)["status"] == CacheStatus.RESTORED, "control: unchanged, served"

    _run(statement_processor, "cfg = {'w': 2}")
    edited = _run(statement_processor, statement)
    assert ns["r"] == [6], shape
    assert edited["status"] == CacheStatus.COMPUTED

    _run(statement_processor, "cfg['w'] = 5")
    _run(statement_processor, statement)
    assert ns["r"] == [15], shape

    # The definitions and the holder run again over the changed global.
    _run(statement_processor, definitions)
    _run(statement_processor, hold)
    _run(statement_processor, statement)
    assert ns["r"] == [15], shape


def test_a_holder_too_large_to_walk_runs_uncached(statement_processor, mock_shell, monkeypatch):
    monkeypatch.setattr(held_functions, "WALK_LIMIT", 50)
    _run(statement_processor, "cfg = {'w': 1}")
    _run(statement_processor, MAKE)
    _run(statement_processor, "import types\nfns = [make] + [types.SimpleNamespace(i=i) for i in range(100)]")
    first = _run(statement_processor, "r = fns[0](3)")
    second = _run(statement_processor, "r = fns[0](3)")
    assert first["status"] == second["status"] == CacheStatus.COMPUTED
    assert any("runs every time" in r and "'fns'" in r for r in second["uncacheable_reasons"]), second

    _run(statement_processor, "cfg['w'] = 2")
    _run(statement_processor, "r = fns[0](3)")
    assert mock_shell.user_ns["r"] == [6]


def test_plain_data_is_served_as_before(statement_processor):
    _run(statement_processor, "rows = [{'a': i} for i in range(5000)]")
    _run(statement_processor, "n = len(rows)")
    second = _run(statement_processor, "n = len(rows)")
    assert second["status"] == CacheStatus.RESTORED
    assert not second["uncacheable_reasons"]
