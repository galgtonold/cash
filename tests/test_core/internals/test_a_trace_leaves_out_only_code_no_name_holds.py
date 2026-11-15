"""A call's trace leaves out the code no name holds, and keeps the rest.

A cached call that ran a lambda held only inside a global dict
(``PROBLEMS = {"a": lambda v: v + 1}``) had no trace at all: no item can name
the lambda again, so the whole trace was dropped and the call stored by its
key alone. Every other item went with it, among them the one for a module
function a library's method called by bare name, which was the only check that
would have seen an edit to it (dynesty's ``_configure_batch_sampler``).

Each item only adds a check, so a trace with the unnameable code left out is
still sound for a call the key accepted: the lambda is read through the
global dict that holds it, which the key (or the item of the code that reads
it) checks. A call that picks code by a run-time name needs a
whole trace, and is still refused one.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash.decorator import code_trace

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not code_trace.TRACEABLE, reason="the trace recorder needs sys.monitoring (Python 3.12+)"),
]

MODULE = textwrap.dedent("""
    def _configure(x):
        return x * 2


    class Base:
        def run(self, v):
            return _configure(v)


    PROBLEMS = {"a": lambda v: v + 1}


    def go(v):
        return Base().run(PROBLEMS["a"](v))
""")


@pytest.fixture
def module(tmp_path, monkeypatch):
    name = f"traced_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module(name)
    yield mod
    sys.modules.pop(name, None)


def _unit(mod):
    unit = code_trace.begin()
    if unit is None:
        pytest.skip(code_trace.recorder_or_reason().reason)
    try:
        assert mod.go(5) == 12
    finally:
        code_trace.end(unit)
    return unit


def test_a_call_that_needs_no_trace_keeps_the_items_beside_an_unnamed_lambda(module):
    items, why = code_trace.build(_unit(module), {module.go.__code__}, strict=False)
    assert why == ""
    assert items is not None
    keys = {item[:-1] for item in items}
    assert ("global", module.__name__, "_configure") in keys
    # The cached function reads PROBLEMS, so the key holds it; code outside
    # the key that read it would carry the item instead.
    whole, _ = code_trace.build(_unit(module), set(), strict=False)
    assert ("global", module.__name__, "PROBLEMS") in {item[:-1] for item in whole}


def test_a_call_that_needs_its_trace_is_still_refused_one(module):
    items, why = code_trace.build(_unit(module), {module.go.__code__}, strict=True)
    assert items is None
    assert "no name of" in why
