"""With the trace on by default (3.12+, ``CASH_SHADOW`` unset), a stored
statement or cached call whose trace says code it ran changed runs again,
even when its key did not move.

The edit here is one the key cannot see: a function gets new code outside
any statement (as a callback, a thread or a debugger could do it), so no
statement rebinds it and its lineage stays. The trace recorded that the
function ran, and its code now differs. Before 3.12 nothing is recorded, and
the key decides alone, as before.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook import shadow
from cash.notebook.cache_status import CacheStatus

TRACED = sys.version_info >= (3, 12)

_PERSIST = CacheAnnotation(persist=True)

_IMPL = """
import time

def inner(x):
    return x * 2

def slow_inner(x):
    time.sleep(0.25)
    return inner(x)
"""


@pytest.fixture
def trace_default(tmp_path, monkeypatch, request):
    """The shadow as it runs by default: no CASH_SHADOW, no log."""
    monkeypatch.delenv("CASH_SHADOW", raising=False)
    monkeypatch.delenv("CASH_TRACE", raising=False)
    name = "stale_impl_" + "".join(ch if ch.isalnum() else "_" for ch in request.node.name)
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_IMPL), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = shadow.SHADOW
    shadow.disable()
    on = shadow.enable() if shadow.requested() else None
    yield on, name
    shadow.disable()
    sys.modules.pop(name, None)
    if before is not None:
        shadow.enable()


def _rebind(mod):
    """Give ``inner`` new code outside any statement (a callback, a thread,
    a debugger): no statement says anything changed, so no key moves."""
    sys.modules[mod].inner.__code__ = (lambda x: x * 10).__code__


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def test_a_statement_reruns_when_a_function_it_ran_is_edited(trace_default, statement_processor, mock_shell):
    on, _ = trace_default
    assert (on is not None) == TRACED
    if on is not None:
        assert not on.diagnose
    _run(statement_processor, "def inner(x):\n    return x * 2")
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.COMPUTED
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.RESTORED
    mock_shell.user_ns["inner"].__code__ = (lambda x: x * 10).__code__

    again = _run(statement_processor, "t = inner(3)")
    if TRACED:
        assert again["status"] == CacheStatus.COMPUTED
        assert mock_shell.user_ns["t"] == 30
        assert on.counts["stmt:miss:rerun"] == 1, dict(on.counts)
    else:
        # The key alone, as before 3.12.
        assert again["status"] == CacheStatus.RESTORED


def test_a_cached_call_reruns_when_a_function_it_ran_is_edited(trace_default, statement_processor, mock_shell):
    on, mod = trace_default
    _run(statement_processor, f"import {mod}\nfrom {mod} import slow_inner")
    for _ in range(2):
        statement_processor.process_statement("x = slow_inner(2) + 0")
    assert mock_shell.user_ns["x"] == 4
    if TRACED:
        assert on.counts["call:traces"] >= 1
    _rebind(mod)

    statement_processor.process_statement("x = slow_inner(2) + 1")
    if TRACED:
        assert mock_shell.user_ns["x"] == 21
        assert on.counts["call:miss:rerun"] >= 1, (mock_shell.user_ns["x"], dict(on.counts))
    else:
        # The key alone, as before 3.12: the call is served.
        assert mock_shell.user_ns["x"] == 5
