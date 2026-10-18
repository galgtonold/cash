"""The recorder of what a statement ran keeps no code that is gone.

It kept every code object it armed: a notebook compiles each statement it
runs into code of its own, so a long session grew by one per statement,
for good. It holds them weakly now. Code that is gone cannot run again, so
nothing is lost: what still runs is still recorded, and a function's new
code, once its old code is gone, is recorded afresh and checked.
"""

from __future__ import annotations

import gc
import sys

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.decorator import code_trace
from cash.notebook import shadow
from cash.notebook.cache_status import CacheStatus

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="traces are recorded on Python 3.12+")

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture
def traced(monkeypatch):
    monkeypatch.delenv("CASH_SHADOW", raising=False)
    monkeypatch.delenv("CASH_TRACE", raising=False)
    before = shadow.SHADOW
    shadow.disable()
    on = shadow.enable()
    yield on
    shadow.disable()
    if before is not None:
        shadow.enable()


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)["status"]


def _statement_codes(recorder):
    return [c for c in list(recorder.user_codes.keys()) if c.co_name == "<module>"]


def _clear_bounded_caches():
    """Empty cash's caches keyed by code, each bounded on its own (a few
    thousand entries): what is left alive is what the recorder keeps."""
    from cash.notebook import string_lookups
    from cash.tracking import trace_items

    for memo in (string_lookups._FACTS, string_lookups._STATEMENT_CODES, string_lookups._TRACEABLE):
        memo.clear()
    trace_items._TABLES.clear()
    trace_items._DIGESTS.clear()
    code_trace._INSTRUCTIONS.clear()


def test_a_long_session_keeps_no_statement_code(traced, statement_processor):
    recorder = code_trace._recorder()
    _run(statement_processor, "def f(x):\n    return x + 1")
    for i in range(300):
        _run(statement_processor, f"y{i} = f({i})")
    _clear_bounded_caches()
    gc.collect()
    assert len(_statement_codes(recorder)) < 10
    assert len(recorder.user_codes) < 50


def test_a_function_whose_old_code_is_gone_is_still_checked(traced, statement_processor, mock_shell):
    """Its old code dropped from the recorder, the function's new code is
    seen when it runs: the stored statement reruns, never served stale."""
    recorder = code_trace._recorder()
    _run(statement_processor, "def inner(x):\n    return x * 2")
    assert _run(statement_processor, "t = inner(3)") == CacheStatus.COMPUTED
    gc.collect()
    assert _run(statement_processor, "t = inner(3)") == CacheStatus.RESTORED
    fn = mock_shell.user_ns["inner"]
    old = fn.__code__
    assert old in recorder.user_codes
    fn.__code__ = compile("def inner(x):\n    return x * 10", "<edit>", "exec").co_consts[0]
    del old
    gc.collect()
    assert _run(statement_processor, "t = inner(3)") == CacheStatus.COMPUTED
    assert mock_shell.user_ns["t"] == 30
    assert _run(statement_processor, "t = inner(3)") == CacheStatus.RESTORED
