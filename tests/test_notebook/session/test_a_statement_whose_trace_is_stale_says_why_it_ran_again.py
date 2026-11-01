"""A stored statement whose trace says code it ran changed runs again, and
its miss reason names that code: refused with no reason, a slow statement
that ran again left the user nothing to look at."""

from __future__ import annotations

import sys

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook import shadow
from cash.notebook.cache_status import CacheStatus

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring")

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture
def trace_default(monkeypatch):
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
    return processor.process_statement(code, annotation=_PERSIST)


def test_the_miss_reason_names_the_code_that_changed(trace_default, statement_processor, mock_shell):
    assert trace_default is not None
    _run(statement_processor, "def inner(x):\n    return x * 2")
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.COMPUTED
    mock_shell.user_ns["inner"].__code__ = (lambda x: x * 10).__code__

    again = _run(statement_processor, "t = inner(3)")

    assert again["status"] == CacheStatus.COMPUTED
    reason = again.get("miss_reason") or ""
    assert reason.startswith("code it ran when it was stored (__main__.inner) was replaced or is gone"), reason
