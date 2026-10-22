"""A statement that runs again under the same key hands its new value on.

Some statements run every time although their key does not move: one that
reads from an iterator (``t = next(GEN)``), one that finds its function by a
string, one whose stored entry its trace says is stale (a function it ran got
new code that no statement rebound). Run again, such a statement can bind a
new value. Its outputs kept the lineage its key gave, so ``u = t + 1`` below
was keyed as before and served the value computed from the old ``t``: in a
plain Run All, ``t`` went 0, 1, 2 while ``u`` stayed 1.

Their outputs are keyed by value now, as a ``# @cash:no-cache`` statement's
are: what is built on them misses when the value changed, and still hits
when it did not.
"""

from __future__ import annotations

import sys

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook import shadow
from cash.notebook.cache_status import CacheStatus

TRACED = sys.version_info >= (3, 12)

_PERSIST = CacheAnnotation(persist=True)


@pytest.fixture
def trace_default(tmp_path, monkeypatch, request):
    """The trace as it runs by default (no CASH_SHADOW), and a module of the
    test's own on the path."""
    monkeypatch.delenv("CASH_SHADOW", raising=False)
    monkeypatch.delenv("CASH_TRACE", raising=False)
    name = "rerun_value_" + "".join(ch if ch.isalnum() else "_" for ch in request.node.name)
    (tmp_path / f"{name}.py").write_text("GEN = (i for i in range(100))\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = shadow.SHADOW
    shadow.disable()
    on = shadow.enable() if shadow.requested() else None
    yield on, name
    shadow.disable()
    sys.modules.pop(name, None)
    if before is not None:
        shadow.enable()


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def test_a_value_drawn_from_an_iterator_reaches_what_reads_it(trace_default, statement_processor, mock_shell):
    """THE BUG: u was served 1 on every Run All while t went 0, 1, 2."""
    _, name = trace_default
    ns = mock_shell.user_ns
    seen = []
    for _ in range(3):
        for code in (f"from {name} import GEN", "t = next(GEN)", "u = t + 1"):
            _run(statement_processor, code)
        seen.append((ns["t"], ns["u"]))
    assert seen == [(0, 1), (1, 2), (2, 3)]


def test_a_function_found_by_a_string_hands_its_new_result_on(trace_default, statement_processor, mock_shell):
    """``globals()[name](3)``, the function edited: t reran to 8, u was
    served 4."""
    cells = ["cfg = 1", "name = 'make'", "t = globals()[name](3)", "u = t + 0"]
    ns = mock_shell.user_ns
    for make in ("def make(x):\n    return x + cfg * 1", "def make(x):\n    return x + cfg * 5"):
        _run(statement_processor, make)
        for code in cells:
            _run(statement_processor, code)
        assert ns["u"] == ns["t"] == ns["make"](3)


@pytest.mark.skipif(not TRACED, reason="the trace that sees code changed in place runs on 3.12+")
def test_a_rerun_the_trace_forced_hands_its_new_result_on(trace_default, statement_processor, mock_shell):
    """``inner`` given new code outside any statement: the trace reran t to
    30, and u was served 7."""
    _run(statement_processor, "def inner(x):\n    return x * 2")
    for code in ("t = inner(3)", "u = t + 1"):
        assert _run(statement_processor, code)["status"] == CacheStatus.COMPUTED
    mock_shell.user_ns["inner"].__code__ = (lambda x: x * 10).__code__

    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.COMPUTED
    assert _run(statement_processor, "u = t + 1")["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["u"] == 31
    # Settled: both are served the new values.
    for code in ("t = inner(3)", "u = t + 1"):
        assert _run(statement_processor, code)["status"] == CacheStatus.RESTORED
    assert (mock_shell.user_ns["t"], mock_shell.user_ns["u"]) == (30, 31)


def test_the_same_value_drawn_again_is_still_served_below(trace_default, statement_processor, mock_shell):
    """Keyed by value, not afresh: an equal value keeps what reads it served."""
    _run(statement_processor, "import itertools")
    _run(statement_processor, "it = itertools.repeat(5)")
    statuses = []
    for _ in range(3):
        _run(statement_processor, "t = next(it)")
        statuses.append(_run(statement_processor, "u = t + 1")["status"])
    assert statuses[1:] == [CacheStatus.RESTORED, CacheStatus.RESTORED]
    assert mock_shell.user_ns["u"] == 6
