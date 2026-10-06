"""A statement that draws from an iterator held in a variable runs every time.

``header = parse(next(rows))`` moves ``rows``; a stored entry holds ``header``
only, so a hit would leave ``rows`` where it was and the next reader would get
the same item again.
"""

from __future__ import annotations

from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def _define(processor):
    _run(processor, "def parse(x):\n    return x")


def test_a_draw_runs_again_and_advances_the_iterator(statement_processor, mock_shell):
    _define(statement_processor)
    _run(statement_processor, "rows = iter(['h', 'a', 'b'])")
    first = _run(statement_processor, "header = parse(next(rows))")
    second = _run(statement_processor, "header = parse(next(rows))")
    assert first["status"] == second["status"] == CacheStatus.COMPUTED
    assert any("iterator or stream" in r for r in second["uncacheable_reasons"])
    assert mock_shell.user_ns["header"] == "a"
    assert list(mock_shell.user_ns["rows"]) == ["b"]


def test_the_same_read_of_a_list_is_served(statement_processor):
    _define(statement_processor)
    _run(statement_processor, "rows = ['h', 'a', 'b']")
    _run(statement_processor, "header = parse(rows[0])")
    second = _run(statement_processor, "header = parse(rows[0])")
    assert second["status"] == CacheStatus.RESTORED


def test_inspecting_a_queue_is_served(statement_processor):
    _run(statement_processor, "import queue\nq = queue.Queue()")
    _run(statement_processor, "n = q.qsize()")
    second = _run(statement_processor, "n = q.qsize()")
    assert second["status"] == CacheStatus.RESTORED
