"""A statement that draws from an iterator held in a list, a dict or an
attribute runs every time, as one that draws from a name holding it does.

``rest = list(h[0])`` moves the iterator inside ``h``; the check that refuses
a hit looked only at what the name itself holds, so a hit was served (once
the statement cleared the cost floor) and left the iterator where it was.
Re-run alone, the cell then gave the value it gave on Run All in one config
and the exhausted ``[]`` in another.
"""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus

_PERSIST = CacheAnnotation(persist=True)


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


@pytest.mark.parametrize(
    ("holder", "draw"),
    [
        ("h = [iter(range(6))]", "rest = list(h[0])"),
        ("h = {'k': iter(range(6))}", "rest = list(h['k'])"),
        ("h = ([iter(range(6))],)", "rest = list(h[0][0])"),
        ("class B:\n    pass\nh = B()\nh.it = iter(range(6))", "rest = list(h.it)"),
        ("h = [1, iter(range(6))]\ni = 1", "rest = list(h[i])"),
    ],
    ids=["list", "dict", "nested", "attribute", "index-not-constant"],
)
def test_a_draw_through_a_holder_runs_again(statement_processor, mock_shell, holder, draw):
    _run(statement_processor, holder)
    first = _run(statement_processor, draw)
    assert mock_shell.user_ns["rest"] == [0, 1, 2, 3, 4, 5]
    second = _run(statement_processor, draw)
    assert first["status"] == second["status"] == CacheStatus.COMPUTED
    assert any("iterator or stream" in r for r in second["uncacheable_reasons"])
    assert mock_shell.user_ns["rest"] == []


def test_another_item_of_the_holder_is_served(statement_processor):
    """Control: the statement reads an int beside the iterator."""
    _run(statement_processor, "h = [iter(range(6)), 5]")
    _run(statement_processor, "n = h[1] * 2")
    assert _run(statement_processor, "n = h[1] * 2")["status"] == CacheStatus.RESTORED


def test_a_property_is_not_run_to_look_for_a_stream(statement_processor, mock_shell):
    """Control: the check reads attributes as stored, never through a property."""
    _run(
        statement_processor,
        "class P:\n    reads = 0\n    @property\n    def it(self):\n        P.reads += 1\n        return 3\np = P()",
    )
    _run(statement_processor, "n = list(range(p.it))")
    assert mock_shell.user_ns["P"].reads == 1
