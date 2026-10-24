"""A statement that draws from an iterator held in a list, a dict or an
attribute runs every time, as one that draws from a name holding it does.
Where the way in cannot be read without running code (an index cash cannot
work out, a property, a slot, a method), any iterator the holder keeps counts.

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
        ("h = list(range(100)) + [iter(range(6))]", "rest = list(h[-1])"),
        ("h = [('a', iter(range(6)))]\ni = 0", "rest = list(h[i][1])"),
        ("h = [0] * 70 + [iter(range(6))]\ni = 70", "rest = list(h[i])"),
        ("import collections\nh = collections.deque([iter(range(6))])", "rest = list(h[0])"),
        ("import collections\nh = collections.defaultdict(list)\nh['k'] = iter(range(6))", "rest = list(h['k'])"),
        (
            "import collections\nP = collections.namedtuple('P', 'it')\nh = P(iter(range(6)))",
            "rest = list(h.it)",
        ),
        ("class S:\n    __slots__ = ('it',)\nh = S()\nh.it = iter(range(6))", "rest = list(h.it)"),
        (
            "class S:\n    def __init__(self, s):\n        self._s = s\n"
            "    @property\n    def it(self):\n        return self._s\nh = S(iter(range(6)))",
            "rest = list(h.it)",
        ),
        (
            "class S:\n    def __init__(self, s):\n        self._s = s\n"
            "    def get(self):\n        return self._s\nh = S(iter(range(6)))",
            "rest = list(h.get())",
        ),
        ("class B:\n    pass\nh = B()\nh.d = {'k': iter(range(6))}", "rest = list(h.d['k'])"),
    ],
    ids=[
        "list",
        "dict",
        "nested",
        "attribute",
        "index-not-constant",
        "negative-index-past-64",
        "index-not-constant-then-constant",
        "index-name-past-64",
        "deque",
        "defaultdict",
        "namedtuple",
        "slots",
        "property",
        "method",
        "dict-in-attribute",
    ],
)
def test_a_draw_through_a_holder_runs_again(statement_processor, mock_shell, holder, draw):
    _run(statement_processor, holder)
    first = _run(statement_processor, draw)
    assert mock_shell.user_ns["rest"] == [0, 1, 2, 3, 4, 5]
    second = _run(statement_processor, draw)
    assert first["status"] == second["status"] == CacheStatus.COMPUTED
    # A method call on a notebook object already runs as a change to it.
    assert any("iterator or stream" in r or "In-place mutation on: h" in r for r in second["uncacheable_reasons"])
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


def test_an_item_only_bound_to_a_name_is_served(statement_processor):
    """Control: ``x = h[i]`` does not draw from what it reads, so a holder too
    big to read through does not stop the hit."""
    _run(statement_processor, "h = list(range(2_000_000)) + [iter(range(6))]\ni = 5")
    _run(statement_processor, "x = h[i]")
    assert _run(statement_processor, "x = h[i]")["status"] == CacheStatus.RESTORED


def test_a_method_whose_result_is_dropped_is_not_a_draw(statement_processor, mock_shell):
    """Control: ``h.append(..)`` as its own statement draws nothing, however
    much ``h`` holds."""
    from cash.notebook.consumables import drawn_stream_inputs

    _run(statement_processor, "h = list(range(2_000_000)) + [iter(range(6))]")
    tree = __import__("ast").parse("h.append(1)")
    assert drawn_stream_inputs(tree, ["h"], [], mock_shell.user_ns) == []
