"""The share check is not asked again with anchors when it found no
variable holding an object of the outputs.

When the outputs' objects are held by something that is not a variable,
the check that refused is asked again with variables allowed to be
anchors. An anchor is a variable found holding an object of the outputs:
when the first check found none, the second one, which starts with the
same round, finds none either, and climbed the same referrers again. A
list nested 20,000 deep paid 16 heap scans for it. Skipped, the statement
gets the answer the retry gave.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell

SHAPES = {
    "a list nested deeper than the climb": [
        "big = []\ncur = big\nfor _ in range(2_000):\n    nxt = []\n    cur.append(nxt)\n    cur = nxt\ncur.append([1, 2])",
        "v = cur[0]",
    ],
    "held by a function's defaults": ["data = [1, 2]\ndef f(x=data):\n    return x", "w = f()"],
    "held by a closure": [
        "def make():\n    held = [1, 2]\n    return held, lambda: held\npair = make()",
        "w = pair[0]",
    ],
    "held by a variable": ["data = [1, 2]\nboth = {'d': data}", "w = both['d']"],
    "held by a closure and a variable": [
        "full = {'pre': [1, 2]}",
        "keep = (lambda held: lambda: held)(full)",
        "gs = {'est': full, 'w': 2}",
    ],
}


@pytest.mark.parametrize("cells", SHAPES.values(), ids=SHAPES.keys())
def test_a_skipped_retry_would_have_been_thrown_away(cash_magics, monkeypatch, cells):
    """Asked again all the same, the check finds no anchor wherever the skip
    applies, and its answer is dropped for the first one's: what the skip
    leaves in place."""
    verdicts = []
    real_possible = processor._anchors_possible

    def possible(gave_up):
        verdicts.append(real_possible(gave_up))
        return True

    retries = []
    real_reason = processor.unrestorable_output_reason

    def reason(*args, **kwargs):
        result = real_reason(*args, **kwargs)
        if kwargs.get("anchorable") is not None:
            retries.append((result, dict(kwargs["anchors"])))
        return result

    monkeypatch.setattr(processor, "_anchors_possible", possible)
    monkeypatch.setattr(processor, "unrestorable_output_reason", reason)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert len(verdicts) == len(retries)
    for would_retry, (result, anchors) in zip(verdicts, retries):
        if not would_retry:
            assert result is not None or not anchors, "the retry found an anchor the skip would miss"


def test_the_retry_still_runs_where_a_variable_can_be_an_anchor(cash_magics, monkeypatch):
    """``full`` is held by a closure, which no restore rebinds, and by the
    variable ``full``: the retry stores it as that variable."""
    anchored = []
    real_reason = processor.unrestorable_output_reason

    def reason(*args, **kwargs):
        result = real_reason(*args, **kwargs)
        if kwargs.get("anchorable") is not None and result is None:
            anchored.append(set(kwargs["anchors"]))
        return result

    monkeypatch.setattr(processor, "unrestorable_output_reason", reason)
    cells = SHAPES["held by a closure and a variable"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert {"full"} in anchored


def test_the_retry_is_skipped_for_a_deep_list(cash_magics, monkeypatch):
    verdicts = []
    real_possible = processor._anchors_possible
    monkeypatch.setattr(
        processor, "_anchors_possible", lambda gave_up: verdicts.append(real_possible(gave_up)) or verdicts[-1]
    )
    cells = SHAPES["a list nested deeper than the climb"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert False in verdicts
