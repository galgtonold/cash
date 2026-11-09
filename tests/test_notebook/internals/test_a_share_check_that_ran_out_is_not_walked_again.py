"""A share check measured running out of its budget over the same objects,
with the same content, while the notebook binds the same values, is not
walked again: the statement is refused as the walk would refuse it.

``voting_clf.voting = 'soft'`` on a big fitted ensemble walks everything the
ensemble holds to find its other holders, runs out of its 0.1 s, and is
refused as re-running every time. The same statement on the same content
does the same walk and runs out of the same budget again. What a skipped
walk leaves is what a walk that ran out leaves: the same reason, the
statement not stored, the search from the outputs up for the variables to
link (`_holders_of`). Any change to the objects, their content, the other
variables or a larger budget walks again.
"""

from __future__ import annotations

import pytest

from cash import value_hash
from cash.notebook.statement import processor as processor_module
from cash.notebook.statement.processor import StatementProcessor
from tests._cell_driver import run_cash_cell

# A chain of items, each a step of the walk of its own.
_BOX = (
    "class Item:\n"
    "    def __init__(s, i, after=None):\n"
    "        s.i = i\n"
    "        s.after = after\n"
    "class Box:\n"
    "    def __init__(s, n):\n"
    "        s.items = []\n"
    "        after = None\n"
    "        for i in range(n):\n"
    "            after = Item(i, after)\n"
    "            s.items.append(after)\n"
    "        s.flag = 0\n"
)
_REASON = "finding which other variables hold objects of its outputs would take longer than"


@pytest.fixture
def spent(cash_magics, monkeypatch):
    """A notebook with ``box``, more objects than the walk takes before it
    first looks at its budget, and a budget spent at once, so every share
    check runs out. Gives ``(magics, walks, refusals)``: the outputs of
    each share check walked, and ``(code, reason, share_unchecked,
    share_gave_up)`` of each statement refused."""
    m = cash_magics
    # The content digest the check is told the same by, not cut short by a
    # loaded machine (`budgeted_hash`).
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", 60.0)
    run_cash_cell(m, _BOX)
    run_cash_cell(m, "box = Box(2000)")
    monkeypatch.setattr(processor_module, "SHARE_CHECK_FLOOR_S", 1e-9)
    monkeypatch.setattr(processor_module, "SHARE_CHECK_FACTOR", 0.0)
    walks: list[frozenset[str]] = []
    real = processor_module.unrestorable_output_reason

    def walk(outputs, *args, **kwargs):
        walks.append(frozenset(outputs))
        return real(outputs, *args, **kwargs)

    monkeypatch.setattr(processor_module, "unrestorable_output_reason", walk)
    refusals: list[tuple] = []
    note = StatementProcessor._note_share_refusal

    def noted(self, run, reason):
        if reason is not None:
            refusals.append((run.code, reason, run.share_unchecked, run.share_gave_up))
        return note(self, run, reason)

    monkeypatch.setattr(StatementProcessor, "_note_share_refusal", noted)
    return m, walks, refusals


def _box_walks(walks):
    return sum(1 for outputs in walks if "box" in outputs)


def _ran_out(refusals) -> bool:
    """Whether the first ``box.flag`` statement's check ran out, which the
    tests below need to mean anything."""
    return any(code.startswith("box.flag") and _REASON in reason for code, reason, *_ in refusals)


def test_the_same_check_on_the_same_content_is_not_walked_again(spent):
    m, walks, refusals = spent
    run_cash_cell(m, "box.flag = 1")
    assert _box_walks(walks) == 1
    run_cash_cell(m, "box.flag = 1")
    # Refused without the walk, as the walk refused it.
    assert _box_walks(walks) == 1
    assert len(refusals) == 2
    first, again = refusals
    assert _REASON in first[1]
    assert first[1:] == again[1:] == (first[1], True, None)
    assert m.shell.user_ns["box"].flag == 1


def test_changed_content_is_walked(spent):
    m, walks, refusals = spent
    run_cash_cell(m, "box.flag = 1")
    run_cash_cell(m, "box.flag = 2")
    assert _ran_out(refusals) and _box_walks(walks) == 2


def test_another_variable_bound_since_is_walked(spent):
    m, walks, refusals = spent
    run_cash_cell(m, "box.flag = 1")
    run_cash_cell(m, "first = box.items[0]")
    run_cash_cell(m, "box.flag = 1")
    assert _ran_out(refusals) and _box_walks(walks) == 2


def test_another_variable_changed_in_place_since_is_walked(spent):
    m, walks, refusals = spent
    run_cash_cell(m, "kept = []")
    run_cash_cell(m, "box.flag = 1")
    run_cash_cell(m, "kept.append(box.items[0])")
    run_cash_cell(m, "box.flag = 1")
    assert _ran_out(refusals) and _box_walks(walks) == 2


def test_a_larger_budget_is_walked(spent, monkeypatch):
    m, walks, refusals = spent
    run_cash_cell(m, "box.flag = 1")
    monkeypatch.setattr(processor_module, "SHARE_CHECK_FLOOR_S", 1e-6)
    run_cash_cell(m, "box.flag = 1")
    assert _ran_out(refusals) and _box_walks(walks) == 2


def test_a_holder_found_from_the_outputs_up_still_moves_its_reader(spent):
    """The skipped walk leaves the search from the outputs up to link
    ``items`` to ``box``, as the walk that ran out does: a change through
    ``box`` moves the reader of ``items`` on, which runs again, as in plain
    Python."""
    m, walks, refusals = spent
    run_cash_cell(m, "items = box.items")
    run_cash_cell(m, "box.flag = 1")
    run_cash_cell(m, "box.flag = 1")
    assert _ran_out(refusals) and _box_walks(walks) == 1
    run_cash_cell(m, "n = len(items)")
    assert m.shell.user_ns["n"] == 2000
    run_cash_cell(m, "box.items.append(Item(-1))")
    run_cash_cell(m, "n = len(items)")
    assert m.shell.user_ns["n"] == 2001


def test_the_memo_is_bounded_and_holds_no_value():
    from cash.notebook.statement.share_ran_out import ShareRanOut

    memo = ShareRanOut(3)
    for i in range(5):
        memo.note((("x", i, f"d{i}"),), 0.1, tuple)
    assert len(memo) == 3
    assert not memo.known((("x", 0, "d0"),), 0.1, tuple)
    assert memo.known((("x", 4, "d4"),), 0.1, tuple)
    assert not memo.known((("x", 4, "d4"),), 0.2, tuple)
    assert not memo.known(None, 0.1, tuple)
