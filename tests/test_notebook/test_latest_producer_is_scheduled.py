"""Every scheduled statement gets the LATEST producer of each input before it.

Round 25's r25s2 (WRONG, silent): after a restart, running only the export cell
rebuilt the cleaning cell without ``sales['refund'] = is_refund.astype(int)``,
and the summary showed 0 refunds for three stores. The trace showed the plan
schedule every statement of the cell but that one: completing
``sales['quantity'] = sales['quantity'].abs()``, the planner found
``sales['timestamp'] = ...`` already scheduled and took it as the producer of
``sales``, though the refund write sits between them.

The notebook arm is
``tests/test_notebook_integration/test_a_rebuilt_cleaning_cell_keeps_every_column_write.py``;
the tester's own repro (3 million rows) is the one that reached this state.
"""
from __future__ import annotations

import types

from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner


def _planner(user_ns: dict) -> ReexecutionPlanner:
    vl = types.SimpleNamespace(shell=types.SimpleNamespace(user_ns=user_ns))
    return ReexecutionPlanner(vl, classifier=None, debug=False)


def _entry(stmt, outputs=(), inputs=()):
    return (stmt, set(outputs), list(inputs), {}, {}, None)


CLEANING = [
    _entry("sales = raw.drop_duplicates()", ("sales",), ("raw",)),                          # 0
    _entry("sales['timestamp'] = pd.to_datetime(sales['timestamp'])", ("sales",), ("sales", "pd")),  # 1
    _entry("is_refund = (sales['refund'] == 1) | (sales['quantity'] < 0)", ("is_refund",), ("sales",)),  # 2
    _entry("sales['refund'] = is_refund.astype(int)", ("sales",), ("sales", "is_refund")),   # 3
    _entry("sales['quantity'] = sales['quantity'].abs()", ("sales",), ("sales",)),           # 4
    _entry("weekly = sales.groupby('store').refund.sum()", ("weekly",), ("sales",)),          # 5
]


def test_a_write_between_a_scheduled_producer_and_its_reader_is_scheduled():
    planner = _planner({"raw": object(), "pd": object()})
    scheduled = planner._complete_inputs_produced_before([1, 4, 5], CLEANING)
    assert 3 in scheduled, "the refund write between the timestamp write and its reader was left out"
    assert scheduled == [0, 1, 2, 3, 4, 5]


def test_a_live_input_still_needs_no_producer():
    planner = _planner({"raw": object(), "pd": object(), "sales": object(), "is_refund": object()})
    assert planner._complete_inputs_produced_before([5], CLEANING) == [5]
