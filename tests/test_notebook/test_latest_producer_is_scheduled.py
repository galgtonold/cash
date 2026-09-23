"""Every scheduled statement gets the LATEST producer of each input before it.

A silent wrong answer: after a restart, running only the export cell
rebuilt the cleaning cell without ``sales['refund'] = is_refund.astype(int)``,
and the summary showed 0 refunds for three stores. The trace showed the plan
schedule every statement of the cell but that one: completing
``sales['quantity'] = sales['quantity'].abs()``, the planner found
``sales['timestamp'] = ...`` already scheduled and took it as the producer of
``sales``, though the refund write sits between them.

The notebook arm is
``tests/test_notebook_integration/test_a_rebuilt_cleaning_cell_keeps_every_column_write.py``;
the original repro (3 million rows) is the one that reached this state.
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
    _entry("sales = raw.drop_duplicates()", ("sales",), ("raw",)),  # 0
    _entry("sales['timestamp'] = pd.to_datetime(sales['timestamp'])", ("sales",), ("sales", "pd")),  # 1
    _entry("is_refund = (sales['refund'] == 1) | (sales['quantity'] < 0)", ("is_refund",), ("sales",)),  # 2
    _entry("sales['refund'] = is_refund.astype(int)", ("sales",), ("sales", "is_refund")),  # 3
    _entry("sales['quantity'] = sales['quantity'].abs()", ("sales",), ("sales",)),  # 4
    _entry("weekly = sales.groupby('store').refund.sum()", ("weekly",), ("sales",)),  # 5
]


def test_a_write_between_a_scheduled_producer_and_its_reader_is_scheduled():
    planner = _planner({"raw": object(), "pd": object()})
    scheduled = planner._complete_inputs_produced_before([1, 4, 5], CLEANING)
    assert 3 in scheduled, "the refund write between the timestamp write and its reader was left out"
    assert scheduled == [0, 1, 2, 3, 4, 5]


def test_a_live_input_still_needs_no_producer():
    planner = _planner({"raw": object(), "pd": object(), "sales": object(), "is_refund": object()})
    assert planner._complete_inputs_produced_before([5], CLEANING) == [5]


# Run the export cell (adds `results["f1"] = ...` in place),
# re-run the sweep cell (rebinds `results`, no f1), then the chart cell. The
# plan re-ran `best = results.sort_values(['f1', ...])` without the f1 write
# above it -- `results` was live, so it needed no producer -- and raised
# UpstreamStateError: 'f1'. Live is not enough when the live value is not what
# the latest producer made: the sweep's table, not the one with f1.
def _lineaged(stmt, outputs=(), inputs=(), produced=None):
    return (stmt, set(outputs), list(inputs), {}, dict(produced or {}), None)


SWEEP_THEN_PICK = [
    _lineaged("results = pd.DataFrame(rows)", ("results",), ("pd", "rows"), {"results": "L-sweep"}),  # 0
    _lineaged("results['f1'] = 2 * results.precision", ("results",), ("results",), {"results": "L-f1"}),  # 1
    _lineaged("best = results.sort_values(['f1'])", ("best",), ("results",), {"best": "L-best"}),  # 2
]


def _planner_with_lineage(user_ns, lineage):
    vl = types.SimpleNamespace(shell=types.SimpleNamespace(user_ns=user_ns), variable_lineage=lineage)
    return ReexecutionPlanner(vl, classifier=None, debug=False)


def test_a_live_input_that_is_not_what_its_latest_producer_made_gets_that_producer():
    planner = _planner_with_lineage({"results": object(), "pd": object(), "rows": []}, {"results": "L-sweep"})
    assert planner._complete_inputs_produced_before([2], SWEEP_THEN_PICK) == [1, 2]


def test_a_live_input_its_latest_producer_made_needs_nothing():
    planner = _planner_with_lineage({"results": object(), "pd": object(), "rows": []}, {"results": "L-f1"})
    assert planner._complete_inputs_produced_before([2], SWEEP_THEN_PICK) == [2]


def test_a_live_input_matching_no_producer_keeps_the_old_answer():
    """A per-iteration loop's ``u = u + 0.01``: live ``u`` is none of the
    versions the trace recorded, which is no evidence it is behind."""
    planner = _planner_with_lineage({"results": object(), "pd": object(), "rows": []}, {"results": "L-something-else"})
    assert planner._complete_inputs_produced_before([2], SWEEP_THEN_PICK) == [2]
