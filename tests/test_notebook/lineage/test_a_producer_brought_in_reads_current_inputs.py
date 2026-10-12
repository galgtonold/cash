"""A statement a completion pass brings in gets the producer of an input the
kernel holds at another version.

``rows = [[SCALE, 2], [3, 4]]`` brought in for a shared list ran on the
``SCALE`` of an earlier edit: ``SCALE`` is bound once, so the shadowed-name
pass did not look, and the input pass trusts a live value matching no
producer.
"""

from __future__ import annotations

import types

from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.simulator import NotebookSimulator


def _planner(live: dict[str, str]):
    state = TrackingState()
    for name, lineage in live.items():
        state.lineage.record(name, lineage)
    return NotebookSimulator(types.SimpleNamespace(user_ns={}), None, state).planner


def _entry(stmt, outputs=(), inputs=(), consumed=None, produced=None):
    return TraceEntry(stmt, set(outputs), set(inputs), dict(consumed or {}), dict(produced or {}), False)


TRACE = [
    _entry("SCALE = 3", ("SCALE",), (), None, {"SCALE": "S3"}),  # 0
    _entry("rows = [[SCALE, 2], [3, 4]]", ("rows",), ("SCALE",), {"SCALE": "S3"}, {"rows": "R3"}),  # 1
]


def test_the_producer_of_an_input_at_another_version_is_scheduled():
    scheduled = _planner({"SCALE": "S2"})._complete_shadowed_var_producers([1], TRACE, {"SCALE": "S3", "rows": "R3"})
    assert scheduled == [0, 1], scheduled


def test_an_input_the_kernel_holds_at_its_version_is_left_alone():
    """Control: the same plan with ``SCALE`` current."""
    scheduled = _planner({"SCALE": "S3"})._complete_shadowed_var_producers([1], TRACE, {"SCALE": "S3", "rows": "R3"})
    assert scheduled == [1], scheduled


def test_a_loop_name_is_left_to_the_loop_rules():
    scheduled = _planner({"SCALE": "S2"})._complete_shadowed_var_producers(
        [1], TRACE, {"SCALE": "S3", "rows": "R3"}, loop_vars={"SCALE"}
    )
    assert scheduled == [1], scheduled
