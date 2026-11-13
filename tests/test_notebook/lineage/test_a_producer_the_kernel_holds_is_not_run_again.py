"""A completion pass does not bring in a producer whose value the kernel holds.

``data = df_final`` re-run reads ``df_final`` as ``df_final = df.drop(...)``
makes it, and ``data['date'] = ...`` below changes both names. The backward
scan restored the drop's entry, so the kernel holds that version: re-running
the drop makes what is already there, and brought in every statement
building ``df`` above it. Both passes that brought it in -- the one for a
name bound more than once and the one for an initializer behind a
statement that changes a name without binding it -- leave it out while the
kernel holds its version, and bring it in when the kernel holds another,
or when a name it reads is remade.
"""

from __future__ import annotations

import types

from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.simulator import NotebookSimulator


def _planner(live: dict[str, str], bound: set[str] | None = None):
    """A planner whose kernel binds *bound* (every name of *live* by
    default), each at its lineage in *live*."""
    state = TrackingState()
    for name, lineage in live.items():
        state.lineage.record(name, lineage)
    user_ns = {name: object() for name in (live if bound is None else bound)}
    return NotebookSimulator(types.SimpleNamespace(user_ns=user_ns), None, state).planner


def _entry(stmt, outputs=(), inputs=(), consumed=None, produced=None):
    return TraceEntry(stmt, set(outputs), set(inputs), dict(consumed or {}), dict(produced or {}), False)


TRACE = [
    _entry("df = load()", ("df",), ("load",), {"load": "LD"}, {"df": "D1"}),  # 0
    _entry("df = df[df.x > 0]", ("df",), ("df",), {"df": "D1"}, {"df": "D2"}),  # 1
    _entry("df_final = df.drop(columns=['s'])", ("df_final",), ("df",), {"df": "D2"}, {"df_final": "F1"}),  # 2
    _entry("data = df_final", ("data",), ("df_final",), {"df_final": "F1"}, {"data": "A1"}),  # 3
    _entry(
        "data['date'] = data['x'] * 10",
        ("data", "df_final"),
        ("data",),
        {"data": "A1"},
        {"data": "A2", "df_final": "F2"},
    ),  # 4
]
FINAL = {"df": "D2", "df_final": "F2", "data": "A2"}


def test_the_shadowed_name_pass_leaves_out_a_producer_the_kernel_holds():
    scheduled = _planner({"df_final": "F1", "data": "A0", "df": "D0"})._complete_shadowed_var_producers(
        [3, 4], TRACE, FINAL
    )
    assert scheduled == [3, 4], scheduled


def test_the_shadowed_name_pass_brings_in_a_producer_the_kernel_does_not_hold():
    """Control: the kernel holds ``df_final`` at another version."""
    scheduled = _planner({"df_final": "F0", "data": "A0", "df": "D0"})._complete_shadowed_var_producers(
        [3, 4], TRACE, FINAL
    )
    assert 2 in scheduled, scheduled


def test_a_lineage_for_a_name_the_kernel_does_not_bind_brings_the_producer_in():
    """The simulation records an import's lineage for a name the kernel
    does not bind (after a restart): that is not a value the kernel holds."""
    planner = _planner({"df_final": "F1", "data": "A0", "df": "D0"}, bound={"data", "df"})
    assert 2 in planner._complete_shadowed_var_producers([3, 4], TRACE, FINAL)
    assert 2 in planner._schedule_conditional_producer_inits([3, 4], TRACE, {"df_final", "data"})


def test_the_initializer_pass_leaves_out_a_producer_the_kernel_holds():
    scheduled = _planner({"df_final": "F1"})._schedule_conditional_producer_inits([3, 4], TRACE, {"df_final", "data"})
    assert scheduled == [3, 4], scheduled


def test_the_initializer_pass_brings_in_a_producer_the_kernel_does_not_hold():
    """Control: the kernel holds ``df_final`` at another version."""
    scheduled = _planner({"df_final": "F0"})._schedule_conditional_producer_inits([3, 4], TRACE, {"df_final", "data"})
    assert scheduled == [2, 3, 4], scheduled


# The cell below draws its chart after setting ``date`` again, a second
# change of both names the plan re-runs after the first.
TWICE = [
    *TRACE,
    _entry(
        "data['date'] = data['x'] * 10",
        ("data", "df_final"),
        ("data",),
        {"data": "A2"},
        {"data": "A3", "df_final": "F3"},
    ),  # 5
]


def test_a_change_the_plan_re_runs_after_the_producer_does_not_bring_it_in():
    planner = _planner({"df_final": "F1"})
    assert planner._schedule_conditional_producer_inits([3, 4, 5], TWICE, {"df_final", "data"}) == [3, 4, 5]
    shadowed = planner._complete_shadowed_var_producers([3, 4, 5], TWICE, {"df": "D2", "df_final": "F3", "data": "A3"})
    assert shadowed == [3, 4, 5], shadowed


def test_a_statement_re_run_before_the_producer_brings_it_in():
    """Control: ``df_final = seed()`` re-runs first and changes what the kernel holds."""
    seeded = [
        _entry("df_final = seed()", ("df_final",), ("seed",), {"seed": "SD"}, {"df_final": "F0"}),
        *TRACE,
    ]
    scheduled = _planner({"df_final": "F1"})._complete_shadowed_var_producers([0, 4, 5], seeded, FINAL)
    assert 3 in scheduled, scheduled


def test_a_producer_reading_a_name_the_scan_restored_is_brought_in():
    """The scan restored ``df`` as a new object: ``df_final`` may hold the old one."""
    planner = _planner({"df_final": "F1", "data": "A0", "df": "D2"})
    shadowed = planner._complete_shadowed_var_producers([3, 4], TRACE, FINAL, restored=frozenset({"df"}))
    assert 2 in shadowed, shadowed
    inits = planner._schedule_conditional_producer_inits(
        [3, 4], TRACE, {"df_final", "data"}, restored=frozenset({"df"})
    )
    assert inits == [2, 3, 4], inits
