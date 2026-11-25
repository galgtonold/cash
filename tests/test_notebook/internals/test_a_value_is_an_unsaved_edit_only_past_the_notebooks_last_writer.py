"""When a value built by code that is not in the notebook counts as an edit
the user ran and did not save.

The upstream check keeps such a value instead of rebuilding it from the
saved code. It is one only when the kernel has run the statement of the
notebook that writes the name last (the saved version, which the edit then
replaced), and the value is not one the check built itself from the
notebook. Asking for any statement of the notebook in the name's history
took ``x = 1`` as proof for an ``x`` that ``x = x + 100`` had made, after
that statement was edited to ``x = x + 200`` and saved: the old ``x`` was
kept, and the cell below computed from it.

Pins the rule itself; the notebook behaviour is in
``tests/test_notebook_integration/upstream/test_a_saved_edit_is_not_taken_for_one_not_saved.py``.
"""

from __future__ import annotations

from cash.notebook.upstream._types import TraceEntry
from tests._cell_driver import run_cash_cell

RAN = ["x = 1", "x = x + 100", "acc = []", "for i in range(2):\n    acc.append(i)"]


def _entry(code: str, *outputs: str) -> TraceEntry:
    return TraceEntry(code, set(outputs), set(), {}, {}, False)


def _is_unsaved_edit(magics, name: str, trace: list[TraceEntry]) -> bool:
    return magics._upstream_checker.simulator.classifier._ran_the_notebook_version(name, trace)


def _run(magics) -> None:
    for cell in RAN:
        run_cash_cell(magics, cell, cells=RAN)


def test_the_last_writer_ran(cash_magics):
    _run(cash_magics)
    assert _is_unsaved_edit(cash_magics, "x", [_entry("x = 1", "x"), _entry("x = x + 100", "x")])


def test_the_last_writer_did_not_run_though_one_above_it_did(cash_magics):
    _run(cash_magics)
    assert not _is_unsaved_edit(cash_magics, "x", [_entry("x = 1", "x"), _entry("x = x + 200", "x")])


def test_a_statement_below_that_does_not_write_the_name_is_passed_over(cash_magics):
    _run(cash_magics)
    trace = [_entry("x = 1", "x"), _entry("x = x + 100", "x"), _entry("y = 5", "y")]
    assert _is_unsaved_edit(cash_magics, "x", trace)


def test_a_loop_that_changes_the_name_in_place_leaves_it_to_the_writer_above(cash_magics):
    """The loop's run leaves no record under ``acc``: ``acc = []`` answers."""
    _run(cash_magics)
    trace = [_entry("acc = []", "acc"), _entry(RAN[3], "acc", "i")]
    assert _is_unsaved_edit(cash_magics, "acc", trace)
    assert not _is_unsaved_edit(cash_magics, "acc", [_entry("acc = [0]", "acc"), _entry(RAN[3], "acc", "i")])


def test_a_name_never_written_here_is_not_one(cash_magics):
    _run(cash_magics)
    assert not _is_unsaved_edit(cash_magics, "z", [_entry("z = 1", "z")])


def test_a_value_the_upstream_check_built_is_not_one(cash_magics):
    _run(cash_magics)
    state = cash_magics.tracking_state
    trace = [_entry("x = 1", "x"), _entry("x = x + 100", "x")]
    state.rebuilt_by_check["x"] = state.variable_lineage["x"]
    assert not _is_unsaved_edit(cash_magics, "x", trace)
    # Written again since, by a cell the user ran: the mark no longer holds.
    state.rebuilt_by_check["x"] = "the lineage the check left, moved on from since"
    assert _is_unsaved_edit(cash_magics, "x", trace)
