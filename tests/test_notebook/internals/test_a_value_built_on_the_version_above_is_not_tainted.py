"""A value memory built on the version of a name the cells above give it is
not stale because the name's last version differs.

``lines = f.readlines()``, ``lines = split(lines)`` and ``parsed`` built from
the split lines, then the read again in a cell below that has not run since
a restart (JuNE student_2): the kernel holds the split ``lines``, the cells
above end with the read one. Everything built from ``lines`` was taken as
stale and rebuilt before every cell reading it, though memory built each of
them on the version a run from the top gives it there.
"""

from __future__ import annotations

from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.unsaved_edits import UnsavedEdits

READ = "lines = f.readlines()"
SPLIT = "for line in lines:\n    lines2.append(line)"
REBIND = "lines = lines2"
PARSE = "parsed = parse(lines)"
FRAME = "df = frame(parsed)"


def _trace():
    return [
        TraceEntry(READ, {"lines"}, {"f"}, {"f": "f1"}, {"lines": "read"}, False),
        TraceEntry("lines2 = []", {"lines2"}, set(), {}, {"lines2": "empty"}, False),
        TraceEntry(SPLIT, {"lines2", "line"}, {"lines", "lines2"}, {"lines": "read", "lines2": "empty"}, {}, False),
        TraceEntry(REBIND, {"lines"}, {"lines2"}, {"lines2": "filled"}, {"lines": "split"}, False),
        TraceEntry(PARSE, {"parsed"}, {"lines", "parse"}, {"lines": "split", "parse": "p"}, {}, False),
        TraceEntry(FRAME, {"df"}, {"parsed", "frame"}, {"parsed": "parsed", "frame": "fr"}, {}, False),
        TraceEntry(READ, {"lines"}, {"f"}, {"f": "f2"}, {"lines": "read"}, False),
    ]


def _memory(**built_on_lines: str) -> TrackingState:
    """The kernel after a restart that ran every cell but the read below."""
    state = TrackingState()
    state.lineage.record("lines", "split")
    state.executed_cell_codes.update({"lines": REBIND, "lines2": SPLIT, "line": SPLIT, "parsed": PARSE, "df": FRAME})
    state.executed_input_lineages.update(
        {
            "lines": {"lines2": "filled"},
            "lines2": {"lines": built_on_lines.get("lines2", "read"), "lines2": "empty"},
            "line": {"lines": built_on_lines.get("lines2", "read"), "lines2": "empty"},
            "parsed": {"lines": built_on_lines.get("parsed", "split"), "parse": "p"},
            "df": {"parsed": "parsed", "frame": "fr"},
        }
    )
    return state


def _tainted(state: TrackingState, trace=None) -> set[str]:
    trace = trace or _trace()
    return UnsavedEdits(state, statements=None).compute_tainted_vars_from_unsaved_edits(
        {"lines": "read"},
        trace,
        {entry.stmt_code for entry in trace},
        current_cell_idx=len(trace),
        notebook_cells=[entry.stmt_code for entry in trace],
    )


def test_values_built_on_the_versions_above_are_not_tainted():
    assert _tainted(_memory()) == set()


def test_a_value_built_on_another_version_is_tainted_with_what_follows():
    assert _tainted(_memory(parsed="an-unsaved-edit")) == {"parsed", "df"}


def test_a_loop_built_on_another_version_taints_what_it_fills():
    assert {"lines2", "line"} <= _tainted(_memory(lines2="an-unsaved-edit"))


def test_an_output_memory_took_from_a_statement_not_above_is_tainted():
    state = _memory()
    state.executed_cell_codes["parsed"] = "parsed = parse_differently(lines)"
    assert _tainted(state) == {"parsed", "df"}


def test_an_unrecorded_input_version_is_tainted():
    state = _memory()
    del state.executed_input_lineages["parsed"]
    assert _tainted(state) == {"parsed", "df"}


def test_a_name_tainted_through_another_is_not_judged_by_its_lineage():
    # ``df`` reads ``parsed``, tainted through ``lines``: its own matching
    # input lineage says nothing, so it stays tainted.
    state = _memory(parsed="an-unsaved-edit")
    state.executed_input_lineages["df"] = {"parsed": "parsed", "frame": "fr"}
    assert "df" in _tainted(state)


def _with_a_later_writer_of_parsed(code: str, inputs: set[str]):
    trace = _trace()
    trace.insert(5, TraceEntry(code, {"parsed"}, inputs, {}, {}, False))
    return trace


def test_an_output_a_statement_below_binds_anew_is_left_to_it():
    state = _memory(parsed="an-unsaved-edit")
    state.executed_cell_codes["parsed"] = "parsed = load()"
    trace = _with_a_later_writer_of_parsed("parsed = load()", {"load"})
    assert _tainted(state, trace) == set()


def test_an_output_a_statement_below_goes_on_changing_stays_tainted():
    # ``parsed.append(...)`` below changes the ``parsed`` built from another
    # version of ``lines``: memory still holds what that version made.
    state = _memory(parsed="an-unsaved-edit")
    state.executed_cell_codes["parsed"] = "parsed.append(extra)"
    trace = _with_a_later_writer_of_parsed("parsed.append(extra)", {"parsed", "extra"})
    assert _tainted(state, trace) == {"parsed", "df"}


def test_after_an_unsaved_edit_what_was_built_before_it_is_tainted():
    # The kernel's ``lines`` is one no statement above makes: an edit run
    # and not saved. The cells below must see what it changed, so values
    # built on the saved version are stale even when built on that version.
    state = _memory()
    state.lineage.record("lines", "split-as-edited")
    assert _tainted(state) >= {"lines2", "parsed", "df"}
