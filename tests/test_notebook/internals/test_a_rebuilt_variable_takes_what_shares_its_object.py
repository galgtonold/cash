"""Which variables a rebuild before a cell takes along (the closure over the
recorded shared-object edges in the mismatch classification)."""

from __future__ import annotations

from types import SimpleNamespace

from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream._types import CellCheck
from cash.notebook.upstream.mismatch_classifier import MismatchClassifier


def _closed(edges, broken, user_ns, trace_outputs, cells=None):
    state = TrackingState()
    state.derivation_edges.update(edges)
    fake = SimpleNamespace(tracking_state=state, shell=SimpleNamespace(user_ns=user_ns))
    trace = [SimpleNamespace(outputs=set(trace_outputs), inputs=set())]
    broken = set(broken)
    # The first cell is the one running.
    check = None if cells is None else CellCheck(current_cell_idx=0, notebook_cells=cells)
    MismatchClassifier._rebuild_what_shares_a_rebuilt_object(fake, broken, trace, check)
    return broken


def test_the_closure_goes_both_ways_and_through_partners():
    edges = {"df": {"g"}, "data": {"raw"}, "raw": {"data", "view"}}
    ns = dict.fromkeys(["df", "g", "data", "raw", "view", "other"], 0)
    assert _closed(edges, {"df"}, ns, ns) == {"df", "g"}
    assert _closed(edges, {"data"}, ns, ns) == {"data", "raw", "view"}
    assert _closed(edges, {"other"}, ns, ns) == {"other"}


def test_a_holder_no_statement_above_makes_keeps_the_group_live():
    """``g`` made in no statement above: rebuilt, ``df`` would leave it on the
    old frame. The group runs on the live objects, as a plain kernel does."""
    edges = {"df": {"g"}}
    ns = {"df": 0, "g": 0, "x": 0}
    assert _closed(edges, {"df", "x"}, ns, {"df", "x"}) == {"x"}


def test_a_holder_a_cell_makes_is_rebuilt_with_it():
    """``y = (w, v)`` in the running cell, or a cell below, makes ``y``
    again: not a holder to keep live."""
    edges = {"w": {"y"}, "y": {"w"}}
    ns = {"w": 0, "y": 0}
    assert _closed(edges, {"w"}, ns, {"w"}, cells=["w = [1]", "y = (w, 2)"]) == {"w", "y"}


def _restore_check(edges, kept, user_ns, needed=(), resolved=()):
    from cash.notebook.upstream.mismatch_classifier import _BackwardScan

    state = TrackingState()
    state.derivation_edges.update(edges)
    fake = SimpleNamespace(tracking_state=state, shell=SimpleNamespace(user_ns=user_ns))
    scan = _BackwardScan(needed=set(needed), resolved=set(resolved))
    return MismatchClassifier._restore_keeps_sharing(fake, kept, scan)


def test_a_restore_giving_a_kept_variable_a_new_object_needs_all_that_shares_it():
    """``full``'s entry stores ``pre`` and ``lin`` with it; ``late``, made
    below ``full`` on ``pre`` and not rebuilt, would keep the old ``pre``."""
    edges = {"pre": {"full", "lin", "late"}, "full": {"pre"}, "lin": {"pre"}, "late": {"pre"}}
    ns = dict.fromkeys(["pre", "full", "lin", "late"], 0)
    kept = {"pre", "lin"}
    assert not _restore_check(edges, kept, ns)({"pre", "lin"}, {"full"})
    assert kept == {"pre", "lin"}
    assert _restore_check(edges, kept, ns, resolved={"late"})({"pre", "lin"}, {"full"})
    # Restored, they have new objects: no longer kept live.
    assert kept == set()


def test_a_restore_storing_no_kept_variable_is_taken():
    assert _restore_check({"a": {"b"}}, {"pre"}, {"a": 0, "b": 0})({"b"}, {"a"})


def _plan_fallback(trace, scheduled, restored, kept, edges, user_ns):
    from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner

    state = TrackingState()
    state.derivation_edges.update(edges)
    fake = SimpleNamespace(tracking_state=state, shell=SimpleNamespace(user_ns=user_ns))
    return ReexecutionPlanner.rerun_what_changes_kept_sharers(fake, scheduled, trace, restored, kept)


def _entry(code, outputs):
    # Each output moves to a lineage of its own: the statement changes it.
    lineages = {name: f"{code}->{name}" for name in outputs}
    return SimpleNamespace(stmt_code=code, outputs=set(outputs), inputs=set(), produced_lineages=lineages)


def test_a_plan_changing_a_kept_variable_rebuilds_it_and_what_holds_it():
    """A later pass scheduled ``pre.fit(d)``: ``pre`` and ``lin``, kept live,
    are made again from their bindings, and ``full``'s restore, which bound
    the old ``pre``, runs instead."""
    trace = [
        _entry("pre = Pre()", {"pre"}),
        _entry("lin = Pipe([pre])", {"lin"}),
        _entry("lin.fit(d)", {"lin", "pre"}),
        _entry("full = Pipe([pre])", {"full"}),
        _entry("pre.fit(d)", {"pre", "lin", "full"}),
    ]
    edges = {"pre": {"lin", "full"}, "lin": {"pre"}, "full": {"pre"}}
    ns = dict.fromkeys(["pre", "lin", "full"], 0)
    kept = {"pre", "lin"}
    restored = [{"code": "full = Pipe([pre])", "restored_vars": ["full"], "position": 3}]
    assert _plan_fallback(trace, [4], restored, kept, edges, ns) == [0, 1, 2, 3, 4]
    assert kept == set()


def test_a_plan_leaving_kept_variables_alone_is_unchanged():
    trace = [_entry("pre = Pre()", {"pre"}), _entry("full = Pipe([pre])", {"full"})]
    kept = {"pre"}
    assert _plan_fallback(trace, [1], [], kept, {"pre": {"full"}}, {"pre": 0, "full": 0}) == [1]
    assert kept == {"pre"}


def test_an_output_left_at_its_lineage_is_not_a_change():
    """``grid.fit(X, y)`` lists ``pre``, which its search holds, and leaves it
    at the lineage ``lin.fit`` gave it."""
    from cash.notebook.upstream.mismatch_classifier import what_each_statement_changes

    trace = [
        SimpleNamespace(outputs={"pre"}, produced_lineages={"pre": "p0"}),
        SimpleNamespace(outputs={"lin", "pre"}, produced_lineages={"lin": "l1", "pre": "p1"}),
        SimpleNamespace(outputs={"grid", "pre"}, produced_lineages={"grid": "g1", "pre": "p1"}),
        SimpleNamespace(outputs={"x"}, produced_lineages={}),
    ]
    assert what_each_statement_changes(trace) == [{"pre"}, {"lin", "pre"}, {"grid"}, {"x"}]
