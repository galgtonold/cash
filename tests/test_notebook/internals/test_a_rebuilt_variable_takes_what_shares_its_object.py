"""Which variables a rebuild before a cell takes along (the closure over the
recorded shared-object edges in the mismatch classification)."""

from __future__ import annotations

from types import SimpleNamespace

from cash.notebook.upstream.mismatch_classifier import MismatchClassifier


def _closed(edges, broken, user_ns, trace_outputs, cells=None):
    fake = SimpleNamespace(
        tracking_state=SimpleNamespace(derivation_edges=edges), shell=SimpleNamespace(user_ns=user_ns)
    )
    trace = [SimpleNamespace(outputs=set(trace_outputs), inputs=set())]
    broken = set(broken)
    check = None if cells is None else SimpleNamespace(notebook_cells=cells, cell_code=None)
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
