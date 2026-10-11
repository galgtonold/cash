"""Two names for one object move together.

``data = raw``, ``config = {'features': features}``, ``train =
data['train']``: a statement that leaves two variables sharing an object links
them both ways, and a later change through either one gives the other a new
lineage, so the cells reading it are not served the value from before the
change. A statement that only rebinds a name never moves the names it used to
share an object with.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import derivation_edges
from tests._cell_driver import run_cash_cell


def _bump(edges, lineage, outputs, inputs, skip=()):
    return derivation_edges.bump_derived_lineages(
        edges, lineage, set(outputs), set(inputs), record=lineage.__setitem__, present=lambda _: True, skip=skip
    )


def test_a_shared_object_links_both_ways():
    edges: dict[str, set[str]] = {}
    derivation_edges.record_shared_object_edges(edges, {"data"}, {"raw": "l1", "data": "l1"})
    assert edges == {"data": {"raw"}, "raw": {"data"}}
    assert derivation_edges.shared_object_partners(edges, "raw") == {"data"}
    assert derivation_edges.shared_object_partners(edges, "data") == {"raw"}


def test_a_change_through_one_name_moves_the_other():
    edges = {"data": {"raw"}, "raw": {"data"}}
    lineage = {"raw": "r1", "data": "d1"}
    assert _bump(edges, lineage, {"raw"}, {"raw"}) == {"data"}
    assert lineage["data"] != "d1"


def test_a_rebinding_moves_nothing():
    """``raw = [9]`` does not read ``raw``: the old list is gone from that
    name, and ``data`` still holds it unchanged."""
    edges = {"data": {"raw"}, "raw": {"data"}}
    lineage = {"raw": "r1", "data": "d1"}
    assert _bump(edges, lineage, {"raw"}, set()) == set()
    assert lineage["data"] == "d1"


def test_an_output_of_the_statement_is_never_moved_again():
    edges = {"a": {"b"}, "b": {"a"}}
    lineage = {"a": "a1", "b": "b1"}
    assert _bump(edges, lineage, {"a", "b"}, {"a"}) == set()
    assert lineage == {"a": "a1", "b": "b1"}


def test_a_name_the_statement_already_moved_is_skipped():
    """A holder a hit restores with the statement already has its lineage."""
    edges = {"m": {"models"}, "models": {"m"}}
    lineage = {"m": "m1", "models": "x1"}
    assert _bump(edges, lineage, {"m"}, {"m"}, skip={"models"}) == set()
    assert lineage["models"] == "x1"


@pytest.mark.parametrize(
    "link, change, other",
    [
        ("data = raw", "raw.append(4)", "data"),
        ("config = {'features': features}", "features.append('zip')", "config"),
        ("config = {'features': features}", "config['features'].append('zip')", "features"),
        ("t = (features,)", "t[0].append('zip')", "features"),
    ],
    ids=["alias", "held by a dict", "changed through the dict", "held by a tuple"],
)
def test_a_statement_changing_one_name_moves_the_other(cash_magics, link, change, other):
    cells = ["raw = [1]\nfeatures = ['age']", link, change]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    before = cash_magics.tracking_state.variable_lineage[other]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.tracking_state.variable_lineage[other] != before


def test_an_unrelated_name_does_not_move(cash_magics):
    """Control: ``copy`` holds a copy, not the list."""
    cells = ["raw = [1]", "copy = list(raw)", "raw.append(4)"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    before = cash_magics.tracking_state.variable_lineage["copy"]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.tracking_state.variable_lineage["copy"] == before
