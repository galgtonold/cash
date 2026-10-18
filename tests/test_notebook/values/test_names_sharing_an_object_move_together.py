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


#: Something that is not a variable holds the list too, made before the dict
#: or object that holds it: the share check gives up on that holder, and still
#: links the variables it found.
BESIDE_A_HOLDER_NOT_A_VARIABLE = {
    "a function default": (
        "def pick(row, cols=features):\n    return cols",
        "config = {'features': features}",
        "config",
    ),
    "a class attribute": ("class Model:\n    cols = features", "config = {'features': features}", "config"),
    "an argparse namespace": ("import argparse", "args = argparse.Namespace(features=features)", "args"),
    "a deque": ("from collections import deque", "args = deque([features])", "args"),
}


@pytest.mark.parametrize("name", list(BESIDE_A_HOLDER_NOT_A_VARIABLE))
def test_a_holder_that_is_not_a_variable_still_links_the_variables(cash_magics, name):
    other, link, holder = BESIDE_A_HOLDER_NOT_A_VARIABLE[name]
    cells = ["features = ['a', 'b']\nunrelated = ['a', 'b']", other, link, "features.append('zip')"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells[:3]:
        run_cash_cell(cash_magics, cell, cells=cells)
    state = cash_magics.tracking_state
    lineage = state.variable_lineage
    before, control = lineage[holder], lineage["unrelated"]
    # Missed once in tens of thousands of runs (macOS, 3.14) and never
    # reproduced: say which half failed if it happens again.
    edges = {name: sorted(targets) for name, targets in state.derivation_edges.items()}
    changed = lineage["features"]
    run_cash_cell(cash_magics, cells[3], cells=cells)
    seen = f"edges before the change {edges}, features moved: {lineage['features'] != changed}"
    assert lineage[holder] != before, seen
    assert lineage["unrelated"] == control


def test_a_numpy_view_moves_when_its_base_changes(cash_magics):
    """``head = big[:5]``: ``head.base`` holds ``big``, which no count of
    ``head`` shows; ``big *= 2`` changes what ``head`` reads."""
    pytest.importorskip("numpy")
    cells = ["import numpy as np\nbig = np.arange(10.0)\nother = np.arange(5.0)", "head = big[:5]", "big *= 2"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    lineage = cash_magics.tracking_state.variable_lineage
    before, control = lineage["head"], lineage["other"]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert lineage["head"] != before
    assert lineage["other"] == control


def test_a_name_reached_through_several_partners_is_read_once(cash_magics, monkeypatch):
    """``voting.voting = "soft"`` moved on the four models the voting
    classifier holds through each list that held them too, and read each
    model again at every move: 2-3 s for a statement taking 0.1 ms. Each
    name is moved once per partner, and its content read once, after."""
    from cash.notebook.statement import lineage as lineage_module

    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [
        "class Model:\n    def __init__(self, n):\n        self.n = n",
        "a = Model(1)\nb = Model(2)",
        "models = [a, b]\nnamed = [('a', a), ('b', b)]",
        "holder = {'models': models, 'named': named}",
        "holder['models'][0].n = 5",
    ]
    for cell in cells[:4]:
        run_cash_cell(cash_magics, cell, cells=cells)
    before = dict(cash_magics.tracking_state.variable_lineage)
    reads = []
    tracker = next(
        v for v in vars(lineage_module).values() if isinstance(v, type) and "_update_variable_content_hashes" in vars(v)
    )
    real = tracker._update_variable_content_hashes
    monkeypatch.setattr(
        tracker,
        "_update_variable_content_hashes",
        lambda self, state, name, *rest, **kw: reads.append(name) or real(self, state, name, *rest, **kw),
    )
    run_cash_cell(cash_magics, cells[4], cells=cells)
    after = cash_magics.tracking_state.variable_lineage
    moved = {name for name in ("a", "b", "models", "named") if after[name] != before[name]}
    assert "a" in moved and "models" in moved
    assert all(reads.count(name) <= 1 for name in moved), reads
    assert cash_magics.shell.user_ns["a"].n == 5
