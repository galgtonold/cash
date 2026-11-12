"""A repair started from a cell below records the cell of each name it binds.

A check running a cell below first repairs what the cells above it leave,
and the repair runs with no cell running yet. ``prepared =
pipe.fit_transform(data)`` re-run there for a later section was recorded as
bound by no cell. An edit of a cell above that section then re-runs the fit
above to give the edited cell its ``pipe``, binding ``prepared`` on the way,
and the put-back of a name a cell below bound
(``UpstreamChecker._keep_live_bindings``) could not tell the later section
had bound it: ``prepared`` stood at the older value from above, where a run
from the top leaves what the later section makes. The repair now records the
cell of the statement that bound each name, as a run of that cell would.
"""

from __future__ import annotations

import pytest

from cash.notebook.upstream.checker import UpstreamChecker
from tests._cell_driver import run_cash_cell

PIPE = """class Pipe:
    def __init__(self, k):
        self.k = k
        self.fitted = None

    def fit_transform(self, data):
        self.fitted = list(data)
        return [x * self.k for x in data]"""

FIT = "prepared = pipe.fit_transform(data)"
# The later section repeats the fit's statement text, or fits another way.
BELOW = {"repeated": FIT + "\nrows = len(prepared)", "other text": FIT + " + []"}
READER = "reader = [pipe.k, len(pipe.fitted)]"


@pytest.fixture
def magics(cash_magics):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    return cash_magics


def _plain(cells: list[str]) -> dict:
    ns: dict = {}
    for src in cells:
        exec(compile(src, "<plain>", "exec"), ns)
    return ns


def _repair_below_then_edit_above(magics, below):
    """Run the notebook, lose ``prepared`` so that the last cell's check
    rebuilds it from the later section, then edit the reader cell above that
    section and run it alone. Returns the namespace and the cells."""
    ns = magics.shell.user_ns
    cells = [
        PIPE,
        "data = [1]",
        "pipe = Pipe(3)",
        FIT,
        "reader = pipe.k",
        "data = [1, 2]",
        below,
        "out = len(prepared)",
    ]
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    # What a scratch cell's ``del prepared`` leaves: the last cell's check
    # repairs it from the later section, above the last cell.
    del ns["prepared"]
    run_cash_cell(magics, cells[7], cells=cells)
    assert ns["prepared"] == [3, 6]
    cells[4] = READER
    run_cash_cell(magics, cells[4], cells=cells)
    return ns, cells


@pytest.mark.parametrize("below", sorted(BELOW))
def test_an_edit_above_keeps_what_the_later_section_bound(magics, below):
    ns, cells = _repair_below_then_edit_above(magics, BELOW[below])
    plain = _plain(cells)
    assert ns["reader"] == plain["reader"] == [3, 1]  # the pipe a run from the top has there
    assert ns["prepared"] == plain["prepared"] == [3, 6]  # what the later section made


@pytest.mark.parametrize("below", sorted(BELOW))
def test_without_the_record_the_value_from_above_stands(magics, monkeypatch, below):
    """Guard for the test above: without the record the repair leaves
    ``prepared`` from the fit above, so that test is what proves it."""
    monkeypatch.setattr(UpstreamChecker, "_place_repair_bindings", lambda self, *a, **k: None)
    ns, _ = _repair_below_then_edit_above(magics, BELOW[below])
    assert ns["reader"] == [3, 1]
    assert ns["prepared"] == [3]


def test_the_repair_records_the_cell_of_the_later_section(magics):
    _repair_below_then_edit_above(magics, BELOW["repeated"])
    state = magics.tracking_state
    assert state.bound_in_cell["prepared"] == 6
    assert state.bound_by["prepared"] == FIT
