"""A repair leaves the live value of a name a cell below bound when the cell never touches it.

``prepared = pipe.fit_transform(data)`` re-run to give the edited cell the
``pipe`` a run from the top has there binds ``prepared`` on the way. A cell
below bound ``prepared`` anew and the edited cell does not read it: the
cell's run is the same either way, and plain Jupyter keeps the live
``prepared`` afterwards. The repair puts it back
(``UpstreamChecker._keep_live_bindings``), also when the cell below repeats
the fit's text. A cell that reads ``prepared``, by name or by a string,
gets what a run from the top gives it there: the repaired value.
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
# The cell below repeats the fit's statement text, or fits another way.
BELOW = {"repeated": FIT + "\nrows = len(prepared)", "other text": FIT + " + []"}


@pytest.fixture
def magics(cash_magics):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    return cash_magics


def _run(m, cells, *idx):
    for i in idx:
        run_cash_cell(m, cells[i], cells=cells)


def _edit_reader(magics, below, reader):
    """Run the notebook, edit the reader cell to *reader*, run it alone.
    Returns the namespace and the live ``prepared`` before the edit."""
    ns = magics.shell.user_ns
    cells = [PIPE, "data = [1]", "pipe = Pipe(3)", FIT, "reader = pipe.k", "data = [1, 2]", below]
    _run(magics, cells, *range(len(cells)))
    live = ns["prepared"]
    assert live == [3, 6]
    cells[4] = reader
    _run(magics, cells, 4)
    return ns, live


@pytest.mark.parametrize("below", sorted(BELOW))
def test_a_name_the_cell_does_not_read_keeps_its_live_value(magics, below):
    ns, live = _edit_reader(magics, BELOW[below], "reader = [pipe.k, len(pipe.fitted)]")
    assert ns["reader"] == [3, 1]  # the pipe a run from the top has there
    assert ns["prepared"] is live  # what plain Jupyter has after the cell


@pytest.mark.parametrize("below", sorted(BELOW))
def test_without_the_put_back_the_repair_leaves_the_name_from_above(magics, monkeypatch, below):
    """Guard for the test above: the repair does rebind ``prepared``, so
    that test is what proves the put-back."""
    monkeypatch.setattr(UpstreamChecker, "_keep_live_bindings", lambda self, *a, **k: None)
    ns, _ = _edit_reader(magics, BELOW[below], "reader = [pipe.k, len(pipe.fitted)]")
    assert ns["reader"] == [3, 1]
    assert ns["prepared"] == [3]


READS = {
    "by name": "reader = [pipe.k, list(prepared)]",
    "by a string": "reader = [pipe.k, list(globals()['prepared'])]",
    "in a function": "def show():\n    return list(prepared)\nreader = [pipe.k, show()]",
}


@pytest.mark.parametrize("read", sorted(READS))
@pytest.mark.parametrize("below", sorted(BELOW))
def test_a_cell_reading_the_name_gets_the_repaired_value(magics, below, read):
    ns, _ = _edit_reader(magics, BELOW[below], READS[read])
    assert ns["reader"] == [3, [3]]  # what a run from the top gives the cell


def test_a_cell_binding_the_name_keeps_its_own_value(magics):
    ns, _ = _edit_reader(magics, BELOW["repeated"], "reader = pipe.k\nprepared = [0]")
    assert ns["prepared"] == [0]
