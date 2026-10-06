"""The upstream simulation drops only the names a ``%reset`` magic deletes.

A name the simulation drops falls back to its live lineage, so an edit above
the reset cell is not seen as a change and a cell below runs on the old
value. ``%reset out`` flushes only the output history, and
``%reset_selective`` deletes only the names its pattern matches. ``%xdel x``
(or ``get_ipython().run_line_magic("xdel", "x")``) deletes ``x``, which came
back from the cache instead of raising NameError.
"""

from __future__ import annotations

import pytest

from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream import NotebookSimulator


def _after(mock_shell, cash_instance, mid):
    cells = ["x = 1\ntmp_a = 2", mid, "y = x + 1"]
    simulation = NotebookSimulator(mock_shell, cash_instance, TrackingState()).virtual_lineage
    return set(simulation.simulate(2, cells).virtual_lineage)


@pytest.mark.parametrize(
    "mid",
    ["pass", "%reset -f out", "%reset in", "%reset -f dhist", "%reset -f array", "%reset_selective -f zzz"],
)
def test_a_reset_that_deletes_no_variable_keeps_them(mock_shell, cash_instance, mid):
    assert {"x", "tmp_a"} <= _after(mock_shell, cash_instance, mid)


@pytest.mark.parametrize("mid", ["%reset -f", "%reset", "%reset -f --aggressive", "%reset -sf out"])
def test_a_full_reset_drops_every_name(mock_shell, cash_instance, mid):
    assert not {"x", "tmp_a"} & _after(mock_shell, cash_instance, mid)


def test_reset_selective_drops_the_names_its_pattern_matches(mock_shell, cash_instance):
    names = _after(mock_shell, cash_instance, "%reset_selective -f ^tmp_")
    assert "x" in names
    assert "tmp_a" not in names


@pytest.mark.parametrize(
    "mid",
    ["%xdel tmp_a", "x = 3\n%xdel tmp_a", "get_ipython().run_line_magic('xdel', 'tmp_a')"],
    ids=["magic", "magic-after-python", "hand-written"],
)
def test_xdel_drops_the_name_it_deletes(mock_shell, cash_instance, mid):
    names = _after(mock_shell, cash_instance, mid)
    assert "x" in names
    assert "tmp_a" not in names


def test_xdel_drops_only_that_name(mock_shell, cash_instance):
    assert {"x", "tmp_a"} <= _after(mock_shell, cash_instance, "%xdel tmp")
