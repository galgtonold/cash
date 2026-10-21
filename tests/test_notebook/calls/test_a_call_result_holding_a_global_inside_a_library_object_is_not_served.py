"""A call's result holding a notebook object inside a library object is not
served as a copy.

``def mk(): fig = Figure(); fig.add_subplot().add_patch(rect); return
argparse.Namespace(fig=fig)``: the result holds ``rect`` inside a figure
inside a namespace, both library objects the walk does not read. A hit
handed back a copy whose patch was not ``rect``, so ``rect.set_width(5)``
no longer showed through ``w``. The statement around the call refused to
store it (what a library object holds is read down there); the call entry
did not look.
"""

from __future__ import annotations

import argparse
import time

import pytest

pytest.importorskip("matplotlib")

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from cash.notebook.call_interception import CallSite

RECT = Rectangle((0, 0), 1, 1)
ROWS = [1, 2]


def mk():
    time.sleep(0.05)  # above the call cost floor, so a miss is stored
    fig = Figure()
    fig.add_subplot().add_patch(RECT)
    return argparse.Namespace(fig=fig)


def wrap_rows():
    time.sleep(0.05)
    return argparse.Namespace(rows=ROWS)


def fresh():
    time.sleep(0.05)
    fig = Figure()
    fig.add_subplot().add_patch(Rectangle((0, 0), 1, 1))
    return argparse.Namespace(fig=fig)


def _twice(call_unit_harness, fn, name, between=None):
    unit = call_unit_harness(lineage={name: "hash-" + name}, user_ns={})
    site = CallSite(source=f"{name}()", free_names=frozenset({name}), occurrence_index=0)
    unit.begin_statement()
    unit.wrap(fn, site)()
    if between is not None:
        between()
    unit.begin_statement()
    return unit, unit.wrap(fn, site)()


def test_the_users_patch_inside_a_figure_inside_a_namespace_stays_the_users(call_unit_harness):
    def run_the_cell_again():
        # A Run All makes a new ``rect`` under the same lineage; an artist
        # cannot go on a second figure.
        global RECT
        RECT = Rectangle((0, 0), 1, 1)

    _unit, w = _twice(call_unit_harness, mk, "mk", run_the_cell_again)
    assert w.fig.axes[0].patches[0] is RECT


def test_the_users_list_inside_a_namespace_stays_the_users(call_unit_harness):
    _unit, w = _twice(call_unit_harness, wrap_rows, "wrap_rows")
    assert w.rows is ROWS


def test_a_fresh_figure_inside_a_namespace_is_still_served(call_unit_harness):
    unit, w = _twice(call_unit_harness, fresh, "fresh")
    assert [e["cache_hit"] for e in unit.drain()] == [False, True]
    assert w.fig.axes[0].patches[0].get_width() == 1
