"""An Axes is part of its Figure: drawing on ``ax`` changes what ``fig.savefig`` writes.

Until round 21 a chart re-drew after an upstream edit only because ``fig`` also
drifted for no reason (matplotlib's fonts and the PNG itself were recorded as
file dependencies). With those gone, nothing linked ``ax.bar(names, totals)``
to ``fig``, and editing ``totals`` left the old chart on disk -- the
``ax -> fig`` derivation edge is that dependency.
"""
import pytest

from cash.notebook.statement.derivation_edges import (
    bump_derived_lineages,
    clear_edges_for,
    detect_derivation_edges,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


def test_an_axes_bumps_its_named_figure():
    fig, ax = plt.subplots()
    try:
        ns = {"fig": fig, "ax": ax}
        edges: dict[str, set[str]] = {}
        detect_derivation_edges(edges, "ax", ax, ns)
        assert edges == {"ax": {"fig"}}

        lineage = {"fig": "f0", "ax": "a1"}
        bumped = bump_derived_lineages(edges, lineage, {"ax"}, {"ax", "totals"},
                                       record=lineage.__setitem__, present=lambda _: True)
        assert bumped == {"fig"} and lineage["fig"] != "f0"
    finally:
        plt.close(fig)


def test_an_array_of_axes_bumps_its_named_figure():
    fig, axes = plt.subplots(1, 2)
    try:
        edges: dict[str, set[str]] = {}
        detect_derivation_edges(edges, "axes", axes, {"fig": fig, "axes": axes})
        assert edges == {"axes": {"fig"}}
    finally:
        plt.close(fig)


def test_an_unnamed_figure_records_nothing():
    """Control: no edge to a figure no variable holds."""
    fig, ax = plt.subplots()
    try:
        edges: dict[str, set[str]] = {}
        detect_derivation_edges(edges, "ax", ax, {"ax": ax})
        assert edges == {}
    finally:
        plt.close(fig)


def test_clearing_the_figure_before_detecting_keeps_the_edge():
    """``fig, ax = plt.subplots()`` rebinds both. Clearing ``fig`` drops edges
    INTO it, so clearing must finish before detection starts, whatever order
    the output set yields."""
    fig, ax = plt.subplots()
    try:
        ns = {"fig": fig, "ax": ax}
        edges: dict[str, set[str]] = {"ax": {"fig"}}
        for out in ("ax", "fig"):
            clear_edges_for(edges, out)
        for out in ("ax", "fig"):
            detect_derivation_edges(edges, out, ns[out], ns)
        assert edges == {"ax": {"fig"}}
    finally:
        plt.close(fig)
