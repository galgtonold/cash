"""Drawing a table on a chart leaves the table as it was.

``forest(axes[0], overall.assign(label=...), 'Overall')`` leaves ``axes``
holding parts of ``overall``, and the statement records that edge. A draw on
``axes`` changes ``axes``, not ``overall``: the edge the statement made
itself is a creation, as ``v = a[...]`` is, and moved ``overall`` on, so a
later summary re-ran the chart's drawing to rebuild it and wrote half a
chart. An edge the statement found already there (``ALIAS = points[7]``,
then ``ALIAS.x = len(points)``) still moves the variable it read.

After a restart, the axes a helper draws on (``forest(axes[1], s)``) are not
live, and only the verdict the runtime recorded says the call changed them:
without it the simulation left that panel out of the replay.
"""

import ast

import pytest

from cash.analysis.mutation_effects import classify_receivers
from cash.notebook.statement.derivation_edges import bump_derived_lineages, edges_among


def _bump(edges, before, outputs, inputs):
    lineage = {name: f"{name}0" for name in ("axes", "fig", "overall", "points", "ALIAS")}
    bumped = bump_derived_lineages(
        edges, lineage, outputs, inputs, record=lineage.__setitem__, present=lambda _: True, before=before
    )
    return bumped


def test_an_edge_the_statement_made_moves_no_input_on():
    before = edges_among({"axes": {"fig"}}, {"axes", "overall"})
    edges = {"axes": {"fig", "overall"}, "overall": {"axes"}}
    assert _bump(edges, before, {"axes"}, {"axes", "overall"}) == {"fig"}


def test_an_edge_that_was_there_moves_the_input_on():
    edges = {"ALIAS": {"points"}, "points": {"ALIAS"}}
    before = edges_among(edges, {"ALIAS", "points"})
    assert _bump(edges, before, {"ALIAS"}, {"ALIAS", "points"}) == {"points"}


def test_a_variable_the_statement_does_not_read_moves_on_either_way():
    edges = {"axes": {"fig", "overall"}, "overall": {"axes"}}
    assert _bump(edges, {}, {"axes"}, {"axes"}) == {"fig", "overall"}


def test_a_drawn_on_axes_argument_is_told_apart():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, axes = plt.subplots(1, 2)
    try:
        ns = {"axes": axes, "fig": fig, "s": [1], "forest": lambda ax, df, title: None, "plt": plt}
        tree = ast.parse("forest(axes[1], s, 'Conversion')")
        classes = classify_receivers(tree, ns, lambda: None, arguments={"axes", "s"})
        assert classes.drawn == {"axes"} and "axes" in classes.mutated
        closed = classify_receivers(ast.parse("plt.close(fig)"), ns, lambda: None, arguments={"fig"})
        assert closed.drawn == {"fig"}
    finally:
        plt.close(fig)
