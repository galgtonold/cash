"""Which statements belong to a figure's history (round 21, Phase 2).

A figure is drawn by more than calls on ``ax`` itself: ``tot.plot(ax=axes[0])``
hands the axes to a call on something else, ``axes[1].set_xlabel`` reaches it
through a subscript. Leaving either out of a replay re-drew the figure without
it -- the blank chart. And after a restart there is no live ``fig`` to
classify, so the carrier has to be recognised by the code that made it.
"""

from cash.notebook.upstream.reexecution_planner import _fills_carrier, _passes_carrier_to_a_call
from cash.notebook.upstream.stateful_carriers import carrier_kind_from_producer

SIBLINGS = {"fig", "axes"}


def _entry(code, outputs=()):
    return (code, set(outputs), set())


def test_a_call_handed_the_axes_is_a_fill():
    assert _passes_carrier_to_a_call("tot.plot(ax=axes[0], title='t')", SIBLINGS)
    assert _passes_carrier_to_a_call("imp.plot.barh(x='f', y='w', ax=axes, legend=False)", SIBLINGS)
    assert _passes_carrier_to_a_call("draw_panel(axes[1], data)", SIBLINGS)


def test_a_call_on_a_part_of_the_figure_is_a_fill():
    assert _passes_carrier_to_a_call("axes[1].set_xlabel('week')", SIBLINGS)
    assert _passes_carrier_to_a_call("fig.axes[0].xaxis.set_visible(False)", SIBLINGS)


def test_unrelated_statements_are_not_fills():
    """Control: the check must not pull in the rest of the cell."""
    assert not _passes_carrier_to_a_call("tot = weekly.groupby('week').sum()", SIBLINGS)
    assert not _passes_carrier_to_a_call("print(len(weekly))", SIBLINGS)
    assert not _fills_carrier(_entry("series = weekly.pivot_table(index='sku')", {"series"}), SIBLINGS)


def test_outputs_still_count():
    assert _fills_carrier(_entry("axes.bar(names, totals)", {"axes"}), SIBLINGS)


def test_the_carrier_is_recognised_by_its_producer_after_a_restart():
    assert carrier_kind_from_producer("fig, axes = plt.subplots(1, 2, figsize=(8, 3))") == "matplotlib Figure"
    assert carrier_kind_from_producer("fig = plt.figure()") == "matplotlib Figure"
    assert carrier_kind_from_producer("rng = np.random.default_rng(7)") == "numpy Generator"
    assert carrier_kind_from_producer("r = random.Random(3)") == "random.Random"
    assert carrier_kind_from_producer("tot = weekly.groupby('week').sum()") is None
