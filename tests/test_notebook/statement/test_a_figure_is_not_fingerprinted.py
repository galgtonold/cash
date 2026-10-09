"""A plot cell does not pickle its figures to fingerprint them.

Plot statements run every time (a figure is tied to pyplot by identity), yet
each figure and axes they bound got a content hash -- a pickle of the whole
figure -- as its session hash, and ``for g, ax in zip(groups, axes):``
hashed each axes per iteration to key the loop body: 8 plot cells took
4.1-5.9 s a Run All against 2.0-2.4 s plain. A figure, its axes and what is
drawn on them are now hashed by lineage, like a frame; a loop variable bound
to one gets a digest no other binding gets, so nothing keyed by it hits.
"""

from __future__ import annotations

import pytest

from cash.notebook import restored_var
from tests._cell_driver import run_cash_cell

plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def _agg_and_clean_registry():
    import matplotlib

    matplotlib.use("Agg")
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def pickled_figures(monkeypatch):
    """How many times a figure was pickled."""
    from matplotlib.figure import Figure

    count = {"n": 0}
    real = Figure.__getstate__

    def getstate(self):
        count["n"] += 1
        return real(self)

    monkeypatch.setattr(Figure, "__getstate__", getstate)
    return count


def test_figures_axes_and_artists_are_hashed_by_lineage():
    fig, ax = plt.subplots()
    (line,) = ax.plot([1, 2])

    assert restored_var.hashed_by_lineage(fig)
    assert restored_var.hashed_by_lineage(ax)
    assert restored_var.hashed_by_lineage(line)
    assert restored_var.hashed_by_lineage((fig, ax))
    assert restored_var.hashed_by_lineage([line])
    assert not restored_var.hashed_by_lineage([1, 2])
    assert not restored_var.hashed_by_lineage(object())


def test_each_binding_of_a_figure_gets_its_own_digest():
    fig, ax = plt.subplots()

    first, again = restored_var.identity_digest(ax), restored_var.identity_digest(ax)

    assert first and again and first != again
    assert restored_var.identity_digest(fig) not in (first, again)
    assert restored_var.identity_digest([1, 2]) is None
    assert restored_var.identity_digest("identity:x") is None


def test_a_plot_cell_does_not_pickle_its_figure(cash_magics, pickled_figures):
    run_cash_cell(cash_magics, "import matplotlib.pyplot as plt")
    pickled_figures["n"] = 0

    for _ in range(2):
        run_cash_cell(cash_magics, "fig, ax = plt.subplots()\nax.plot([1, 2, 3])\nax.set_title('a')")

    assert pickled_figures["n"] == 0
    ns = cash_magics.shell.user_ns
    assert ns["ax"].get_title() == "a" and len(ns["ax"].lines) == 1 and ns["ax"].figure is ns["fig"]


def test_a_loop_over_axes_does_not_pickle_their_figure(cash_magics, pickled_figures):
    run_cash_cell(cash_magics, "import matplotlib.pyplot as plt")
    cell = "fig, axes = plt.subplots(1, 3)\nfor g, ax in zip(range(3), axes):\n    ax.plot([g, g + 1])\n    n = len(ax.lines)"
    pickled_figures["n"] = 0

    for _ in range(2):
        run_cash_cell(cash_magics, cell)

    assert pickled_figures["n"] == 0
    ns = cash_magics.shell.user_ns
    assert [len(a.lines) for a in ns["axes"]] == [1, 1, 1]
    assert ns["n"] == 1


def test_a_statement_keyed_by_an_axes_the_loop_visits_twice_runs_each_time(cash_magics):
    """The same axes twice: the body sees what the first visit drew."""
    run_cash_cell(cash_magics, "import matplotlib.pyplot as plt")
    cell = (
        "fig, ax = plt.subplots()\ncounts = []\nfor a in [ax, ax]:\n    a.plot([1, 2])\n    counts.append(len(a.lines))"
    )

    for _ in range(2):
        run_cash_cell(cash_magics, cell)
        assert cash_magics.shell.user_ns["counts"] == [1, 2]
