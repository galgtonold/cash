"""Making a seaborn grid opens a pyplot figure, so its statement runs every time.

The user's objects a grid's figure holds (an artist added to its axes, a
frame) are kept right on every run because the statement that makes the grid
is not stored: `pyplot_state` reads differently after it (CHANGED_PYPLOT_REASON).
A ``Figure()`` made without pyplot leaves the reading alone; what it holds is
the share check's to find.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
sns = pytest.importorskip("seaborn")
pd = pytest.importorskip("pandas")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cash.notebook.pyplot_state import pyplot_state

FRAME = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "x"]})


@pytest.fixture(autouse=True)
def _no_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "make",
    [lambda: sns.PairGrid(FRAME, vars=["a", "b"]), lambda: sns.FacetGrid(FRAME, col="c")],
    ids=["pair", "facet"],
)
def test_a_grid_changes_the_reading(make):
    before = pyplot_state()
    make()
    assert pyplot_state() != before


def test_a_grid_made_while_another_figure_is_open_changes_the_reading_too():
    plt.figure(1).add_subplot()
    before = pyplot_state()
    sns.FacetGrid(FRAME, col="c")
    assert pyplot_state() != before


def test_a_figure_made_without_pyplot_leaves_it_alone():
    before = pyplot_state()
    Figure().add_subplot()
    assert pyplot_state() == before
