"""A statement that echoes a Figure or Axes does not add a figure to pyplot.

`plt.figure(figsize=(12, 12))` and `sns.heatmap(df)` echo their value, and the
entry kept that value: copying it into the RAM tier revived a second figure
in pyplot's registry, which became the current one. The next `plt.yticks()`
or `plt.savefig()` then acted on the empty copy. Such a statement is not
cached, like a name bound to a Figure.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def _agg_and_clean_registry():
    import matplotlib

    matplotlib.use("Agg")
    plt.close("all")
    yield
    plt.close("all")


def test_a_cell_that_echoes_a_figure_makes_exactly_one(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "import matplotlib.pyplot as plt")
    run_cash_cell(cash_magics, "plt.figure(figsize=(4, 4))")
    assert len(plt.get_fignums()) == 1


def test_a_cell_that_echoes_an_axes_makes_no_figure(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "import matplotlib.pyplot as plt")
    run_cash_cell(cash_magics, "plt.figure(figsize=(4, 4))")
    run_cash_cell(cash_magics, "plt.gca()")
    assert len(plt.get_fignums()) == 1


def test_the_figure_the_next_cell_draws_on_is_the_one_just_made(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "import matplotlib.pyplot as plt")
    run_cash_cell(cash_magics, "plt.figure(figsize=(4, 4))\nplt.gca().plot([1, 2, 3])\nplt.gca()")
    assert len(plt.gcf().axes[0].lines) == 1


def test_running_the_same_figure_cell_again_makes_another_figure_as_plain_python_does(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "import matplotlib.pyplot as plt")
    run_cash_cell(cash_magics, "plt.figure(figsize=(4, 4))")
    run_cash_cell(cash_magics, "plt.figure(figsize=(4, 4))")
    assert len(plt.get_fignums()) == 2
