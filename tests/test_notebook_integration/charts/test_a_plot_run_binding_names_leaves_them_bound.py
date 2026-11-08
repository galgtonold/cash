"""A plot whose cell binds names leaves them bound, as plain Python does.

``fig, ax = plt.subplots()`` then drawing on ``ax`` binds ``fig`` and
``ax`` to the live figure and axes, and plain Python (restart + Run All)
keeps them after the inline backend shows and closes the figure: a cell run
later, even one not in the notebook yet, can read them. A plot served from
the cache draws nothing and so has no figure to bind them to; it would
leave them unbound, and that later cell would raise ``NameError``. So a
plot that binds names is drawn every time, under the inline backend that
ipykernel picks by default.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

HEAD = (
    "import matplotlib\nmatplotlib.use('module://matplotlib_inline.backend_inline')\n"
    "import matplotlib.pyplot as plt\nimport numpy as np"
)
IMPLICIT = "plt.plot([1, 2, 3], [3, 1, 2])\nplt.title('implicit figure')"
BARS = (
    "x = np.arange(50) ** 2\n"
    "fig, ax = plt.subplots()\n"
    "ax.bar(range(5), [3, 1, 4, 1, 5])\n"
    "ax.plot(x[:5])\n"
    "print('bars drawn')"
)
SHOWN = "plt.figure()\nplt.plot(x)\nplt.show()"
BOUND = "sorted(name for name in ('fig', 'ax', 'x') if name in globals())"
READ = "(type(fig).__name__, type(ax).__name__, fig.axes[0] is ax, len(ax.patches))"
PLAIN = "('Figure', 'Axes', True, 5)"


def _printed(cell) -> str:
    return "".join(o.get("text", "") for o in cell.get("outputs", []) if o["output_type"] == "stream")


def test_names_a_plot_binds_stay_bound_on_the_next_run_all(nb_runner):
    nb_runner.create_notebook([HEAD, IMPLICIT, BARS, SHOWN])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek(BOUND) == "['ax', 'fig', 'x']"

    nb_runner.run_all()
    assert nb_runner.peek(BOUND) == "['ax', 'fig', 'x']", "the plot left fig and ax unbound"
    assert nb_runner.peek(READ) == PLAIN
    assert "bars drawn" in _printed(nb_runner.get_cell(3))


@pytest.mark.parametrize("save", [True, False], ids=["saved", "not-saved-yet"])
def test_a_cell_added_after_the_run_all_reads_the_names(nb_runner, save):
    nb_runner.create_notebook([HEAD, IMPLICIT, BARS, SHOWN])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    nb_runner.add_cell("print(len(ax.patches), fig.axes[0] is ax)", save=save)
    nb_runner.run_cell(5)
    assert nb_runner.get_output(5).strip() == "5 True"
    assert nb_runner.peek("len(plt.get_fignums())") == "0", "reading the names drew the plot again"
