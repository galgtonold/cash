"""An Axes handed to a plain function is drawn on, so that call always runs.

Round 22 (tester-session tests, r22s1's readout): a forest plot drawn through
``forest(axes[0], df, title)`` was served from the cache on a warm re-run --
the rule that routes an Axes argument as drawn-on sat behind an early return
for statements without a method call -- and the saved chart had an empty
panel. ``df.plot(ax=ax)`` was always right: it has a method-call receiver.
"""

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

pytestmark = [pytest.mark.integration]

CELLS = [
    "import cash\n%cash_on",
    "import time\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "def draw(ax, values, title):\n"
    "    time.sleep(0.05)\n"
    "    ax.bar(range(len(values)), values)\n"
    "    ax.set_title(title)\n"
    "data = [3, 1, 4, 1, 5]",
    "fig, axes = plt.subplots(1, 2, figsize=(6, 3))\n"
    "draw(axes[0], data, 'left')\n"
    "draw(axes[1], data[::-1], 'right')\n"
    "fig.savefig('chart.png', dpi=40)\n"
    "plt.close(fig)",
]


def test_a_warm_rerun_draws_the_same_chart(nb_runner):
    png = Path(nb_runner.work_dir) / "chart.png"
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    first = hashlib.sha256(png.read_bytes()).hexdigest()
    png.unlink()
    nb_runner.run_all()
    assert hashlib.sha256(png.read_bytes()).hexdigest() == first, "the warm run drew a different chart"


# Round 23 (r23s1): the same helper called inside a loop, one figure per item.
# A loop body is decomposed per iteration and its calls are call-cached, which
# the fix above never reached -- the call cache served `draw(ax, ...)` and the
# re-run saved blank charts.
LOOP_CELL = (
    "for name, values in [('a', data), ('b', data[::-1])]:\n"
    "    fig, ax = plt.subplots(figsize=(3, 3))\n"
    "    draw(ax, values, name)\n"
    "    fig.savefig(f'{name}.png', dpi=40)\n"
    "    plt.close(fig)"
)


def test_a_rerun_of_a_loop_draws_the_same_charts(nb_runner):
    work = Path(nb_runner.work_dir)
    nb_runner.create_notebook(CELLS[:2] + [LOOP_CELL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = {n: hashlib.sha256((work / f"{n}.png").read_bytes()).hexdigest() for n in "ab"}
    nb_runner.run_cell(3)
    again = {n: hashlib.sha256((work / f"{n}.png").read_bytes()).hexdigest() for n in "ab"}
    assert again == first, "an unchanged re-run of the loop saved different (blank) charts"
