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
