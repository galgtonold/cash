"""Drawing with ``zip(axes.flat, df.index)`` does not tie the frame to the axes.

The loop's share check linked every target to every argument of the ``zip``,
so ``ax`` was taken to share an object with ``df``. Each later draw on the
axes then counted as a change of ``df``: rebuilding ``df`` before a cell
that only reads it re-ran the chart loop on whatever figure was current and
saved over ``grid.png`` -- a file the cell run never writes, holding another
chart than a plain run leaves there.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

pytest.importorskip("matplotlib")

CELLS = [
    "import cash\n%cash_on",
    "import pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "from pathlib import Path\nOUT = Path('out')\nOUT.mkdir(exist_ok=True)\n"
    "def save(fig, name):\n    fig.savefig(OUT / name, dpi=30)\n    plt.close(fig)",
    "SCALE = 1\n"
    "df = pd.DataFrame({'m1': [1.0, 2.0, 3.0, 4.0], 'm2': [2.0, 3.0, 4.0, 5.0]}, index=list('abcd')) * SCALE",
    "fig, axes = plt.subplots(2, 2)\nfor ax, cc in zip(axes.flat, df.index):\n    df.loc[cc].plot(ax=ax, title=cc)\n"
    "ax.set_xlabel('month')\nsave(fig, 'grid.png')",
    "fig, ax = plt.subplots()\ndf.sum().plot.bar(ax=ax, title='totals')\nsave(fig, 'totals.png')",
    "print('sum', float(df.sum().sum()))",
]


def test_rebuilding_the_frame_does_not_redraw_the_chart_loop(nb_runner):
    work = Path(nb_runner.work_dir)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    # The frame changes and the chart grid is looked at again...
    nb_runner.set_cell_source(3, CELLS[2].replace("SCALE = 1", "SCALE = 2"))
    nb_runner.run_cell(4)
    grid = work / "out" / "grid.png"
    before = grid.read_bytes()
    stamp = grid.stat().st_mtime_ns
    # ...then changes again, and only the summary below is run.
    nb_runner.set_cell_source(3, CELLS[2].replace("SCALE = 1", "SCALE = 3"))
    nb_runner.run_cell(6)
    assert "sum 72.0" in nb_runner.get_output(6)
    assert grid.stat().st_mtime_ns == stamp, "the summary cell re-drew and saved the chart grid"
    assert grid.read_bytes() == before
