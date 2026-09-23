"""The STALE FILE line judges a file by what its writer recorded when it wrote it.

Round 29 found the line added after round 28 wrong both ways:

* r29s4 (WRONG, 3/3): a chart drawn inside ``for ax, col in zip(axes, ...)``
  was never named after an upstream change, while the CSVs were. The line
  asked whether the writer's INPUTS had drifted, and ``fig.savefig`` reads only
  ``fig``, made by ``plt.subplots()``: what was drawn on it came in through the
  loop variable. Drawing on ``axes[0]`` directly was caught.
* r29s5 (WRONG, 3/3): after a kernel restart no chart was ever named. The
  drift test compared against the runtime lineage, which a new kernel has
  none of.
* r29s2: ``os.makedirs('report', exist_ok=True)`` was named -- a folder.

The writer's own provenance -- the lineages of what it read and, for a
figure, the history of what was drawn into it, recorded when it wrote --
answers all three.
"""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.files, pytest.mark.timeout(300)]

SETUP = (
    "import cash\n%cash_on\n%cash_badge print\n"
    "import numpy as np\nimport pandas as pd\nfrom pathlib import Path\n"
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "Path('out').mkdir(exist_ok=True)"
)

LOOP_CHART = (
    "fig, axes = plt.subplots(1, 2, figsize=(4, 2))\n"
    "for ax, col in zip(axes, ['a', 'b']):\n"
    "    ax.plot(emb[col])\n"
    "fig.savefig('out/chart.png', dpi=30)\n"
    "plt.close(fig)"
)


def _cells(chart):
    return [
        SETUP,
        "K = 2",
        "emb = pd.DataFrame({'a': np.arange(50) * K, 'b': np.arange(50) % 7})",
        chart,
        "emb.describe().to_csv('out/table.csv')",
        "print('rows', len(emb))",
    ]


def _stale_lines(out):
    return [line for line in out.splitlines() if "STALE FILE" in line]


def test_a_chart_drawn_in_a_loop_over_axes_is_named(nb_runner):
    nb_runner.create_notebook(_cells(LOOP_CHART))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, "K = 3")
    nb_runner.run_cell(6)
    lines = _stale_lines(nb_runner.get_output(6))
    assert any("chart.png" in line for line in lines), nb_runner.get_raw_output(6)
    assert any("table.csv" in line for line in lines), nb_runner.get_raw_output(6)


def test_a_chart_is_named_after_a_restart(nb_runner):
    nb_runner.create_notebook(_cells(LOOP_CHART))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.set_cell_source(2, "K = 3")
    nb_runner.run_cell(6)
    lines = _stale_lines(nb_runner.get_output(6))
    assert any("chart.png" in line for line in lines), nb_runner.get_raw_output(6)
    assert any("table.csv" in line for line in lines), nb_runner.get_raw_output(6)


def test_nothing_is_named_when_nothing_changed(nb_runner):
    nb_runner.create_notebook(_cells(LOOP_CHART))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(6)
    assert not _stale_lines(nb_runner.get_output(6)), nb_runner.get_raw_output(6)


def test_a_folder_is_never_named(nb_runner):
    cells = _cells(LOOP_CHART)
    cells[1] = "import os\nos.makedirs('report', exist_ok=True)\nK = 2"
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, "import os\nos.makedirs('report', exist_ok=True)\nK = 3")
    nb_runner.run_cell(6)
    lines = _stale_lines(nb_runner.get_output(6))
    assert not any("report" in line and "report/" not in line for line in lines), lines
