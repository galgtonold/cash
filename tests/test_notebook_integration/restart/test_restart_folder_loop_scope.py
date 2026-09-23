"""After a restart, a cell below a chart cell does not re-draw the charts.

The planner re-fires a file writer above the cell being run only if that cell
reads what it writes -- a scope gate that needs every file the cell's
reconstruction reads. ``for f in files: parts.append(pd.read_csv(f))`` names
no path a reader can resolve, and after a restart nothing recorded what it
read, so the read set was unknown, nothing could be ruled out, and running a
table cell re-drew the report above it: ``rmtree`` of the report folder, every
chart, and everything the charts read (in one user's notebook: 84 s, the 1,200-file
folder read again). A loop keeps what it read now, as a statement does.

Observed on disk: the report folder and its chart keep their ``st_mtime_ns``.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

READ = (
    "import glob\nimport time\nimport shutil\nfrom pathlib import Path\nimport pandas as pd\n"
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "files = sorted(glob.glob('exports/*.csv'))\n"
    "parts = []\n"
    "for f in files:\n    time.sleep(0.2)\n    parts.append(pd.read_csv(f))\n"
    "raw = pd.concat(parts, ignore_index=True)"
)
REPORT = (
    "OUT = Path('report')\n"
    "if OUT.exists():\n    shutil.rmtree(OUT)\n"
    "OUT.mkdir()\n"
    "fig, ax = plt.subplots(figsize=(4, 3))\n"
    "ax.bar(raw['k'], raw['v'])\n"
    "fig.savefig(OUT / 'chart.png'); plt.close(fig)"
)
TABLE = "total = int(raw['v'].sum())\nprint('TOTAL', total)"
CELLS = ["import cash\n%cash_on", READ, REPORT, TABLE]
# Reads the chart: its writer, and the ``mkdir`` of the folder it is in, matter here.
VIEW = "print('CHART', len(Path('report/chart.png').read_bytes()) > 0)"


def _exports(work_dir: Path) -> None:
    folder = work_dir / "exports"
    folder.mkdir()
    for i in range(4):
        (folder / f"e{i}.csv").write_text(f"k,v\n{i},{10 * (i + 1)}\n", encoding="utf-8")


def _chart(nb_runner) -> int:
    return (Path(nb_runner.work_dir) / "report" / "chart.png").stat().st_mtime_ns


def test_a_table_cell_below_a_chart_cell_does_not_redraw_it_after_a_restart(nb_runner):
    _exports(Path(nb_runner.work_dir))
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "TOTAL 100" in nb_runner.get_output(len(CELLS))
    drawn = _chart(nb_runner)

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(len(CELLS))

    assert "TOTAL 100" in nb_runner.get_output(len(CELLS))
    # Measured before the fix: the folder removed and the chart drawn again.
    assert _chart(nb_runner) == drawn, "the chart cell was re-run for the table below it"


def test_a_cell_that_reads_a_deleted_chart_gets_its_folder_back(nb_runner):
    """The control: a folder made by the chart cell is read by whatever reads
    a file inside it, so it is made again before the chart is drawn into it."""
    import shutil

    _exports(Path(nb_runner.work_dir))
    nb_runner.create_notebook([*CELLS, VIEW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    shutil.rmtree(Path(nb_runner.work_dir) / "report")

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(len(CELLS) + 1)

    assert "CHART True" in nb_runner.get_output(len(CELLS) + 1)
    assert (Path(nb_runner.work_dir) / "report" / "chart.png").exists()
