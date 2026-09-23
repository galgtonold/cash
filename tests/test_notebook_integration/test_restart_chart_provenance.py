"""After a restart, a chart already on disk is not drawn again.

A writer's provenance vouches for its file by the lineages of what it read,
compared with the simulation's after a restart. A figure's lineage cannot take
part: the runtime advances ``fig`` through live Axes -> Figure edges
(``ax.bar(...)``) that a restarted kernel does not have, so the simulation
never arrived at the same lineage, and every ``fig.savefig`` re-fired with
everything the chart reads. One user's notebook re-read 1,200 exports that way,
84 s, to run a table cell below two charts.

The record now carries the figure's drawing history instead -- the statements
from ``plt.subplots`` to the write, with the lineages of the data they read --
which the simulation does reproduce. The last cell reads the charts back, so
the freshness check is what decides (a writer whose file the cell does not
read is not re-fired in the first place).

Observed on disk: a re-drawn chart gets a new ``st_mtime_ns``.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

SETUP = (
    "import time\nfrom pathlib import Path\nimport matplotlib\nmatplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\nOUT = Path('out')\nOUT.mkdir(exist_ok=True)\n"
    "def slow(n):\n    time.sleep(0.3)\n    return list(range(1, n + 1))\n"
    "data = slow(5)"
)
# Two figures, as a report cell draws them: the first is filled through ``ax``
# and closed, so ``fig`` has moved on by the time the second one is saved.
CHART = (
    "fig, ax = plt.subplots(figsize=(4, 3))\n"
    "ax.bar(range(len(data)), data)\n"
    "ax.set_title('first')\n"
    "fig.tight_layout()\n"
    "fig.savefig(OUT / 'a.png'); plt.close(fig)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(4, 3))\n"
    "axes[0].bar(range(3), data[:3])\n"
    "axes[1].set_title('second')\n"
    "fig.savefig(OUT / 'b.png'); plt.close(fig)"
)
READ = "drawn = [len((OUT / n).read_bytes()) > 0 for n in ('a.png', 'b.png')]\nprint('DRAWN', drawn, sum(data))"
CELLS = ["import cash\n%cash_on", SETUP, CHART, READ]


def _charts(nb_runner) -> dict[str, tuple[bytes, int]]:
    out = Path(nb_runner.work_dir) / "out"
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(out.glob("*.png"))}


def _restart_and_read(nb_runner) -> str:
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(len(CELLS))
    return nb_runner.get_output(len(CELLS))


def _run_all(nb_runner) -> dict[str, tuple[bytes, int]]:
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    charts = _charts(nb_runner)
    assert sorted(charts) == ["a.png", "b.png"]
    return charts


def test_a_chart_on_disk_is_not_drawn_again_after_a_restart(nb_runner):
    before = _run_all(nb_runner)

    out = _restart_and_read(nb_runner)

    assert "DRAWN [True, True] 15" in out, out
    # Measured before the fix: both charts drawn again.
    assert _charts(nb_runner) == before, "a chart was drawn again after the restart"


def test_an_edited_chart_is_drawn_again(nb_runner):
    """The write itself is unchanged; the title drawn before it is not."""
    before = _run_all(nb_runner)
    nb_runner.set_cell_source(3, CHART.replace("'first'", "'renamed'"))

    out = _restart_and_read(nb_runner)

    assert "DRAWN [True, True] 15" in out, out
    after = _charts(nb_runner)
    assert after["a.png"][0] != before["a.png"][0], "the edited chart was not drawn again"


def test_a_chart_of_changed_data_is_drawn_again(nb_runner):
    """``data`` changes upstream: the bars read it, the figure does not."""
    before = _run_all(nb_runner)
    nb_runner.set_cell_source(2, SETUP.replace("slow(5)", "slow(6)"))

    out = _restart_and_read(nb_runner)

    assert "DRAWN [True, True] 21" in out, out
    after = _charts(nb_runner)
    assert after["a.png"][0] != before["a.png"][0], "a chart of changed data was not drawn again"
