"""A plot whose inputs did not change is shown from the cache, not drawn again.

Under the inline backend a plot ends where it began, with no figure open: it
leaves only what it showed. Run again with the same code and inputs, cash
shows that output and skips the drawing; the figure is then not live, so a
plot whose figure a later cell reads is drawn again, and one that writes a
file or reads its data from one still runs every time.

Each drawing helper counts its calls with an ``os.open`` append, waived with
``# @cash:assume-safe`` (see the agent instructions): a hit cannot restore it.
"""

from __future__ import annotations

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

pytest.importorskip("matplotlib")

# Calls are not cached on their own (``call_cost_floor_seconds``): a count
# then says whether the statement drawing the plot ran.
HEAD = (
    "%matplotlib inline\nimport os, time\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport cash\n"
    "cash.configure(call_cost_floor_seconds=1000.0)\nLOG = r'{log}'"
)
COUNT = (
    "def count():\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
)
RUN = (
    COUNT + "def run(n):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    x = np.arange(n)\n"
    "    plt.plot(x, x ** 2)\n"
    "    plt.title(f'n={n}')\n"
    "    plt.show()\n"
    "def points(n):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return np.arange(n), np.arange(n) ** 2\n"
)


def _calls(log) -> int:
    return len(log.read_text()) if log.exists() else 0


def _images(cell) -> list[str]:
    return [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]


def _kinds(cell) -> list[str]:
    return [
        "image" if "image/png" in o.get("data", {}) else o["output_type"]
        for o in cell.get("outputs", [])
        if o["output_type"] in ("execute_result", "display_data")
        and "<IPython.core.display.HTML object>" not in o.get("data", {}).get("text/plain", "")
    ]


def _notebook(nb_runner, tmp_path, *cells: str):
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([HEAD.format(log=log), RUN, *cells])
    nb_runner.start_kernel()
    return log


def test_a_helper_that_plots_is_shown_again_without_running(nb_runner, tmp_path):
    log = _notebook(nb_runner, tmp_path, "N = 10", "run(N)")
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(4))
    assert len(first) == 1 and _calls(log) == 1

    nb_runner.run_all()
    assert _calls(log) == 1, "the plot was drawn again with nothing it reads changed"
    assert _images(nb_runner.get_cell(4)) == first


# A fresh kernel: matplotlib's settings are part of what keys a plot, and a
# reused kernel may hold settings an earlier test changed that the restart drops.
@pytest.mark.fresh_kernel
def test_a_plot_is_shown_from_the_disk_cache_after_a_restart(nb_runner, tmp_path):
    log = _notebook(nb_runner, tmp_path, "N = 10", "run(N)")
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(4))
    assert len(first) == 1

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert _calls(log) == 1, "after a restart the unchanged plot was drawn again"
    assert _images(nb_runner.get_cell(4)) == first


def test_a_plot_whose_input_changed_is_drawn_again(nb_runner, tmp_path):
    log = _notebook(nb_runner, tmp_path, "N = 10", "run(N)")
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(4))
    nb_runner.run_all()
    assert _calls(log) == 1, "control: the unchanged plot is served"

    nb_runner.set_cell_source(3, "N = 12")
    nb_runner.run_all()
    assert _calls(log) == 2, "a plot of a changed input was served"
    assert _images(nb_runner.get_cell(4)) != first


def test_a_figure_drawn_to_the_end_of_the_cell_shows_after_its_value(nb_runner, tmp_path):
    """No ``plt.show()``: the inline backend shows the figure after the
    cell's value; served, it comes in the same place."""
    plot = "fig, ax = plt.subplots()\nax.plot(*points(N))\nax.set_title('squares')"
    log = _notebook(nb_runner, tmp_path, "N = 10", plot)
    nb_runner.run_all()
    first = nb_runner.get_cell(4)
    assert _kinds(first) == ["execute_result", "image"]

    nb_runner.run_all()
    again = nb_runner.get_cell(4)
    assert _calls(log) == 1
    assert _kinds(again) == ["execute_result", "image"]
    assert _images(again) == _images(first)


def test_a_later_cell_reading_the_figure_has_it_drawn(nb_runner, tmp_path):
    out = tmp_path / "fig.png"
    log = _notebook(nb_runner, tmp_path, "N = 10", "fig, ax = plt.subplots()\nax.plot(*points(N))")
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 1, "control: with nothing reading the figure, the plot is served"

    nb_runner.add_cell(f"fig.savefig(r'{out}')", save=True)
    nb_runner.run_all()
    assert _calls(log) == 2, "a plot whose figure a later cell saves was served, leaving no live figure"
    assert out.exists()
    assert "Error" not in nb_runner.get_output(5)


def test_a_later_cell_drawing_on_a_new_figure_matches_a_plain_run(nb_runner, tmp_path):
    """After the inline backend closed the plot's figure, ``plt.xlabel`` in
    the next cell starts a new one, served or not."""
    log = _notebook(nb_runner, tmp_path, "N = 10", "run(N)", "plt.xlabel('later')\nplt.show()")
    nb_runner.run_all()
    later = _images(nb_runner.get_cell(5))
    assert len(later) == 1

    nb_runner.run_all()
    assert _calls(log) == 1
    assert _images(nb_runner.get_cell(5)) == later


def test_a_plot_that_saves_a_file_runs_every_time(nb_runner, tmp_path):
    out = tmp_path / "saved.png"
    shown = "plt.plot(*points(N))\nplt.show()"
    saved = f"plt.plot(*points(N))\nplt.savefig(r'{out}')\nplt.show()"
    log = _notebook(nb_runner, tmp_path, "N = 10", shown, saved)
    nb_runner.run_all()
    assert _calls(log) == 2
    out.unlink()
    nb_runner.run_all()
    # The plot alone is served; the one saving a file runs again.
    assert _calls(log) == 3
    assert out.exists(), "a hit skipped the savefig"


def test_a_pandas_plot_is_shown_again_without_running(nb_runner, tmp_path):
    pytest.importorskip("pandas")
    cell = "import pandas as pd\npd.Series(points(N)[1]).plot(kind='barh')"
    log = _notebook(nb_runner, tmp_path, "N = 5", cell)
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(4))
    assert len(first) == 1

    nb_runner.run_all()
    assert _calls(log) == 1
    assert _images(nb_runner.get_cell(4)) == first


def test_a_seaborn_plot_is_shown_again_without_running(nb_runner, tmp_path):
    pytest.importorskip("seaborn")
    cell = "import seaborn as sns\nsns.lineplot(x=np.arange(N), y=points(N)[1])\nplt.show()"
    log = _notebook(nb_runner, tmp_path, "N = 5", cell)
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(4))
    assert len(first) == 1

    nb_runner.run_all()
    assert _calls(log) == 1
    assert _images(nb_runner.get_cell(4)) == first
