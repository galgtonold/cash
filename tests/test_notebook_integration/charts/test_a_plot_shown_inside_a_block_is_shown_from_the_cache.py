"""A plot shown inside a ``with``, ``if`` or ``try`` block is kept with every
figure it showed.

Such a block runs as one unit whose displays go straight to the notebook,
past the capture a top-level statement has. A plot run holding it kept only
what was captured: one shown only inside the block was never kept, and one
that also called ``plt.show()`` after the block was kept with the last
figure alone, so a second Run All showed one figure where plain Python
shows two.

Each drawing counts its calls with an ``os.open`` append, waived with
``# @cash:assume-safe`` (see the agent instructions): a hit cannot restore it.
"""

from __future__ import annotations

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

pytest.importorskip("matplotlib")

# A fresh style: an earlier test's rcParams in a reused kernel would key the
# plots differently from one run to the next.
HEAD = (
    "import matplotlib.pyplot as plt\nplt.rcdefaults()\n%matplotlib inline\n"
    "import os, time\nimport numpy as np\nimport cash\n"
    "cash.configure(call_cost_floor_seconds=1000.0)\nLOG = r'{log}'\n"
    "def points(n):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return np.arange(n), np.arange(n) ** 2\n"
)


def _calls(log) -> int:
    return len(log.read_text(encoding="utf-8")) if log.exists() else 0


def _images(cell) -> list[str]:
    return [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]


def _run_twice(nb_runner, tmp_path, cell: str):
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([HEAD.format(log=log), "N = 10", cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(3))
    calls = _calls(log)
    nb_runner.run_all()
    return first, _images(nb_runner.get_cell(3)), calls, _calls(log)


@pytest.mark.parametrize(
    "cell",
    [
        "with plt.style.context('ggplot'):\n    plt.plot(*points(N))\n    plt.show()",
        "if N > 5:\n    plt.plot(*points(N))\n    plt.show()",
        "try:\n    plt.plot(*points(N))\n    plt.show()\nexcept ValueError:\n    pass",
    ],
    ids=["with", "if", "try"],
)
def test_a_plot_shown_inside_a_block_is_served(nb_runner, tmp_path, cell):
    first, again, calls, calls_after = _run_twice(nb_runner, tmp_path, cell)
    assert len(first) == 1 and calls == 1
    assert calls_after == calls, "the unchanged plot was drawn again"
    assert again == first


@pytest.mark.parametrize(
    "cell",
    [
        "if N > 5:\n    plt.plot(*points(N))\n    plt.show()\n    plt.plot(*points(N + 1))\nplt.show()",
        "with plt.style.context('ggplot'):\n    plt.plot(*points(N))\n    plt.show()\n    plt.plot(*points(N + 1))\nplt.show()",
    ],
    ids=["if", "with"],
)
def test_a_served_plot_shows_the_figure_shown_inside_a_block_too(nb_runner, tmp_path, cell):
    first, again, calls, calls_after = _run_twice(nb_runner, tmp_path, cell)
    assert len(first) == 2 and calls == 2
    assert again == first, "the plot shown from the cache lost the figure the block showed"
    assert calls_after == calls, "the unchanged plot was drawn again"
