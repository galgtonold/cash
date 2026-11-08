"""A plot that calls a ``functools.cache`` memo cash cannot find is drawn
every time.

``show(get_cities(6))`` reaches the memo through a closure (``make()``
returns a function calling a cached helper it defined), where cash does not
look for memos, so it cannot put back what a run left there. Served, the
memo would stay empty: the same call further down would compute again,
printing "computing cities" and reseeding ``random``, where plain Python
(restart + Run All) finds the answer. So the plot runs, at both places, and
every cell shows and draws what plain Python does.
"""

from __future__ import annotations

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

pytest.importorskip("matplotlib")

HEAD = (
    "%matplotlib inline\nimport os, random, time\nimport matplotlib.pyplot as plt\n"
    "from functools import cache\nLOG = r'{log}'"
)
DEFS = (
    "def count():\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "\n"
    "def make():\n"
    "    @cache\n"
    "    def cities(n):\n"
    "        print('computing cities')\n"
    "        random.seed(n * 7)\n"
    "        return tuple(random.random() for _ in range(n))\n"
    "    def get(n):\n"
    "        return cities(n)\n"
    "    return get\n"
    "\n"
    "get_cities = make()\n"
    "\n"
    "def show(points):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    plt.figure(figsize=(2, 2))\n"
    "    plt.plot(points)\n"
)
PLOT = "show(get_cities(6))"


def _calls(log) -> int:
    return len(log.read_text(encoding="utf-8")) if log.exists() else 0


def _printed(cell) -> str:
    return "".join(o.get("text", "") for o in cell.get("outputs", []) if o["output_type"] == "stream")


@pytest.mark.fresh_kernel
def test_the_plot_runs_at_both_places_and_prints_as_plain_python(nb_runner, tmp_path):
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([HEAD.format(log=log), DEFS, PLOT, PLOT + "  # again", "after = random.random()"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "computing cities" in _printed(nb_runner.get_cell(3))
    after = nb_runner.peek("after")

    for _ in range(2):
        nb_runner.run_all()
        assert "computing cities" not in _printed(nb_runner.get_cell(4)), "the second place showed the first's print"
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert "computing cities" in _printed(nb_runner.get_cell(3))
    assert "computing cities" not in _printed(nb_runner.get_cell(4)), "the second place showed the first's print"
    assert nb_runner.peek("after") == after
    assert _calls(log) == 8, "a plot calling a memo cash cannot find was served"
