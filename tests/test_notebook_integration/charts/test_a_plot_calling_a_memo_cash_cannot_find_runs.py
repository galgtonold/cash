"""A plot that calls a ``functools.cache`` memo cash cannot find is drawn
every time.

``show(get_cities(6))`` reaches the memo through a closure, a dict, a list,
an object's attribute or a ``functools.partial``, where cash does not look
for memos, so it cannot put back what a run left there. Served, the
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
COUNT = (
    "def count():\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "\n"
    "def _cities(n):\n"
    "    print('computing cities')\n"
    "    random.seed(n * 7)\n"
    "    return tuple(random.random() for _ in range(n))\n"
    "\n"
    "def show(points):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    plt.figure(figsize=(2, 2))\n"
    "    plt.plot(points)\n"
    "\n"
)
HOLDERS = {
    "closure": "def make():\n    c = cache(_cities)\n    def get(n):\n        return c(n)\n    return get\nget_cities = make()\n",
    "dict": "H = {'c': cache(_cities)}\ndef get_cities(n):\n    return H['c'](n)\n",
    "list": "MEMOS = [cache(_cities)]\ndef get_cities(n):\n    return MEMOS[0](n)\n",
    "namespace": "import types\nNS = types.SimpleNamespace(c=cache(_cities))\ndef get_cities(n):\n    return NS.c(n)\n",
    "instance": (
        "class Loader:\n    def __init__(self):\n        self.c = cache(_cities)\n"
        "L = Loader()\ndef get_cities(n):\n    return L.c(n)\n"
    ),
    "partial": "from functools import partial\nget_cities = partial(cache(_cities))\n",
}
PLOT = "show(get_cities(6))"


def _calls(log) -> int:
    return len(log.read_text(encoding="utf-8")) if log.exists() else 0


def _printed(cell) -> str:
    return "".join(o.get("text", "") for o in cell.get("outputs", []) if o["output_type"] == "stream")


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("holder", list(HOLDERS))
def test_the_plot_runs_at_both_places_and_prints_as_plain_python(nb_runner, tmp_path, holder):
    log = tmp_path / "calls.log"
    cells = [HEAD.format(log=log), COUNT + HOLDERS[holder], PLOT, PLOT + "  # again", "after = random.random()"]
    nb_runner.create_notebook(cells)
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


@pytest.mark.fresh_kernel
def test_a_memo_built_again_on_each_run_all_leaves_plots_that_never_call_it_served(nb_runner, tmp_path):
    """A cell above that builds such a memo again on every Run All, and
    that no plot calls, is no reason to draw the plots: one met with no
    calls was not called by the run."""
    log = tmp_path / "calls.log"
    cells = [HEAD.format(log=log), COUNT + HOLDERS["dict"], "show([1, 2])", "show([2, 1])"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    drawn = _calls(log)
    nb_runner.run_all()
    assert _calls(log) == drawn, "a memo no plot calls kept the plots from being served"


@pytest.mark.fresh_kernel
def test_a_plot_clearing_such_a_memo_through_a_helper_runs(nb_runner, tmp_path):
    """Cleared by the plot, the memo's counters read as a memo never
    called; they were not before the plot, which cash reads as it starts."""
    log = tmp_path / "calls.log"
    defs = COUNT + HOLDERS["dict"] + "def reset():\n    H['c'].cache_clear()\n"
    cells = [
        HEAD.format(log=log),
        defs,
        "pts = get_cities(6)",
        "plt.figure()\nreset()\nshow([1, 2])",
        "again = get_cities(6)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert "computing cities" in _printed(nb_runner.get_cell(5))
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert "computing cities" in _printed(nb_runner.get_cell(5)), "the plot clearing the memo was served"
