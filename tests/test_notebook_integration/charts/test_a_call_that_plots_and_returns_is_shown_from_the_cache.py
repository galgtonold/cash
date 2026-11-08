"""A call to a function of yours that draws a new figure and returns a value
is shown from the cache when its inputs are unchanged.

``run(tsp, cities)`` times a slow search, opens a figure with
``plt.figure``, draws the tour through a nested helper and returns a
result object. With the inline backend the figure is shown and closed at
the end of the cell, so the call is a whole plot: on a later run with the
same inputs (the same code re-run, an unrelated cell edited, a restart and
Run All) cash shows the stored image and value instead of searching again.

The slow search counts its calls with an ``os.open`` append, waived with
``# @cash:assume-safe`` (see the agent instructions): a hit cannot restore it.
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
SEARCH = (
    "def count():\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "\n"
    "@cache\n"
    "def random_cities(n, seed=1234):\n"
    "    random.seed(n * seed)\n"
    "    return frozenset(complex(random.uniform(0, 300), random.uniform(0, 200)) for _ in range(n))\n"
    "\n"
    "def slow_tsp(cities):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return sorted(cities, key=lambda c: (c.real, c.imag))\n"
)
PLOT = (
    "def plot_tour(tour, style='bo-'):\n"
    "    plt.figure(figsize=(3, 2))\n"
    "    plot_segment([*tour, tour[0]], style)\n"
    "    plot_segment([tour[0]], 'rD')\n"
    "\n"
    "def plot_segment(segment, style='bo:'):\n"
    "    plt.plot([c.real for c in segment], [c.imag for c in segment], style, clip_on=False)\n"
    "    plt.axis('scaled')\n"
    "    plt.axis('off')\n"
    "\n"
    "class Run:\n"
    "    def __init__(self, tour, seconds):\n"
    "        self.tour = tour\n"
    "        self.msec = 1000 * seconds\n"
    "    def __repr__(self):\n"
    "        return f'[{len(self.tour)} cities in {self.msec:,.0f} msecs]'\n"
    "\n"
    "def run(tsp, cities, plot=True):\n"
    "    t0 = time.perf_counter()\n"
    "    tour = tsp(cities)\n"
    "    t1 = time.perf_counter()\n"
    "    assert sorted(tour, key=abs) == sorted(cities, key=abs)\n"
    "    if plot:\n"
    "        plot_tour(tour)\n"
    "    return Run(tour, t1 - t0)\n"
)
CALL = "run(slow_tsp, random_cities(10))"
BOUNDARY = (
    "def plot_boundary(model, xs, ys):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    grid = [[model * (x + y) for x in range(10)] for y in range(10)]\n"
    "    plt.contourf(range(10), range(10), grid, alpha=0.3)\n"
    "    plt.plot(xs, ys, 'g^')\n"
    "    plt.axis([0, 9, 0, 9])\n"
)


def _calls(log) -> int:
    return len(log.read_text(encoding="utf-8")) if log.exists() else 0


def _images(cell) -> list[str]:
    return [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]


def _value(cell) -> str:
    for o in cell.get("outputs", []):
        if o["output_type"] == "execute_result":
            return o["data"].get("text/plain", "")
    return ""


#: Where a run leaves what plain Python goes on with: the random stream
#: ``random_cities`` seeded and drew from, and the answers its memo holds.
LEFT = "(hash(random.getstate()), random_cities.cache_info().currsize)"


def _notebook(nb_runner, tmp_path, *cells: str):
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([HEAD.format(log=log), SEARCH, PLOT, *cells])
    nb_runner.start_kernel()
    return log


def test_a_plotting_call_is_shown_again_on_a_rerun_and_an_unrelated_edit(nb_runner, tmp_path):
    log = _notebook(nb_runner, tmp_path, "x = 1", CALL)
    nb_runner.run_all()
    first = nb_runner.get_cell(5)
    assert len(_images(first)) == 1 and _calls(log) == 1
    assert "10 cities" in _value(first)
    left = nb_runner.peek(LEFT)

    nb_runner.run_all()
    again = nb_runner.get_cell(5)
    assert _calls(log) == 1, "the unchanged plotting call ran its search again"
    assert _images(again) == _images(first)
    assert _value(again) == _value(first)
    assert nb_runner.peek(LEFT) == left, "served, the call left the stream or the memo otherwise than its run"

    nb_runner.set_cell_source(4, "x = 2")
    nb_runner.run_all()
    assert _calls(log) == 1, "an edit of an unrelated cell ran the plotting call again"
    assert len(_images(nb_runner.get_cell(5))) == 1
    assert nb_runner.peek(LEFT) == left


@pytest.mark.fresh_kernel
def test_a_plotting_call_is_shown_from_disk_after_a_restart(nb_runner, tmp_path):
    log = _notebook(nb_runner, tmp_path, CALL)
    nb_runner.run_all()
    first = nb_runner.get_cell(4)
    assert len(_images(first)) == 1
    left = nb_runner.peek(LEFT)

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert _calls(log) == 1, "after a restart the unchanged plotting call ran its search again"
    assert _images(nb_runner.get_cell(4)) == _images(first)
    assert _value(nb_runner.get_cell(4)) == _value(first)
    assert nb_runner.peek(LEFT) == left


def test_a_plotting_call_of_changed_cities_runs(nb_runner, tmp_path):
    log = _notebook(nb_runner, tmp_path, "N = 10", "run(slow_tsp, random_cities(N))")
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 1, "control: the unchanged call is served"
    nb_runner.set_cell_source(4, "N = 11")
    nb_runner.run_all()
    assert _calls(log) == 2
    assert "11 cities" in _value(nb_runner.get_cell(5))


def test_a_plotting_call_binding_its_result_and_figure_runs(nb_runner, tmp_path):
    """Served, ``r`` and ``fig`` would be unbound where plain Python binds
    them, so the call runs every time and a later cell can save the figure."""
    out = tmp_path / "tour.png"
    log = _notebook(nb_runner, tmp_path, f"r = {CALL}\nfig = plt.gcf()")
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 2, "a call binding r and fig was served, leaving them unbound"

    nb_runner.add_cell(f"fig.savefig(r'{out}')", save=True)
    nb_runner.run_all()
    assert _calls(log) == 3
    assert out.exists()
    assert "Error" not in nb_runner.get_output(5)


def test_a_helper_drawing_on_the_current_axes_is_shown_again(nb_runner, tmp_path):
    log = _notebook(nb_runner, tmp_path, BOUNDARY, "plot_boundary(2, [1, 2, 3], [3, 2, 1])")
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(5))
    assert len(first) == 1 and _calls(log) == 1

    nb_runner.run_all()
    assert _calls(log) == 1, "the unchanged boundary plot was drawn again"
    assert _images(nb_runner.get_cell(5)) == first


SAMPLED = (
    "def sampled_tsp(cities):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return random.sample(sorted(cities, key=abs), len(cities))\n"
)


def test_a_plotting_call_drawing_from_a_seeded_stream_follows_the_seed(nb_runner, tmp_path):
    """The call draws from the stream a cell above seeded: served while
    that seed stands, run again under another one."""
    call = "run(sampled_tsp, frozenset(complex(i, i % 3) for i in range(8)))"
    log = _notebook(nb_runner, tmp_path, SAMPLED, "random.seed(7)", call)
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(6))
    state = nb_runner.peek("hash(random.getstate())")
    nb_runner.run_all()
    assert _calls(log) == 1, "the call under the same seed ran again"
    assert _images(nb_runner.get_cell(6)) == first
    assert nb_runner.peek("hash(random.getstate())") == state

    nb_runner.set_cell_source(5, "random.seed(8)")
    nb_runner.run_all()
    assert _calls(log) == 2, "the call drawing under a changed seed was served"


def test_a_plotting_call_drawing_from_an_unseeded_stream_runs_every_time(nb_runner, tmp_path):
    call = "run(sampled_tsp, frozenset(complex(i, i % 3) for i in range(8)))"
    log = _notebook(nb_runner, tmp_path, SAMPLED, call)
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 2
    assert len(_images(nb_runner.get_cell(5))) == 1


def test_a_decision_boundary_of_a_fitted_model_is_shown_again(nb_runner, tmp_path):
    """``plot_decision_boundary(clf, X, y)`` draws a fitted model's
    predictions on the current axes, with no ``plt.figure``."""
    pytest.importorskip("sklearn")
    helper = (
        "import numpy as np\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "def plot_decision_boundary(clf, X, y):\n"
        "    count()\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    x1, x2 = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))\n"
        "    y_pred = clf.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)\n"
        "    plt.contourf(x1, x2, y_pred, alpha=0.3)\n"
        "    for idx in (0, 1):\n"
        "        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx], 'o^'[idx], linestyle='none')\n"
        "    plt.axis([0, 1, 0, 1])\n"
    )
    fit = (
        "X = np.array([[0.1, 0.2], [0.8, 0.9], [0.3, 0.7], [0.9, 0.1]])\ny = np.array([0, 1, 0, 1])\n"
        "clf = DecisionTreeClassifier(random_state=42).fit(X, y)"
    )
    log = _notebook(nb_runner, tmp_path, helper, fit, "plot_decision_boundary(clf, X, y)")
    nb_runner.run_all()
    first = _images(nb_runner.get_cell(6))
    assert len(first) == 1 and _calls(log) == 1

    nb_runner.run_all()
    assert _calls(log) == 1, "the unchanged decision boundary was drawn again"
    assert _images(nb_runner.get_cell(6)) == first
