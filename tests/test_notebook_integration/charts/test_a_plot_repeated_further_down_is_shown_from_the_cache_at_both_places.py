"""A plotting call written again further down is shown from the cache at
both places.

``run(exhaustive_tsp, random_cities(6))`` runs a slow search over cities a
``functools.cache`` helper makes, draws the tour through helpers that open
a figure with ``plt.figure`` and draw on it with ``plt.plot``, and returns
a result holding the time the search took. The tutorial runs the same call
again further down. The first time, ``random_cities(6)`` computes the
cities (seeding and drawing from ``random``); the second time its memo
already holds them. The two places keep what they showed apart: on the
next Run All, after a restart or not, each is shown from the cache, and
every cell below sees what plain Python (restart + Run All) gives.

A cell drawing through ``plot_segment`` with no ``plt.figure`` of its own
draws on whatever figure is current: one that starts on a figure it made
itself in the same cell is shown whole, and one starting where no figure
is open draws on a new one, as plain Python does.

The slow parts count their calls with an ``os.open`` append, waived with
``# @cash:assume-safe`` (see the agent instructions): a hit cannot restore it.
"""

from __future__ import annotations

import json

import pytest

from tests._scripts import run_python
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

pytest.importorskip("matplotlib")

HEAD = (
    "%matplotlib inline\nimport itertools, os, random, time\nimport matplotlib.pyplot as plt\n"
    "from collections import Counter\nfrom functools import cache\nLOG = r'{log}'"
)
SEARCH = (
    "def count(tag):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, tag.encode())\n"
    "    os.close(fd)\n"
    "\n"
    "def tour_length(tour):\n"
    "    return sum(abs(tour[i] - tour[i - 1]) for i in range(len(tour)))\n"
    "\n"
    "def valid_tour(tour, cities):\n"
    "    return Counter(tour) == Counter(cities)\n"
    "\n"
    "@cache\n"
    "def random_cities(n, seed=1234, width=3000, height=2000):\n"
    "    random.seed(n * seed)\n"
    "    return frozenset(complex(random.uniform(0, width), random.uniform(0, height)) for c in range(n))\n"
    "\n"
    "def exhaustive_tsp(cities):\n"
    "    count('t')\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return list(min(itertools.permutations(cities), key=tour_length))\n"
    "\n"
    "def mst(vertexes):\n"
    "    count('m')\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    ordered = sorted(vertexes, key=abs)\n"
    "    return {a: [b] for a, b in zip(ordered, ordered[1:])}\n"
)
PLOT = (
    "def plot_tour(tour, tour_style='bo-', start_style='rD'):\n"
    "    scale = 1 + len(tour) ** 0.5 // 10\n"
    "    plt.figure(figsize=((3 * scale, 2 * scale)))\n"
    "    start = tour[0]\n"
    "    plot_segment([*tour, start], tour_style)\n"
    "    plot_segment([start], start_style)\n"
    "\n"
    "def plot_segment(segment, style='bo:'):\n"
    "    xs = [city.real for city in segment]\n"
    "    ys = [city.imag for city in segment]\n"
    "    plt.plot(xs, ys, style,  clip_on=False)\n"
    "    plt.axis('scaled')\n"
    "    plt.axis('off')\n"
    "\n"
    "def plot_graph(graph):\n"
    "    links = ((parent, child) for parent in graph for child in graph[parent])\n"
    "    for link in links:\n"
    "        plot_segment(link, 'g.-')\n"
)
RUN = (
    "def run(tsp, cities, plot=True):\n"
    "    t0   = time.perf_counter()\n"
    "    tour = tsp(cities)\n"
    "    t1   = time.perf_counter()\n"
    "    assert valid_tour(tour, cities)\n"
    "    if plot:\n"
    "        plot_tour(tour)\n"
    "    return Run(tour, t1 - t0, tsp)\n"
    "\n"
    "class Run:\n"
    "    def __init__(self, tour, seconds, tsp):\n"
    "        self.tour = tour; self.msec = 1000 * seconds; self.name = tsp.__name__\n"
    "        self.length = tour_length(tour)\n"
    "    def __repr__(self):\n"
    "        return f'[{self.name}({len(self.tour)} cities) => {self.length:,.0f} length tour]'\n"
)
CALL = "run(exhaustive_tsp, random_cities(6))"
AGAIN = "run(exhaustive_tsp, random_cities(6))  # the same run, from the memo"
GRAPH = "plot_graph(mst(random_cities(7)))"
AFTER = "after = random.random()"


def _cells(log, *middle: str) -> list[str]:
    return [HEAD.format(log=log), SEARCH, PLOT, RUN, *middle, AFTER]


def _calls(log) -> str:
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _images(cell) -> list[str]:
    return [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]


def _value(cell) -> str:
    for o in cell.get("outputs", []):
        if o["output_type"] == "execute_result":
            return o["data"].get("text/plain", "")
    return ""


#: Where the run leaves what the cells below go on with.
LEFT = ("after", "hash(random.getstate())", "random_cities.cache_info().currsize", "len(plt.get_fignums())")


def _plain(tmp_path, cells: list[str]) -> list[str]:
    """`LEFT` after *cells* run top to bottom in a fresh interpreter, each
    cell's figures closed after it as the inline backend closes them."""
    lines = ["import matplotlib", "matplotlib.use('Agg')", "import matplotlib.pyplot as plt", "ns = {}"]
    for i, cell in enumerate(cells):
        src = "\n".join(line for line in cell.splitlines() if not line.startswith("%"))
        lines.append(f"exec(compile({src!r}, 'cell{i}', 'exec'), ns)")
        if i == 0:
            lines.append(f"ns['LOG'] = {str(tmp_path / 'plain.log')!r}")
        lines.append("plt.close('all')")
    lines.append(f"import json; print(json.dumps([repr(eval(e, ns)) for e in {list(LEFT)!r}]))")
    script = tmp_path / "plain.py"
    script.write_text("\n".join(lines), encoding="utf-8")
    out = run_python(script, cwd=tmp_path)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _left(nb_runner) -> list[str]:
    return [nb_runner.peek(e) for e in LEFT]


def _notebook(nb_runner, tmp_path, *middle: str):
    log = tmp_path / "calls.log"
    cells = _cells(log, *middle)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    return log, cells


def test_a_repeated_plotting_call_is_shown_at_both_places_on_the_next_run_all(nb_runner, tmp_path):
    log, cells = _notebook(nb_runner, tmp_path, CALL, AGAIN)
    nb_runner.run_all()
    assert _calls(log) == "tt"
    first, again = _images(nb_runner.get_cell(5)), _images(nb_runner.get_cell(6))
    assert len(first) == 1 and len(again) == 1
    value = _value(nb_runner.get_cell(5))
    assert "6 cities" in value and _value(nb_runner.get_cell(6)) == value

    nb_runner.run_all()
    assert _calls(log) == "tt", "a place of the repeated plotting call ran its search again"
    assert _images(nb_runner.get_cell(5)) == first
    assert _images(nb_runner.get_cell(6)) == again
    assert _value(nb_runner.get_cell(6)) == value
    assert _left(nb_runner) == _plain(tmp_path, cells)


@pytest.mark.fresh_kernel
def test_a_repeated_plotting_call_is_shown_at_both_places_after_a_restart(nb_runner, tmp_path):
    log, cells = _notebook(nb_runner, tmp_path, CALL, AGAIN)
    nb_runner.run_all()
    first, again = _images(nb_runner.get_cell(5)), _images(nb_runner.get_cell(6))
    value = _value(nb_runner.get_cell(6))

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert _calls(log) == "tt", "after a restart a place of the repeated plotting call ran its search again"
    assert _images(nb_runner.get_cell(5)) == first
    assert _images(nb_runner.get_cell(6)) == again
    assert _value(nb_runner.get_cell(6)) == value
    assert _left(nb_runner) == _plain(tmp_path, cells)

    nb_runner.run_all()
    assert _calls(log) == "tt"
    assert _left(nb_runner) == _plain(tmp_path, cells)


def test_an_edited_drawing_helper_draws_both_places_again(nb_runner, tmp_path):
    log, cells = _notebook(nb_runner, tmp_path, CALL, AGAIN)
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == "tt", "control: unchanged, both places are served"
    first = _images(nb_runner.get_cell(5))

    edited = PLOT.replace("plt.plot(xs, ys, style,", "plt.plot(xs, ys, style, linewidth=4,")
    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()
    assert _calls(log) == "tttt", "after plot_segment changed, a place of the plotting call was served"
    assert _images(nb_runner.get_cell(5)) != first
    assert _images(nb_runner.get_cell(5)) == _images(nb_runner.get_cell(6))
    cells[2] = edited
    assert _left(nb_runner) == _plain(tmp_path, cells)


def test_other_cities_above_run_the_places_they_reach(nb_runner, tmp_path):
    log, cells = _notebook(nb_runner, tmp_path, CALL, AGAIN)
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == "tt", "control: unchanged, both places are served"

    other = "run(exhaustive_tsp, random_cities(5))"
    nb_runner.set_cell_source(5, other)
    nb_runner.run_all()
    # Plain Python now computes the six cities at the second place, as the
    # first place did before: that place's earlier run is shown there.
    assert _calls(log) == "ttt"
    assert "5 cities" in _value(nb_runner.get_cell(5))
    assert "6 cities" in _value(nb_runner.get_cell(6))
    cells[4] = other
    assert _left(nb_runner) == _plain(tmp_path, cells)


def test_a_graph_drawn_where_no_figure_is_open_is_shown_again(nb_runner, tmp_path):
    """``plot_graph`` makes no figure: it draws on a new one, as the cell
    starts with none open."""
    log, cells = _notebook(nb_runner, tmp_path, CALL, GRAPH)
    nb_runner.run_all()
    graph = _images(nb_runner.get_cell(6))
    assert len(graph) == 1 and _calls(log) == "tm"

    nb_runner.run_all()
    assert _calls(log) == "tm", "the unchanged graph was drawn again"
    assert _images(nb_runner.get_cell(6)) == graph
    assert _left(nb_runner) == _plain(tmp_path, cells)


def test_a_graph_drawn_on_the_tour_of_its_cell_is_drawn_on_that_figure(nb_runner, tmp_path):
    """``plot_graph`` after ``run`` in one cell draws on the tour's figure:
    one image holding both, shown again whole on the next Run All."""
    both = f"{CALL}\n{GRAPH}"
    log, cells = _notebook(nb_runner, tmp_path, both)
    nb_runner.run_all()
    shown = _images(nb_runner.get_cell(5))
    assert len(shown) == 1 and _calls(log) == "tm"

    nb_runner.run_all()
    assert _calls(log) == "tm", "the unchanged tour and graph were drawn again"
    assert _images(nb_runner.get_cell(5)) == shown
    assert _left(nb_runner) == _plain(tmp_path, cells)
