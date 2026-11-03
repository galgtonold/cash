"""A benchmark cell whose helpers seed the stream before drawing is served
after an edit to a solver that samples above it, and gives what plain
Python gives.

The TSP notebook's shape: ``random_cities`` (``functools.cache``) calls
``random.seed`` and then draws; ``b2 = benchmarks(exact, test_suite(4, 7))``
fills it. What ``b2`` computes comes from those seeds alone, wherever the
stream stood when it began. Its run was recorded as a draw from the stream
all the same, so editing ``rep(c, k=2)`` to ``k=3``, which samples in the
cell above, ran it again ("random state changed: random") although
``exact`` holds no ``rep``.

The control ``c = first_draws(3)`` draws before it seeds: the edited draws
above reach it, and it runs again.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\n%cash_badge print\nimport functools, random, time\nfrom statistics import mean"
CITIES = (
    "@functools.cache\n"
    "def random_cities(n, seed=1234):\n"
    "    random.seed(n * seed)\n"
    "    return frozenset(complex(round(random.uniform(0, 300)), round(random.uniform(0, 200))) for _ in range(n))\n"
    "@functools.cache\n"
    "def test_suite(k, n):\n"
    "    return tuple(random_cities(n, seed=i) for i in range(k))"
)
SOLVERS = (
    "def tour_length(t):\n"
    "    return sum(abs(t[i] - t[i - 1]) for i in range(len(t)))\n"
    "def greedy(c):\n"
    "    return sorted(c, key=lambda z: (z.real, z.imag))\n"
    "def rep(c, k=2):\n"
    "    return min((sorted(c, key=lambda z, s=s: abs(z - s)) for s in random.sample(sorted(c, key=abs), k)), key=tour_length)\n"
    "def first_draws(n):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    v = random.random()\n"
    "    random.seed(n)\n"
    "    return round(v + random.random(), 6)\n"
    "exact = (greedy,)\n"
    "inexact = (greedy, rep)"
)
BENCH = (
    "@functools.cache\n"
    "def benchmark(tsp, suite):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return [tour_length(tsp(c)) for c in suite]\n"
    "def benchmarks(tsps, suite):\n"
    "    return [round(mean(benchmark(t, suite)), 3) for t in tsps]"
)
CELLS = [
    SETUP,
    CITIES,
    SOLVERS,
    BENCH,
    "b1 = benchmarks(inexact, test_suite(4, 12))",
    "c = first_draws(3)",
    "b2 = benchmarks(exact, test_suite(4, 7))",
    "x = round(random.random(), 6)",
]
B1, C, B2 = 5, 6, 7
NAMES = ("b1", "c", "b2", "x")


def _plain(cells: list[str]) -> dict:
    """What a restart and Run All of *cells* leave in plain Python."""
    import random

    state = random.getstate()
    ns: dict = {"__name__": "__plain__"}
    try:
        for cell in cells:
            exec(cell.replace("import cash\n%cash_on\n%cash_badge print\n", ""), ns)
        return {name: ns[name] for name in NAMES}
    finally:
        random.setstate(state)


def _edited() -> list[str]:
    cells = list(CELLS)
    cells[2] = cells[2].replace("k=2", "k=3")
    return cells


def _values(nb_runner) -> dict:
    return {name: eval(nb_runner.peek(name)) for name in NAMES}


def test_the_seeded_benchmark_is_served_after_an_edited_sampler_above(nb_runner):
    edited = _edited()
    plain_before, plain_after = _plain(CELLS), _plain(edited)
    # The edit moves the draws above: b1 and the control change, b2 does not.
    assert plain_before["b1"] != plain_after["b1"]
    assert plain_before["c"] != plain_after["c"]
    assert plain_before["b2"] == plain_after["b2"]

    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _values(nb_runner) == plain_before
    nb_runner.run_all()
    assert _values(nb_runner) == plain_before
    assert shows_cached(nb_runner.get_output(B2)), nb_runner.get_output(B2)

    nb_runner.set_cell_source(3, edited[2])
    nb_runner.run_all()
    assert _values(nb_runner) == plain_after
    assert not shows_cached(nb_runner.get_output(B1)), nb_runner.get_output(B1)
    assert not shows_cached(nb_runner.get_output(C)), nb_runner.get_output(C)
    assert shows_cached(nb_runner.get_output(B2)), nb_runner.get_output(B2)

    nb_runner.restart()
    nb_runner.run_all()
    assert _values(nb_runner) == plain_after
    assert shows_cached(nb_runner.get_output(B2)), nb_runner.get_output(B2)
