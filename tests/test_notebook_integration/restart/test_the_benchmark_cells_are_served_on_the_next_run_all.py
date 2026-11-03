"""The TSP notebook's benchmark cells are served on the next Run All of the
same kernel, and give what plain Python gives.

``benchmark`` memoises on the solver, and ``ensemble`` takes solvers as
defaults; a Run All defines them all again. ``held_karp`` empties
``shortest_segment`` for each problem, called through a tuple of solvers
cash cannot read as clearing, and a cell above left that memo bigger than
cash reads. On the second Run All both cells ran: the memory tier's copy
held the first pass's solvers, with no reason given, and the emptied memo
was refused ("It emptied the helper memo shortest_segment").

``segment`` is defined above ``held_karp`` here. Defined below it, as in the
notebook, ``held_karp``'s lineage differs between a first Run All and the
next (the name is missing when it is defined, there the first time), which
is another matter than the memos.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import functools, random, time\nfrom statistics import mean"
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
    "def ensemble(c, tsps=(greedy, rep)):\n"
    "    return min((t(c) for t in tsps), key=tour_length)"
)
EXACT = (
    "@functools.cache\n"
    "def segment(A, Bs, C):\n"
    "    if not Bs:\n"
    "        return [A, C]\n"
    "    return min((segment(A, Bs - {B}, B) + [C] for B in Bs), key=lambda s: tour_length(s) - abs(s[0] - s[-1]))\n"
    "def held_karp(c):\n"
    "    A = min(c, key=abs)\n"
    "    segment.cache_clear()\n"
    "    return min((segment(A, c - {A, C}, C) for C in c - {A}), key=tour_length)\n"
    "exact = (greedy, held_karp)\n"
    "inexact = (greedy, rep, ensemble)"
)
BENCH = (
    "class Runs:\n"
    "    def __init__(self, tours):\n"
    "        self.lengths = [tour_length(t) for t in tours]\n"
    "        self.shortest = None\n"
    "@functools.cache\n"
    "def benchmark(tsp, suite):\n"
    "    time.sleep(0.1)\n"
    "    return Runs(tsp(c) for c in suite)\n"
    "def benchmarks(tsps, suite):\n"
    "    results = [benchmark(t, suite) for t in tsps]\n"
    "    best = min(mean(r.lengths) for r in results)\n"
    "    for r in results:\n"
    "        r.shortest = best\n"
    "    return [round(mean(r.lengths), 3) for r in results]"
)
CELLS = [
    "%cash_badge print\n" + SETUP,
    CITIES,
    SOLVERS,
    EXACT,
    BENCH,
    "big = held_karp(random_cities(9))",
    "b1 = benchmarks(inexact, test_suite(4, 12))",
    "b2 = benchmarks(exact, test_suite(4, 7))",
    "x = random.random()",
]
#: Below the 8 * 2**7 answers ``held_karp`` of 9 cities leaves in
#: ``segment``, above the 6 * 2**5 of 7 cities.
MAX_KEYED = 256


def _plain() -> dict:
    """What a Run All in plain Python leaves."""
    import random

    state = random.getstate()
    ns: dict = {"__name__": "__plain__"}
    try:
        for cell in CELLS:
            exec(cell.replace("%cash_badge print\n", ""), ns)
        return {name: ns[name] for name in ("b1", "b2", "x")} | {"segment": ns["segment"].cache_info().currsize}
    finally:
        random.setstate(state)


def test_the_benchmark_cells_are_served_on_the_next_run_all(nb_runner):
    plain = _plain()
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.peek(f"setattr(__import__('cash.notebook.helper_memos', fromlist=['_']), 'MAX_KEYED', {MAX_KEYED})")
    nb_runner.run_all()
    nb_runner.run_all()
    for cell in (7, 8):
        assert shows_cached(nb_runner.get_output(cell)), (cell, nb_runner.get_output(cell))
    for name in ("b1", "b2"):
        assert nb_runner.peek(name) == repr(plain[name]), name
    assert float(nb_runner.peek("x")) == plain["x"]
    # The memo the exact cell emptied holds what its run left.
    assert nb_runner.peek("segment.cache_info().currsize") == repr(plain["segment"])
    # The memo the benchmark cell filled finds its answer: no body runs.
    found = "(lambda m: (benchmark(ensemble, test_suite(4, 12)), benchmark.cache_info().misses == m)[1])"
    assert nb_runner.peek(found + "(benchmark.cache_info().misses)") == "True"
