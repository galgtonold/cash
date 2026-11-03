"""A statement whose run fills a ``functools.cache`` memo is served, and the
memo is filled as the run left it, after a restart.

Shaped like Norvig's TSP notebook: ``random_cities`` reseeds ``random`` on a
miss, ``test_suite`` and ``benchmark`` are memos too, and ``benchmarks``
returns the very objects the ``benchmark`` memo holds, then changes them.
Such a statement was never stored, so every benchmark cell ran on every
pass. Served without the memo, a later call of ``random_cities`` would run
its body and reseed the stream, so the draw below would differ from the
run's. Served with it, the draw, the memo's answers and the objects the
outputs share with them are the ones a restart and Run All in plain Python
give.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import functools, os, random, time\nLOG = 'bench_calls.log'"
NOTE = (
    "def note(tag):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (tag + '\\n').encode())\n"
    "    os.close(fd)"
)
HELPERS = (
    "@functools.cache\n"
    "def random_cities(n, seed=42):\n"
    "    random.seed(seed + n)\n"
    "    return frozenset(round(random.random(), 6) for _ in range(n))\n"
    "@functools.cache\n"
    "def test_suite(n, k):\n"
    "    return tuple(random_cities(n, 42 + i) for i in range(k))\n"
    "class Runs:\n"
    "    def __init__(self, tours):\n"
    "        self.tours = tours\n"
    "        self.shortest = None\n"
    "@functools.cache\n"
    "def benchmark(fn, data):\n"
    "    note('benchmark')\n"
    "    return Runs([fn(c) for c in data])\n"
    "def tour_a(c):\n"
    "    return sorted(c)\n"
    "def tour_b(c):\n"
    "    return sorted(c, reverse=True)\n"
    "def benchmarks(fns, data):\n"
    "    runs = [benchmark(f, data) for f in fns]\n"
    "    for r in runs:\n"
    "        r.shortest = min(t[0] for t in r.tours)\n"
    "    time.sleep(0.3)\n"
    "    return runs"
)
CELLS = [
    "%cash_badge print\n" + SETUP,
    NOTE,
    HELPERS,
    "res = benchmarks((tour_a, tour_b), test_suite(5, 3))",
    "res2 = benchmarks((tour_a,), test_suite(6, 2))",
    "x = random.random()\nprint('X', x)",
]


def _plain() -> dict:
    """What restart and Run All in plain Python leave."""
    import random

    state = random.getstate()
    ns: dict = {"__name__": "__plain__"}
    try:
        for cell in CELLS:
            exec(cell.replace("%cash_badge print\n", "").replace("'bench_calls.log'", "os.devnull"), ns)
        return {
            "x": ns["x"],
            "sizes": (ns["random_cities"].cache_info().currsize, ns["benchmark"].cache_info().currsize),
            "shortest": ns["res"][0].shortest,
        }
    finally:
        random.setstate(state)


def _calls(nb_runner) -> int:
    return int(nb_runner.peek("len(open(LOG).read().split()) if os.path.exists(LOG) else 0"))


def test_the_benchmark_cells_are_served_with_the_memo_filled(nb_runner):
    plain = _plain()
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _calls(nb_runner) == 3
    nb_runner.settle_writes()

    for rep in range(2):
        nb_runner.restart()
        nb_runner._init_cash()
        nb_runner.run_all()
        for cell in (4, 5):
            assert shows_cached(nb_runner.get_output(cell)), (rep, cell, nb_runner.get_output(cell))
        # Served: the benchmark body ran in no later pass.
        assert _calls(nb_runner) == 3, rep
        # The draw below is the plain run's: no call found its memo empty
        # and reseeded the stream.
        assert float(nb_runner.peek("x")) == plain["x"], rep
        sizes = nb_runner.peek("(random_cities.cache_info().currsize, benchmark.cache_info().currsize)")
        assert sizes == repr(plain["sizes"]), rep
        # The output and the memo's answer are one object, changed as the run changed it.
        assert nb_runner.peek("res[0] is benchmark(tour_a, test_suite(5, 3))") == "True", rep
        assert float(nb_runner.peek("benchmark(tour_a, test_suite(5, 3)).shortest")) == plain["shortest"], rep
        # A later call finds the answer, and runs no body.
        nb_runner.peek("random_cities(5, 42)")
        assert nb_runner.peek("random_cities.cache_info().misses") == "0", rep


def test_a_cell_above_that_fills_a_key_the_statement_computed_reruns_it(nb_runner):
    """Control: once a cell above fills ``benchmark(tour_a, ...)``, a run of
    the statement finds the answer, and the entry -- which would put its own
    copy into the memo -- is not served."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.settle_writes()
    edited = "first = benchmark(tour_a, test_suite(5, 3))"
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.set_cell_source(4, edited + "\nres = benchmarks((tour_a, tour_b), test_suite(5, 3))")
    nb_runner.run_all()
    assert nb_runner.peek("res[0] is first") == "True"
    assert nb_runner.peek("benchmark(tour_a, test_suite(5, 3)) is first") == "True"
