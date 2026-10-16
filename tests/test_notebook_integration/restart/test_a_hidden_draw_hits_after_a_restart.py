"""A statement that draws from an RNG inside a function hits after a restart.

``total = solve(10)`` with ``random.sample`` inside ``solve`` draws while its
text names no RNG. cash learns that as it runs and stores the entry under a
key holding the stream; the next kernel must look it up under that same key,
not under the one without the stream, which would miss on every restart.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

HELPERS = "import random, time\ndef solve(n):\n    time.sleep(0.3)\n    return sum(random.sample(range(1000), n))\n"


@pytest.mark.parametrize("seeded", [True, False])
def test_a_hidden_draw_hits_after_a_restart(nb_runner, seeded):
    seed = "random.seed(7)" if seeded else "x = 1"
    cells = ["%cash_badge print", HELPERS, seed, "total = solve(10)\nprint('T', total)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(4)
    assert "T " in first, first
    nb_runner.settle_writes()

    for rep in range(2):
        nb_runner.restart()
        nb_runner._init_cash()
        nb_runner.run_all()
        again = nb_runner.get_output(4)
        assert shows_cached(again), f"rep {rep}: {again!r}"
        assert again.split("T ")[1].split()[0] == first.split("T ")[1].split()[0], (first, again)


def test_a_different_seed_still_misses(nb_runner):
    """What was learnt keys the stream in: a new seed above re-runs the draw."""
    cells = ["%cash_badge print", HELPERS, "random.seed(7)", "total = solve(10)\nprint('T', total)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.settle_writes()

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.set_cell_source(3, "random.seed(8)")
    nb_runner.run_all()
    out = nb_runner.get_output(4)
    assert not shows_cached(out), out
