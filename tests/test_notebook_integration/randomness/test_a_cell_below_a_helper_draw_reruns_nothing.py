"""A cell below a helper that draws from a global generator runs alone.

Seen in a notebook::

    rng = np.random.default_rng(42)              # cell 3
    x0 = rng.normal()                            # cell 4
    def boot(g): ...rng.normal()...              # cell 5, reads rng as a global
    res = {g: boot(g) for g in groups}           # cell 6, slow, stored
    top = max(res, key=res.get)                  # cell 7

The draw in cell 6 moves ``rng`` on, and ``rng`` gets a new lineage for it.
The upstream simulation keyed cell 6 on the lineage ``rng`` holds now, after
the draw, instead of the one it held when cell 6 ran: the key missed, the
simulated ``rng`` came out elsewhere than the live one, and every run of
cell 7 re-ran the bootstrap above it, from the moved generator. The oracle
is the same cells run top to bottom without cash.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_executed

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

np = pytest.importorskip("numpy")

SETUP = "import cash\n%cash_on\n%cash_badge print"
CELLS = [
    SETUP,
    "import numpy as np, time",
    "rng = np.random.default_rng(42)",
    "x0 = rng.normal()",
    "def boot(g):\n    time.sleep(0.03)\n    return round(float(rng.normal()) + g, 9)",
    "res = {g: boot(g) for g in range(4)}",
    "top = max(res, key=res.get)\nprint('TOP', top, round(sum(res.values()), 9))",
]


def _oracle(first_draw: bool = True) -> str:
    r = np.random.default_rng(42)
    if first_draw:
        r.normal()
    res = {g: round(float(r.normal()) + g, 9) for g in range(4)}
    return f"TOP {max(res, key=res.get)} {round(sum(res.values()), 9)}"


@pytest.mark.parametrize("first_draw", [True, False], ids=["after_a_draw", "first_draw"])
def test_the_cell_below_a_helper_draw_does_not_rerun_it(nb_runner, first_draw):
    cells = CELLS if first_draw else [c for c in CELLS if c != "x0 = rng.normal()"]
    below = len(cells)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(below)
    assert _oracle(first_draw) in out, out
    assert "^EXECUTED: res" not in out, "a top-to-bottom run re-ran the draw above"

    for _ in range(3):
        nb_runner.run_cell(below)
        out = nb_runner.get_output(below)
        assert _oracle(first_draw) in out, out
        assert not shows_executed(out.replace("EXECUTED: top", "").replace("EXECUTED: print", "")), out


def test_rerunning_the_helper_draw_alone_restores_it(nb_runner):
    """Run alone, the draw first rebuilds ``rng`` where a top-to-bottom run
    has it, as for a generator named in the cell, so its entry is current:
    it is restored, and the cell below it reads the top-to-bottom values."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()

    for _ in range(2):
        nb_runner.run_cell(6)
        out = nb_runner.get_output(6)
        assert "CACHED: res" in out and "EXECUTED: res" not in out, out
        nb_runner.run_cell(7)
        out = nb_runner.get_output(7)
        assert _oracle() in out, out
        assert "EXECUTED: res" not in out, out
