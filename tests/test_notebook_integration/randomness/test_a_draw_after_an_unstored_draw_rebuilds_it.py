"""A draw run alone after a restart starts where the draws above it left off.

Seen in a notebook::

    rng = np.random.default_rng(42)              # cell 2
    x0 = rng.normal()                            # cell 3, cheap, never stored
    res = {g: boot(g, rng) for g in groups}      # cell 4, slow, stored
    top = max(res, key=res.get)                  # cell 5

After Run All and a kernel restart, running cell 4 alone first rebuilds
``rng``. The second time it was run alone, the rebuild re-ran
``rng = ...`` but not the unstored draw ``x0``, so ``res`` came from a fresh
generator and cell 5 printed a value no top-to-bottom run gives. These pin
that the rebuild leaves the generator where a top-to-bottom run has it, for
a generator passed as an argument and for one a helper reads as a global.
The oracle is the same cells run top to bottom without cash.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(180)]

np = pytest.importorskip("numpy")

SETUP = "import cash\n%cash_on"
SEED = "import numpy as np, time\nrng = np.random.default_rng(42)"
CHEAP_DRAW = "x0 = rng.normal()"
TOP = "top = max(res, key=res.get)\nprint('TOP', top, round(sum(res.values()), 9))"

ARGUMENT = (
    "def boot(g, r):\n"
    "    time.sleep(0.2)\n"
    "    return round(float(r.normal()) + g, 9)\n"
    "res = {g: boot(g, rng) for g in range(4)}"
)
GLOBAL_HELPER = (
    "def boot(g):\n"
    "    time.sleep(0.2)\n"
    "    return round(float(rng.normal()) + g, 9)\n"
    "res = {g: boot(g) for g in range(4)}"
)


def _oracle() -> str:
    r = np.random.default_rng(42)
    r.normal()
    res = {g: round(float(r.normal()) + g, 9) for g in range(4)}
    return f"TOP {max(res, key=res.get)} {round(sum(res.values()), 9)}"


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("draw", [ARGUMENT, GLOBAL_HELPER], ids=["argument", "global_helper"])
def test_a_draw_run_alone_after_a_restart_starts_after_the_unstored_draw(nb_runner, draw):
    nb_runner.create_notebook([SETUP, SEED, CHEAP_DRAW, draw, TOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert _oracle() in out, out

    nb_runner.restart()
    nb_runner.run_cell(1)
    for attempt in range(3):
        nb_runner.run_cell(4)
        nb_runner.run_cell(5)
        out = nb_runner.get_output(5)
        assert _oracle() in out, f"run {attempt + 1} of the draw alone: {out}"
