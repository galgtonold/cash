"""A draw cash runs again above a cell, from where its seeded stream stands
now rather than where a run from the top has it, says so (RANDOM-REDRAWN).

The stream is seeded inside a ``functools.cache`` helper whose answers are
kept, so nothing cash can run again seeds it before the draw. Editing ``k``
and running the cell that prints makes cash run ``m`` again for it: the
numbers it draws then differ from a run from the top, never silently. A
literal seed above the draw is run again with it, and stays quiet.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

SEEDS = {
    "in a memo": "@cache\ndef rc(n):\n    random.seed(n)\n    return tuple(range(n))\nsuite = (rc(5), rc(6))",
    "written out": "random.seed(5)\nsuite = (tuple(range(5)), tuple(range(6)))",
}


@pytest.mark.parametrize("how", list(SEEDS))
def test_a_redrawn_value_warns_unless_its_seed_runs_again(nb_runner, how):
    cells = [
        "%cash_badge print\nimport random, time\nfrom functools import cache",
        SEEDS[how],
        "k = 2",
        "m = [k * random.random() for _ in suite] + [time.sleep(0.2)]",
        "print('M', m[:2])",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, "k = 3")
    nb_runner.run_cell(5)
    shown = nb_runner.get_raw_output(5)
    if how == "in a memo":
        assert "RANDOM-REDRAWN" in shown and "m = [k * random.random()" in shown, shown
    else:
        assert "RANDOM-REDRAWN" not in shown, shown
