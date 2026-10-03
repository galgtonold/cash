"""Both lineage engines move a generator on where a statement draws from it.

A statement that draws from ``rng`` without rebinding it gives ``rng`` a new
lineage when it runs and when it is restored (``carrier_advances``). The
upstream simulation never runs anything, so it reads which generators the
statement drew from (this session's record, or the entry's metadata after a
restart) and gives the same lineage. Where the two disagree, the next cell
reading ``rng`` sees it "changed" with nothing changed and re-runs the cells
above it.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

np = pytest.importorskip("numpy")

CELLS = [
    "import cash\n%cash_on\n",
    "import numpy as np, time\nrng = np.random.default_rng(7)\n",
    "def draw(k, r):\n    time.sleep(0.2)\n    return round(float(r.normal()) + k, 9)\nd = [draw(k, rng) for k in range(3)]\n",
    "x = draw(10, rng)\n",
    "print('REPORT', d, x)\n",
]
NEXT = "print('NEXT', round(float(rng.normal()), 9))\n"


def _oracle() -> tuple[str, str]:
    r = np.random.default_rng(7)
    d = [round(float(r.normal()) + k, 9) for k in range(3)]
    x = round(float(r.normal()) + 10, 9)
    return f"REPORT {d} {x}", f"NEXT {round(float(r.normal()), 9)}"


def _disagreements(t):
    return [
        (r["cell_idx"], r["vars"])
        for r in t.run_all + t.rerun
        if r.get("event") == "lineage_disagreement" and r["vars"]
    ]


def test_a_cell_after_two_draws_runs_nothing_above(upstream_trace, nb_runner):
    t = upstream_trace(CELLS, lambda r: r.run_cell(5))
    assert not _disagreements(t)
    assert not [r["stmt"] for r in t.run_all + t.rerun if r.get("event") == "schedule_reexec"]
    assert _oracle()[0] in nb_runner.get_output(5), nb_runner.get_output(5)


def test_after_a_restart_the_last_cell_alone_draws_where_a_full_run_does(upstream_trace, nb_runner):
    """After a restart only the entries' metadata says which statements drew."""

    def restart_then_the_last_two_cells(r):
        r.restart()
        r.run_cell(1)
        r.run_cell(5)
        r.run_cell(6)

    upstream_trace([*CELLS, NEXT], restart_then_the_last_two_cells)
    report, after = _oracle()
    assert report in nb_runner.get_output(5), nb_runner.get_output(5)
    assert after in nb_runner.get_output(6), nb_runner.get_output(6)


def test_a_loop_that_draws_moves_the_generator_alike_in_both_engines(upstream_trace, nb_runner):
    loop_cells = [
        CELLS[0],
        CELLS[1],
        "d = []\nfor k in range(3):\n    d.append(round(float(rng.normal()) + k, 9))\n",
        CELLS[3].replace(
            "x = ", "def draw(k, r):\n    time.sleep(0.2)\n    return round(float(r.normal()) + k, 9)\nx = "
        ),
        CELLS[4],
    ]
    t = upstream_trace(loop_cells, lambda r: r.run_cell(5))
    assert not _disagreements(t)
    assert not [r["stmt"] for r in t.run_all + t.rerun if r.get("event") == "schedule_reexec"]
    assert _oracle()[0] in nb_runner.get_output(5), nb_runner.get_output(5)
