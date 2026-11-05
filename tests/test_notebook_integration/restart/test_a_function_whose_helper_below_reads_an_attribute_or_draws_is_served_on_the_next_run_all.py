"""A function whose helper, defined below it in the same cell, reads an
attribute or draws with pyplot keeps one lineage, so what holds it is
served on the next Run All.

The TSP notebook defines ``mst_tsp`` above ``preorder_traversal``, which
calls ``tree.get``, and ``plot_tour`` above ``plot_segment``, which calls
``plt.plot``. The function's lineage leaves such a helper out, but two
things the ``def`` statement took from the helper's body did not: the
globals its code names (``get``, ``plot``), which the key's walk of the
called functions added once the helper was bound, and the verdict that the
``def`` "reads pyplot's current figure through plot_segment", which put the
function's value into its lineage. Neither is there on a first Run All or
after a restart, both are on the next Run All, so the tuples of solvers
holding the function took another lineage, and every benchmark cell
reading them missed ("input changed since it was last stored:
benchmarks, inexact_tsps").
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "%cash_badge print\nimport time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"
#: ``walk`` calls ``preorder``, defined below it, which reads ``tree.get``.
TREE = (
    "def walk(tree):\n"
    "    return list(preorder(tree, 0))\n"
    "\n"
    "def preorder(tree, root):\n"
    "    yield root\n"
    "    for child in tree.get(root, ()):\n"
    "        yield from preorder(tree, child)"
)
#: ``walk`` calls ``plot_segment``, defined below it, which draws.
PLOT = (
    "def walk(tree, plot=False):\n"
    "    if plot:\n"
    "        plt.figure()\n"
    "        plot_segment(list(tree))\n"
    "    return sorted(tree) + [c for kids in tree.values() for c in kids]\n"
    "\n"
    "def plot_segment(segment):\n"
    "    plt.plot(segment)"
)
HOLDER = "solvers = (len, walk)"
CALL = "r = [f({0: [1, 2], 1: [3]}) for f in solvers if not time.sleep(0.15)]\nprint('R', r)"
WANT = {TREE: "R [2, [0, 1, 3, 2]]", PLOT: "R [2, [0, 1, 1, 2, 3]]"}


@pytest.mark.parametrize("helper", [TREE, PLOT], ids=["reads_an_attribute", "draws_with_pyplot"])
def test_what_holds_the_function_is_served_on_the_next_run_all_and_after_a_restart(nb_runner, helper):
    nb_runner.create_notebook([SETUP, helper, HOLDER, CALL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert WANT[helper] in nb_runner.get_output(4), nb_runner.get_output(4)

    nb_runner.run_all()
    out = nb_runner.get_output(4)
    assert WANT[helper] in out, out
    assert shows_cached(out), out

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    out = nb_runner.get_output(4)
    assert WANT[helper] in out, out
    assert shows_cached(out), out


def test_an_edit_to_the_helper_below_runs_the_call_again(nb_runner):
    """Control: the helper is read at the call, so new code in it re-runs
    the call, in the same kernel and after a restart."""
    nb_runner.create_notebook([SETUP, TREE, HOLDER, CALL])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(2, TREE.replace("yield root", "yield root * 10"))
    nb_runner.run_all()
    out = nb_runner.get_output(4)
    assert "R [2, [0, 10, 30, 20]]" in out, out
    assert not shows_cached(out), out

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert "R [2, [0, 10, 30, 20]]" in nb_runner.get_output(4), nb_runner.get_output(4)
