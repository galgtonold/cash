"""A repair restores a loop run as one unit that changed a list in place.

Such a loop stores what it changed with the lineages its statement gave
them, then moves them on, as after any loop that changes a value in place.
The simulation took the moved lineages from the loop's recorded outcome and
asked the entry for them, so the entry never matched: every repair that
needed the loop's list replayed the loop, re-running what it reads first,
in the session that stored it and after a restart alike. The runtime now
records the lineages the entry stores; the entry is checked for those, and
what it restores is moved on as at a hit.
"""

import ast

import pytest

pytestmark = [pytest.mark.timeout(300)]

XS = "import time\nxs = list(range(3000))"
LOOP = "acc = []\nfor x in xs:\n    time.sleep(0.0001)\n    acc.append(x * 2)"
POP = "last = acc.pop()"
SHOW = "print(len(acc), last)"
STATUSES = (
    "[(s['code'].splitlines()[0], str(s['status'])) "
    "for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"
)


def _statuses(nb_runner) -> dict[str, str]:
    """Each statement of the last cell run and of the repair before it, by
    its first line, with its status."""
    pairs = ast.literal_eval(nb_runner.peek(STATUSES))
    assert len({code for code, _ in pairs}) == len(pairs), pairs
    return dict(pairs)


def _restored_without_rerunning(nb_runner, above: tuple[str, ...]) -> None:
    """The loop came back from its entry: nothing it reads ran again to
    replay it."""
    statuses = _statuses(nb_runner)
    assert "RESTORED" in statuses["for x in xs:"], statuses
    for code in above:
        assert "COMPUTED" not in statuses.get(code, ""), statuses


@pytest.mark.fresh_kernel
def test_running_the_pop_again_restores_the_loop_each_time(nb_runner):
    nb_runner.create_notebook([XS, LOOP, POP, SHOW])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3])

    nb_runner.run_cell(3)
    _restored_without_rerunning(nb_runner, ("acc = []",))
    # Restored, the list leaves with the lineage the simulation gave it: the
    # next repair finds the loop's entry the same way.
    nb_runner.run_cell(3)
    _restored_without_rerunning(nb_runner, ("acc = []",))

    nb_runner.run_cell(4)
    assert nb_runner.get_output(4).strip().splitlines()[-1] == "2999 5998"
    assert nb_runner.peek("(acc[:3], acc[-1])") == "([0, 2, 4], 5996)"


@pytest.mark.fresh_kernel
def test_after_a_restart_the_pop_restores_the_loop(nb_runner):
    nb_runner.create_notebook([XS, LOOP, POP, SHOW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.settle_writes()

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(3)
    _restored_without_rerunning(nb_runner, ("import time", "xs = list(range(3000))", "acc = []"))

    nb_runner.run_cell(4)
    assert nb_runner.get_output(4).strip().splitlines()[-1] == "2999 5998"
