"""A repair restores a loop although the namespace it set aside holds the loop's objects.

An upstream repair keeps a copy of the namespace from before it, to put back
if it cannot finish. Re-running ``data = []`` left the previous ``data``
list in that copy, and with it the ``actions`` list of every session; the
loop filling ``data`` changes ``actions`` in place, so its hit found the
``actions`` it restores over held by another container, and the loop over
every row ran again: 9 s for a "Cache hit". The copy is cash's own. A
variable of the notebook's bound to the same list still is the user's, and
the loop runs instead of restoring over it.
"""

import ast

import pytest

pytestmark = [pytest.mark.timeout(300)]

RAW = "raw = [f'u{i};a|{i};b|{i + 1}' for i in range(40_000)]"
LOOP = (
    "data = []\n"
    "for row in raw:\n"
    "    uid, *parts = row.split(';')\n"
    "    actions = []\n"
    "    for p in parts:\n"
    "        actions.append(p.split('|')[0])\n"
    "    data.append((uid, actions))"
)
POP = "dropped = data.pop(0)"
STATUSES = (
    "[(s['code'].splitlines()[0], str(s['status'])) "
    "for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"
)


def _loop_status(nb_runner) -> str:
    statuses = [status for code, status in ast.literal_eval(nb_runner.peek(STATUSES)) if code == "for row in raw:"]
    assert len(statuses) == 1, statuses
    return statuses[0]


@pytest.mark.fresh_kernel
def test_running_the_pop_again_restores_the_loop(nb_runner):
    nb_runner.create_notebook([RAW, LOOP, POP, "print(len(data), dropped)"])
    nb_runner.start_kernel()
    for cell in (1, 2, 3):
        nb_runner.run_cell(cell)

    nb_runner.run_cell(3)

    assert "RESTORED" in _loop_status(nb_runner)
    nb_runner.run_cell(4)
    assert nb_runner.get_output(4).strip().splitlines()[-1] == "39999 ('u0', ['a', 'b'])"


@pytest.mark.fresh_kernel
def test_a_variable_bound_to_the_loops_list_still_runs_the_loop(nb_runner):
    nb_runner.create_notebook([RAW, LOOP, "keep = actions", POP, "print(len(data), dropped, keep)"])
    nb_runner.start_kernel()
    for cell in (1, 2, 3, 4):
        nb_runner.run_cell(cell)

    nb_runner.run_cell(4)

    assert "RESTORED" not in _loop_status(nb_runner)
    nb_runner.run_cell(5)
    assert nb_runner.get_output(5).strip().splitlines()[-1] == "39999 ('u0', ['a', 'b']) ['a', 'b']"
