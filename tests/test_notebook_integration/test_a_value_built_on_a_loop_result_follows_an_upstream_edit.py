"""A value built on a loop's result is rebuilt when the loop re-ran on new data.

A user swept settings in a loop (``sweep_rows.append(...)``), picked
the best row in the next cell (``best``, ``BEST_W``, ``best_scores``) and read
those in a later analysis cell. After editing a constant above the sweep and
re-running that cell and the sweep, the analysis cell printed the old episode
count and precision: the cell between the sweep and the analysis was never
re-run, and nothing on screen said so.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

SWEEP = "rows = []\nfor w in [1, 2, 3]:\n    rows.append(w * N)\ntotal = sum(rows)"


@pytest.mark.parametrize(
    "pick, shown",
    [
        ("best = total + 1", "print('BEST', best)"),
        ("best = total + 1\nscores = [best * k for k in range(2)]", "print('BEST', scores[1])"),
        ("def score(k):\n    return k * sum(rows)\nbest = score(1) + 1", "print('BEST', best)"),
    ],
    ids=["direct", "via_comprehension", "via_function_global"],
)
def test_a_cell_below_the_pick_shows_the_new_sweep(nb_runner, pick, shown):
    nb_runner.create_notebook(["import cash\n%cash_on", "N = 1", SWEEP, pick, shown])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "BEST 7" in nb_runner.get_output(5), nb_runner.get_output(5)

    nb_runner.set_cell_source(2, "N = 2")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    nb_runner.run_cell(5)

    assert "BEST 13" in nb_runner.get_output(5), nb_runner.get_output(5)
