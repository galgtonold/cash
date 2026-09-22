"""A figure's own artists are not data, so a colorbar chart is not "stale".

Round 30, r30s1 and r30s5 (3 occurrences each): running a cell printed
"^STALE FILE: report/chart_...png not rewritten, though its data changed
upstream" for a chart whose data had not changed -- compared against a
freshly drawn one it was identical. "The kind of warning that trains me to
ignore the real ones" (r30s5).

A figure is vouched for by its history: the statements from
``fig, ax = plt.subplots()`` to the write, each with the lineages of what it
read. ``sc = ax.scatter(...)`` hands back an artist, and ``fig.colorbar(sc)``
reads it -- but the runtime and the simulation give ``sc`` different lineages,
exactly as they do for ``fig`` and ``ax``, which the history already leaves
out. So no history ever matched and the chart read as stale whenever some
other repair made the planner ask.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]


def _cells(folder):
    return [
        "import cash\n%cash_on\n%cash_badge print\nimport pandas as pd\n"
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
        "K2 = 1\nSCALE = 1",
        "data = pd.DataFrame({'v': range(50), 'w': range(50)}) * SCALE",
        "fig, ax = plt.subplots()\nsc = ax.scatter(data.v, data.w, c=data.v)\n"
        "fig.colorbar(sc, label='v')\nax.set_title('a')\n"
        f"fig.savefig(r'{folder}/a.png')\nplt.close(fig)",
        "other = pd.DataFrame({'x': range(10)})",
        "other['y'] = other.x * K2",
        "print('OUT', len(data), other.y.sum())",
    ]


def test_an_unrelated_repair_does_not_call_the_chart_stale(nb_runner, tmp_path):
    nb_runner.create_notebook(_cells(tmp_path.as_posix()))
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(2, "K2 = 2\nSCALE = 1")
    nb_runner.run_cell(7)
    raw = nb_runner.get_raw_output(7)
    assert "OUT 50 90" in nb_runner.get_output(7), raw
    assert "STALE FILE" not in raw, (
        "the chart's data did not change, and it was called stale:\n" + raw)


def test_a_chart_whose_data_changed_is_still_called_stale(nb_runner, tmp_path):
    """Control: the same notebook, with the edit changing what was plotted."""
    nb_runner.create_notebook(_cells(tmp_path.as_posix()))
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(2, "K2 = 1\nSCALE = 3")
    nb_runner.run_cell(7)
    raw = nb_runner.get_raw_output(7)
    assert "STALE FILE" in raw and "a.png" in raw, (
        "the plotted data changed and the chart was not called stale:\n" + raw)
