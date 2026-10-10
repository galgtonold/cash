"""A cell that starts with ``plt.figure(...)`` draws on a new figure, as in plain Python.

Two cells open with the same ``plt.figure(figsize=(20, 10))`` text and draw
different charts: the first plots dates, the second bars of categories. The
statement has no inputs but the ``plt`` module, so its key was the same in
both cells. Once the first one was stored (it took more than the persistence
floor), the second was a hit: no new figure was made, and the bars landed on
the date axis. matplotlib warned "This axis already has a converter set", and
a later date plot raised ``TypeError: tz must be string or tzinfo subclass``.
A call that makes, picks or draws on pyplot's current figure now always runs.

``%cash_persist on`` stores every statement, so the test does not depend on
how long ``plt.figure`` happens to take.
"""

import pytest

pytest.importorskip("matplotlib")

pytestmark = pytest.mark.libraries

SETUP = (
    "import cash\n%cash_on\n%cash_persist on\n"
    "import datetime as dt\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "dates = [dt.date(2020, 1, d) for d in range(1, 29)]"
)

#: For each open figure: (lines, bar patches) on its one axes.
FIGURES = "[(len(plt.figure(n).axes[0].lines), len(plt.figure(n).axes[0].patches)) for n in plt.get_fignums()]"


@pytest.mark.timeout(90)
def test_the_second_figure_cell_gets_its_own_figure(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "plt.figure(figsize=(20, 10))\nplt.plot(dates, range(28))",
            "plt.figure(figsize=(20, 10))\nplt.bar(['north', 'south'], [3, 5])",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "Error" not in nb_runner.get_output(3)
    # Plain Python: the date line on figure 1, the two bars on figure 2.
    assert nb_runner.peek(FIGURES) == "[(1, 0), (0, 2)]"


@pytest.mark.timeout(90)
def test_a_library_drawing_on_the_current_figure_runs_every_time(nb_runner):
    """pandas draws on pyplot's current axes without any ``plt.`` in the
    statement; a hit of ``s.plot()`` left the new figure empty."""
    pytest.importorskip("pandas")
    nb_runner.create_notebook(
        [
            SETUP + "\nimport pandas as pd\ns = pd.Series(range(28), index=dates)",
            "plt.figure(figsize=(20, 10))\ns.plot()\nprint('first')",
            "plt.figure(figsize=(20, 10))\ns.plot()\nprint('second')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek(FIGURES) == "[(1, 0), (1, 0)]"


@pytest.mark.timeout(120)
def test_seaborn_date_plot_after_a_bar_plot_does_not_raise(nb_runner):
    """The notebook the bug came from: a seaborn date line plot, a bar plot,
    then a date line plot again, each after ``plt.figure(figsize=(20, 10))``."""
    pytest.importorskip("seaborn")
    pytest.importorskip("pandas")
    nb_runner.create_notebook(
        [
            SETUP + "\nimport pandas as pd\nimport seaborn as sns\n"
            "data = pd.DataFrame({'date': dates * 2, 'action': ['open'] * 28 + ['close'] * 28, 'n': range(56)})",
            "plt.figure(figsize=(20, 10))\nsns.lineplot(data=data, x='date', y='n', hue='action')\nprint('lines')",
            "plt.figure(figsize=(20, 10))\ncounts = data.groupby('action')['n'].sum().reset_index()\n"
            "graph = sns.barplot(x='action', y='n', data=counts)",
            "plt.figure(figsize=(20, 10))\ngraph = sns.lineplot(x='date', y='n', hue='action', data=data)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    for cell in (3, 4):
        out = nb_runner.get_output(cell)
        assert "converter" not in out and "Error" not in out, out
    assert nb_runner.peek("len(plt.get_fignums())") == "3"
