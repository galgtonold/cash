"""A change made through another name is rebuilt after a restart.

``y = x`` then ``y.append(2)``: after a restart, running only a cell that
reads ``x`` rebuilt ``x`` and ``y`` but skipped ``y.append(2)``, so it read
``[1]``. Which names share an object was known only to the kernel that ran
the cells; each statement now keeps what it linked for a later kernel. The
same for a frame changed through a second name, a row of an array changed
through a view, a dict changed through a list holding it, and a figure
drawn through pyplot.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\nimport time\nimport numpy as np\nimport pandas as pd"

CASES = {
    "list": ("x = [1]", "y = x", "y.append(2)", "print(x)", "[1, 2]"),
    "frame": (
        "df = pd.DataFrame({'a': [1, 2]})",
        "d2 = df",
        "d2['b'] = d2['a'] * 10",
        "print(df.columns.tolist())",
        "['a', 'b']",
    ),
    "view": ("big = np.arange(6.0)", "head = big[:3]", "head *= 2", "print(big)", "[0. 2. 4. 3. 4. 5.]"),
    "holder": ("cfg = {'lr': 1}", "runs = [cfg]", "runs[0]['lr'] = 5", "print(cfg)", "{'lr': 5}"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_cell_run_alone_after_a_restart_sees_the_change(nb_runner, case):
    make, alias, change, read, expected = CASES[case]
    nb_runner.create_notebook([SETUP, make, alias, change, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(5), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner.run_cells([1, 5])

    assert expected in nb_runner.get_output(5), nb_runner.get_output(5)


FIGURE = [
    "import cash\n%cash_on\nimport time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "def nlines(f):\n    time.sleep(0.3)\n    return [len(a.lines) for a in f.axes], [a.get_title() for a in f.axes]",
    "fig = plt.figure()",
    "plt.plot([1, 2])\nplt.title('A')",
    "n = nlines(fig)\nprint(n)",
]


def test_a_figure_drawn_through_pyplot_is_redrawn_after_a_restart(nb_runner):
    nb_runner.create_notebook(FIGURE)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "([1], ['A'])" in nb_runner.get_output(4), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner.run_cells([1, 4])

    assert "([1], ['A'])" in nb_runner.get_output(4), nb_runner.get_output(4)
