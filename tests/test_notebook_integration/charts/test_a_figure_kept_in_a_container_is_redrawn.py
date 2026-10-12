"""A figure kept in a dict or a list is redrawn with its drawing cell.

``figs = {'main': plt.figure()}`` then ``plt.plot(...)`` in the next cell:
after an edit to the drawing cell, running only a cell that reads the figure
re-made it without the drawing, as only names bound to the figure itself
counted as drawn on. A container holding it counts too now.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = (
    "import cash\n%cash_on\nimport time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "def nlines(f):\n    time.sleep(0.3)\n    return [len(a.lines) for a in f.axes], [a.get_title() for a in f.axes]"
)
FIRST = "plt.plot([1, 2])\nplt.title('A')"
EDITED = "plt.plot([1, 2])\nplt.plot([2, 1])\nplt.title('B')"
CASES = {
    "dict": ("figs = {'main': plt.figure()}", "n = nlines(figs['main'])\nprint(n)"),
    "list": ("figs = []\nfor i in range(2):\n    figs.append(plt.figure())", "n = nlines(figs[-1])\nprint(n)"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_an_edited_drawing_reaches_the_figure_in_the_container(nb_runner, case):
    make, read = CASES[case]
    nb_runner.create_notebook([SETUP, make, FIRST, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "([1], ['A'])" in nb_runner.get_output(4), "control: the top-to-bottom run"
    nb_runner.set_cell_source(3, EDITED)

    nb_runner.run_cells([4])

    assert "([2], ['B'])" in nb_runner.get_output(4), nb_runner.get_output(4)
