"""A pyplot call that draws on the current figure changes the names bound to it.

``plt.plot(...)`` draws on ``plt.gca()``, which is the ``ax`` of ``fig, ax =
plt.subplots()`` above. After the drawing cell is edited and the notebook runs
again, a cell reading ``ax`` must not be served from the cache with the old
drawing. Compared with plain Python.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = (
    "import time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "def slow(v):\n    time.sleep(0.2)\n    return v"
)

CASES = {
    "a line": ("plt.plot([1, 2, 3])", "plt.plot([1, 2, 3]); plt.plot([3, 2, 1])", "len(ax.lines)", "1", "2"),
    "bars in a loop": (
        "for k in range(2):\n    plt.bar([1, 2], [k, k])",
        "for k in range(3):\n    plt.bar([1, 2], [k, k])",
        "len(ax.patches)",
        "4",
        "6",
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_an_edited_draw_reaches_the_reader(nb_runner, name):
    draw, edited, count, before, after = CASES[name]
    nb_runner.create_notebook([SETUP, "fig, ax = plt.subplots()", draw, f"r = slow({count})\nprint(r)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == before
    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == after
