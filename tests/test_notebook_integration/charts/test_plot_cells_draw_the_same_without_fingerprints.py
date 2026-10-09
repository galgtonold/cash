"""Plot cells draw what plain Python draws, over Run Alls and a restart,
now that their figures, axes and the axes a loop visits are hashed by
lineage and identity rather than by a pickle of the figure.

A plot statement runs every time; what decides a hit elsewhere -- a value
read from the figure, a loop body over its axes -- must still see the
figure the run drew.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    "data = [i % 7 for i in range(500)]",
    "fig, ax = plt.subplots()\nax.hist(data, bins=7)\nax.set_title('h')\nplt.show()",
    "fig2, axes = plt.subplots(1, 3)\nfor g, a in zip(range(3), axes):\n    a.plot([g, g + 1, g * 2])\n    drawn = len(a.lines)\nplt.show()",
    "fig3, ax3 = plt.subplots()\nfor a in [ax3, ax3]:\n    a.plot([1, 2])\n    seen = len(a.lines)",
    "check = (ax.get_title(), len(ax.patches), [len(a.lines) for a in axes], drawn, seen, len(ax3.lines), ax.figure is fig)",
]


def test_plot_cells_draw_the_same_without_fingerprints(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    expected = "('h', 7, [1, 1, 1], 1, 2, 2, True)"
    for label in ("first", "second", "third"):
        nb_runner.run_all()
        assert nb_runner.peek("check") == expected, f"{label} Run All"
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert nb_runner.peek("check") == expected, "after a restart"
