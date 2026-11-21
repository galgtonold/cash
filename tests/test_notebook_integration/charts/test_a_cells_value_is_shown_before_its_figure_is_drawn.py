"""A cell's value is shown before the inline backend draws its figure, as
without cash.

IPython shows the value of the cell's last line, then draws the open figures
after the cell (``post_execute``). cash drew them itself at the end of the
cell's statements, to keep what they showed, before the value was shown: the
``Text`` that ``plt.ylabel`` returns then showed where the draw had placed it
(``Text(23.26, 0.5, ...)``), not ``Text(0, 0.5, ...)``. A figure that fails to
draw lost the error line IPython prints for pyplot's own redraw.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

pytest.importorskip("matplotlib")

PLOT = "f = plt.figure(figsize=(4, 3))\nplt.plot([1, 2], [1, {n}])\nplt.ylabel('y label')"


def test_the_returned_text_shows_where_the_cell_put_it(nb_runner):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "%matplotlib inline\nimport matplotlib.pyplot as plt",
            PLOT.format(n=2),
            PLOT.format(n=2) + "  # again",
            PLOT.format(n=3),
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    for cell in (3, 4, 5):
        shown = nb_runner.get_output(cell)
        assert "Text(0, 0.5, 'y label')" in shown, (cell, shown)
        assert any("image/png" in o.get("data", {}) for o in nb_runner.get_cell(cell).outputs), cell


def test_a_figure_that_fails_to_draw_shows_pyplots_error_line(nb_runner):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "%matplotlib inline\nimport matplotlib.pyplot as plt",
            "def bad():\n    plt.figure()\n    plt.title(r'$\\frac{1}$')\nbad()",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3])

    assert "Error in callback" in nb_runner.get_output(3)
    errors = [o.ename for o in nb_runner.get_cell(3).outputs if o.output_type == "error"]
    assert errors == ["ValueError", "ValueError"]
