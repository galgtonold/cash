"""A helper that reads pyplot's current figure follows an edited drawing.

``n = nlines()``, where ``nlines`` returns ``len(plt.gca().lines)`` and the
title of the current axes, reads pyplot's current figure as ``plt.gca()``
in the statement would. After the drawing cell above is edited and the
notebook runs again, it answers for the new drawing, as plain Python does.
"""

import pytest

pytest.importorskip("matplotlib")

pytestmark = [pytest.mark.timeout(300)]

HELPER = (
    "import time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "def nlines():\n    time.sleep(0.3)\n    return len(plt.gca().lines), plt.gca().get_title()"
)
ONE = "plt.figure()\nplt.plot([1, 2])\nplt.title('A')"
TWO = "plt.figure()\nplt.plot([1, 2])\nplt.plot([2, 1])\nplt.title('B')"


def test_an_edited_drawing_is_read_again(nb_runner):
    nb_runner.create_notebook([HELPER, ONE, "n = nlines()\nprint(n)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "(1, 'A')"
    nb_runner.set_cell_source(2, TWO)
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "(2, 'B')"
