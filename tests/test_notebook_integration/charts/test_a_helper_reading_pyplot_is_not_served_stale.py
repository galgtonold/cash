"""A helper that reads pyplot's current figure runs every time, however it
reaches pyplot.

A method of the notebook's own class, pyplot handed in as an argument,
``plt.gca`` kept as a default, or pyplot imported inside the helper: each
was cached, and after an edit to the drawing and Run All it returned the
old drawing's answer.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

IMPORTS = (
    "import cash\n%cash_on\nimport time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
)
HELPERS = {
    "method": (
        "class Insp:\n    def nlines(self):\n        time.sleep(0.3)\n"
        "        return len(plt.gca().lines), plt.gca().get_title()\ninsp = Insp()",
        "n = insp.nlines()",
    ),
    "argument": (
        "def nlines(p):\n    time.sleep(0.3)\n    return len(p.gca().lines), p.gca().get_title()",
        "n = nlines(plt)",
    ),
    "default": (
        "def nlines(get=plt.gca):\n    time.sleep(0.3)\n    return len(get().lines), get().get_title()",
        "n = nlines()",
    ),
    "local_import": (
        "def nlines():\n    import matplotlib.pyplot as mp\n    time.sleep(0.3)\n"
        "    return len(mp.gca().lines), mp.gca().get_title()",
        "n = nlines()",
    ),
}


@pytest.mark.parametrize("case", sorted(HELPERS))
def test_run_all_after_an_edited_drawing_reads_the_new_one(nb_runner, case):
    helper, call = HELPERS[case]
    nb_runner.create_notebook([IMPORTS + helper, "plt.figure()\nplt.plot([1, 2])\nplt.title('A')", call + "\nprint(n)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "(1, 'A')" in nb_runner.get_output(3), "control: the first run"
    nb_runner.set_cell_source(2, "plt.figure()\nplt.plot([1, 2])\nplt.plot([2, 1])\nplt.title('B')")

    nb_runner.run_all()

    assert "(2, 'B')" in nb_runner.get_output(3), nb_runner.get_output(3)
