"""The user's object a helper puts inside a library object stays the user's.

``def mk(): fig = Figure(); fig.add_subplot().add_patch(rect); return
argparse.Namespace(fig=fig)`` and ``w = mk()``: on every run, as in plain
Python, ``w``'s patch is ``rect``, so ``rect.set_width(5)`` shows through
``w``. The call to ``mk`` was served as a copy on the second run: its entry
did not read what the namespace and the figure hold.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(300)]

pytest.importorskip("matplotlib")

SETUP = (
    "import cash\n%cash_on\nimport time, argparse\nimport matplotlib\nmatplotlib.use('Agg')\n"
    "from matplotlib.figure import Figure\nfrom matplotlib.patches import Rectangle"
)
DATA = {
    "patch_on_a_figure": (
        "rect = Rectangle((0, 0), 1, 1)\n"
        "def mk():\n    time.sleep(1.0)\n    fig = Figure()\n    fig.add_subplot().add_patch(rect)\n"
        "    return argparse.Namespace(fig=fig)"
    ),
    "list_on_a_namespace": ("rect = [1]\ndef mk():\n    time.sleep(1.0)\n    return argparse.Namespace(rows=rect)"),
}
CHANGE = {"patch_on_a_figure": "rect.set_width(5.0)", "list_on_a_namespace": "rect.append(5)"}
READ = {
    "patch_on_a_figure": "p = w.fig.axes[0].patches[0]\nprint('SAME', p is rect, p.get_width())",
    "list_on_a_namespace": "print('SAME', w.rows is rect, w.rows[-1])",
}
PLAIN = {"patch_on_a_figure": "SAME True 5.0", "list_on_a_namespace": "SAME True 5"}


@pytest.mark.parametrize("case", sorted(DATA))
def test_the_object_stays_the_users_on_every_run(nb_runner, case):
    nb_runner.create_notebook([SETUP, DATA[case], "w = mk()", CHANGE[case], READ[case]])
    nb_runner.start_kernel()
    for restart in (False, False, True):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        assert PLAIN[case] in nb_runner.get_output(5), nb_runner.get_raw_output(5)
