"""A widget or animation holding the user's axes and figure keeps them.

``Button(ax, ...)`` and ``FuncAnimation(fig, ...)`` hold the user's ``ax`` and
``fig``, also when a helper reaches them through pyplot (``plt.gca()``,
``plt.gcf()``). A figure is held back by its axes, so the share check cannot
finish for what the widget holds and the statement is not stored: on every
run ``w``'s axes are ``ax``, and a change to ``ax`` shows through ``w``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(300)]

SETUP = (
    "import cash\n%cash_on\nimport time\nimport matplotlib\nmatplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\nfrom matplotlib.widgets import Button\n"
    "from matplotlib.animation import FuncAnimation"
)
FIG = "fig, ax = plt.subplots()\nline, = ax.plot([1, 2, 3])\nt0 = time.perf_counter()"
CELLS = {
    "button": "time.sleep(1.0)\nw = Button(ax, 'go')",
    "button_by_helper": "def mk():\n    return Button(plt.gca(), 'go')\ntime.sleep(1.0)\nw = mk()",
    "anim_by_helper": "def mk():\n    return FuncAnimation(plt.gcf(), lambda i: [], frames=2, cache_frame_data=False)\ntime.sleep(1.0)\nw = mk()",
    "anim": "time.sleep(1.0)\nw = FuncAnimation(fig, lambda i: [], frames=2, cache_frame_data=False)",
}
TOOK = "took = time.perf_counter() - t0"
CHANGE = "ax.set_title('edited')\nline.set_color('red')"
READ = (
    "a = getattr(w, 'ax', None) or w._fig.axes[0]\n"
    "print('SAME', a is ax, a.figure is fig, repr(a.get_title()), a.lines[0] is line, a.lines[0].get_color())"
)


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_widget_keeps_users_axes(nb_runner, cell):
    nb_runner.create_notebook([SETUP, FIG, CELLS[cell], TOOK, CHANGE, READ])
    nb_runner.start_kernel()
    outs = []
    for i in range(2):
        nb_runner.run_all()
        outs.append((nb_runner.get_raw_output(6), nb_runner.peek("took")))
    nb_runner.restart()
    nb_runner.run_all()
    outs.append((nb_runner.get_raw_output(6), nb_runner.peek("took")))
    for out, _ in outs:
        assert "SAME True True 'edited' True red" in out, outs
