"""A plot whose statements unbind a name, or bind one of IPython's history
names themselves, is drawn every time.

``del tmp`` between ``plt.figure()`` and ``plt.plot(...)`` is part of the
plot run. Served, its statements are skipped and ``tmp`` stays bound, where
plain Python (restart + Run All) has it gone. ``_ = plt.plot(...)`` and a
``for _ in ...`` loop bind ``_`` to a value of the cell's, which IPython's
output history then leaves alone; served, ``_`` would hold whatever was
echoed before, and a cell run later, in the notebook or not, would read
that.
"""

from __future__ import annotations

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

pytest.importorskip("matplotlib")

HEAD = (
    "%matplotlib inline\nimport os, time\nimport matplotlib.pyplot as plt\nLOG = r'{log}'\n"
    "def count():\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "def draw(*points):\n"
    "    count()\n"
    "    return plt.plot(*points)\n"
)


def _calls(log) -> int:
    return len(log.read_text(encoding="utf-8")) if log.exists() else 0


@pytest.mark.fresh_kernel
def test_a_plot_deleting_a_name_leaves_it_deleted(nb_runner, tmp_path):
    log = tmp_path / "calls.log"
    plot = "plt.figure()\ndel tmp\nplt.plot([1, 2]); count()"
    nb_runner.create_notebook([HEAD.format(log=log), "tmp = 1", plot, "r = 'tmp' in globals()"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == "False"
    nb_runner.run_all()
    assert nb_runner.peek("r") == "False", "the served plot left tmp bound"
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert nb_runner.peek("r") == "False", "the served plot left tmp bound"


PLOTS = {
    "assigned": ("_ = draw([1, 2])", "list"),
    "loop": ("plt.figure()\nfor _ in range(2):\n    plt.plot([1, _])\ndraw([2, 1])", "int"),
}


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("save", [True, False], ids=["saved", "not-saved-yet"])
@pytest.mark.parametrize("plot", list(PLOTS))
def test_a_plot_binding_underscore_leaves_it_bound(nb_runner, tmp_path, plot, save):
    log = tmp_path / "calls.log"
    code, plain = PLOTS[plot]
    nb_runner.create_notebook([HEAD.format(log=log), code])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    nb_runner.add_cell("r = type(_).__name__", save=save)
    nb_runner.run_cell(3)
    assert nb_runner.peek("r") == f"'{plain}'", "the served plot left _ as it was"
    assert _calls(log) == 3


HELPERS = (
    "def set_new():\n    import __main__\n    __main__.zz = 1\n"
    "def set_old():\n    import __main__\n    __main__.tmp = [2]\n"
)
THROUGH_MAIN = {
    "new name": ("set_new()", "r = globals().get('zz', 'unbound')", "1"),
    "existing name": ("set_old()", "r = tmp", "[2]"),
}


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("case", list(THROUGH_MAIN))
def test_a_plot_whose_helper_binds_a_name_through_main_leaves_it_bound(nb_runner, tmp_path, case):
    """A helper binding a name through ``__main__`` binds it in the
    notebook's namespace without naming it in the plot's text."""
    log = tmp_path / "calls.log"
    call, check, plain = THROUGH_MAIN[case]
    plot = f"plt.figure()\n{call}\nplt.plot([1, 2]); count()"
    nb_runner.create_notebook([HEAD.format(log=log) + HELPERS, "tmp = 1", plot, check])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == plain
    nb_runner.run_all()
    assert nb_runner.peek("r") == plain
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert nb_runner.peek("r") == plain, "the served plot left the name as it was"
