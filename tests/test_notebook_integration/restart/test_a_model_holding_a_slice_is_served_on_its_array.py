"""A statement whose output holds a slice of an array, the slice variable
itself, is served, and the slice it holds sits on the array as in plain
Python.

``X_train = X[:4]`` then ``model.fit(X_train)``, where the model keeps
``X_train`` (a k-nearest-neighbours model keeps the very array it was fitted
on): the fit was refused as a live alias and ran on every Run All and after
every restart. The entry now keeps ``X_train`` as a reference to the
variable, checked by its bytes, and a hit binds the live ``X_train``, which
the slice cell made again on the live ``X``. A write to ``X`` shows through
the model, and a write through the model shows in ``X``, as in a restart
and Run All in plain Python, and a reader of ``X`` below sees both.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached, shows_executed

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
CELLS = [
    SETUP,
    "import time\nimport numpy as np",
    "class Keeper:\n"
    "    def fit(self, X):\n"
    "        time.sleep(0.3)\n"
    "        self.X_ = X\n"
    "        self.total_ = float(X.sum())\n"
    "        return self",
    "X = np.arange(24.0).reshape(6, 4)",
    "X_train = X[:4]",
    "model = Keeper().fit(X_train)",
    "X[0, 0] = 99.0\nprint('BASE', model.X_[0, 0], X_train[0, 0])",
    "model.X_[1, 1] = -7.0\nprint('HELD', X[1, 1], X_train[1, 1])",
    "print('SAME', model.X_ is X_train, bool(np.shares_memory(model.X_, X)), model.total_)",
    "def slow_sum(a):\n    time.sleep(0.3)\n    return float(a.sum())\ntotal = slow_sum(X)\nprint('TOTAL', total)",
]
FIT, BASE, HELD, SAME, TOTAL = 6, 7, 8, 9, 10


def _plain(cells: list[str]) -> list[str]:
    """The lines a restart and Run All print in plain Python, cell by cell."""
    import contextlib
    import io

    ns: dict = {"__name__": "__plain__"}
    lines = []
    for cell in cells[1:]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(cell, ns)
        lines.append(out.getvalue().strip())
    return lines


def _check(nb_runner, plain: list[str], when: str) -> None:
    for cell in (BASE, HELD, SAME, TOTAL):
        expected = plain[cell - 2]
        assert expected, cell
        assert expected in nb_runner.get_output(cell), (when, cell, expected, nb_runner.get_output(cell))


def test_the_fit_is_served_and_its_slice_sits_on_the_array(nb_runner):
    plain = _plain(CELLS)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    _check(nb_runner, plain, "first Run All")

    nb_runner.run_all()
    _check(nb_runner, plain, "second Run All")
    assert shows_cached(nb_runner.get_output(FIT)), nb_runner.get_output(FIT)

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner.run_all()
    _check(nb_runner, plain, "after a restart")
    assert shows_cached(nb_runner.get_output(FIT)), nb_runner.get_output(FIT)


def test_an_edited_write_through_the_model_reaches_the_reader_of_the_array(nb_runner):
    """The write through the model moves ``X`` on: edited, the reader of
    ``X`` below runs again and sees the new value, in the kernel that ran
    the cells and after a restart."""
    cells = list(CELLS)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()

    cells[HELD - 1] = cells[HELD - 1].replace("-7.0", "-8.0")
    nb_runner.set_cell_source(HELD, cells[HELD - 1])
    plain = _plain(cells)
    nb_runner.run_all()
    _check(nb_runner, plain, "after the edit")
    assert shows_executed(nb_runner.get_output(TOTAL)), nb_runner.get_output(TOTAL)

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner.run_all()
    _check(nb_runner, plain, "after a restart")
    assert shows_cached(nb_runner.get_output(FIT)), nb_runner.get_output(FIT)
