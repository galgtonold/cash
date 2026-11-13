"""A variable rebuilt before a cell leaves live what it only holds.

``full = Pipe([pre, Est()])`` holds ``pre``, as do ``lin`` and ``tree`` fitted
above it, and ``grid = Grid(full)`` holds ``full``. When ``full`` is rebuilt
(a cell below bound it again, and a cell reading it runs), ``grid`` must be
rebuilt too: it holds the old ``full``. ``pre``, ``lin`` and ``tree`` need
not: made and last changed above ``full``, they hold no object its rebuild
makes, and ``full`` made again holds the live ``pre``. They were rebuilt
anyway, re-running every fit on ``pre`` (in a real notebook, 50 s before a
cell that printed parameter names).

A variable the rebuild changes is still rebuilt: when ``full.fit(d)`` fits
``pre`` through ``full``, re-running it on the live ``pre`` would fit it once
more than a Run All does.
"""

from __future__ import annotations

import sys

import pytest

from tests._cell_driver import run_cash_cell

LIBRARY = """
import copy, os


def note(what):
    fd = os.open(os.environ["FAKELEARN_LOG"], os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, (what + "\\n").encode())
    os.close(fd)


class Pre:
    def __init__(self):
        self.fits = 0
        self.mean = None

    def fit(self, d):
        note("pre.fit")
        self.fits += 1
        self.mean = sum(d) / len(d)
        return self


class Est:
    def __init__(self, k):
        self.k = k
        self.coef = None

    def fit(self, d):
        note("est.fit %s" % self.k)
        self.coef = self.k * len(d)
        return self


class Pipe:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, d):
        for step in self.steps:
            step.fit(d)
        return self


class Grid:
    def __init__(self, estimator):
        self.estimator = estimator
        self.best = None

    def fit(self, d):
        note("grid.fit")
        self.best = copy.deepcopy(self.estimator).fit(d)
        return self
"""


def _cells(fit_full: bool) -> list[str]:
    return [
        "import fakelearn as fl\nd = [1.0, 2.0, 3.0]",
        "pre = fl.Pre()",
        "lin = fl.Pipe([pre, fl.Est(1)])\nlin.fit(d)",
        "tree = fl.Pipe([pre, fl.Est(2)])\ntree.fit(d)",
        "full = fl.Pipe([pre, fl.Est(4)])" + ("\nfull.fit(d)" if fit_full else ""),
        "grid = fl.Grid(full)\ngrid.fit(d)",
        "print(full.steps[1].k)",
        "full = fl.Pipe([pre, fl.Est(50)])",
    ]


@pytest.fixture
def fakelearn(tmp_path, monkeypatch):
    (tmp_path / "fakelearn.py").write_text(LIBRARY, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fakelearn", raising=False)
    log = tmp_path / "ran.log"
    monkeypatch.setenv("FAKELEARN_LOG", str(log))
    return log


def _ran(log) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def _run_all_plainly(cells: list[str]) -> dict:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


def _state(ns: dict) -> dict:
    return {
        "pre": (ns["pre"].fits, ns["pre"].mean),
        "full_k": ns["full"].steps[1].k,
        "best_coef": ns["grid"].best.steps[1].coef,
        "lin_coef": ns["lin"].steps[1].coef,
        "tree_coef": ns["tree"].steps[1].coef,
        # The sharing a write through one name has to reach through another.
        "grid_holds_full": ns["grid"].estimator is ns["full"],
        "full_holds_pre": ns["full"].steps[0] is ns["pre"],
        "lin_holds_pre": ns["lin"].steps[0] is ns["pre"],
        "tree_holds_pre": ns["tree"].steps[0] is ns["pre"],
    }


def _rebuild_full(cash_magics, fit_full: bool, log) -> list[str]:
    """Run the notebook, then the cell reading ``full`` again after the cell
    below rebound it; what re-ran before it."""
    cells = _cells(fit_full)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    log.write_text("", encoding="utf-8")
    run_cash_cell(cash_magics, cells[6], cells=cells)
    return _ran(log)


def test_what_full_only_holds_stays_live(cash_magics, mock_shell, fakelearn):
    ran = _rebuild_full(cash_magics, False, fakelearn)

    ns = mock_shell.user_ns
    assert _state(ns) == _state(_run_all_plainly(_cells(False)[:7]))
    # The search holding ``full`` searched again; nothing fitted ``lin`` or ``tree``.
    assert ran == ["grid.fit", "pre.fit", "est.fit 4"]


def test_a_shared_object_the_rebuild_changes_is_rebuilt(cash_magics, mock_shell, fakelearn):
    ran = _rebuild_full(cash_magics, True, fakelearn)

    ns = mock_shell.user_ns
    expected = _state(_run_all_plainly(_cells(True)[:7]))
    assert expected["pre"][0] == 3
    assert _state(ns) == expected
    assert "est.fit 1" in ran
    assert "est.fit 2" in ran
