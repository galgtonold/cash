"""An entry stored with a variable that holds its objects too restores when
that variable does not exist yet.

``hgb`` holds the column list ``cols``, and so does ``lr``: ``hgb``'s entry
stores both with it (``StatementCacheMetadata.holders``), so that a restore
gives back one list. After a restart the cells are run again up to ``hgb``,
but not ``lr``'s. Nothing in the kernel can share an object with a variable
that is not there, so the entry restores, with ``cols`` held by ``hgb`` again
(p11's 16.6 s fitted pipeline). Running ``lr``'s cell afterwards gives the
three of them one list, as a plain kernel does.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

CELLS = [
    "import time",
    "cols = ['a', 'b']",
    "lr = {'cols': cols, 'w': (time.sleep(0.2), 1)[1]}",
    "hgb = {'cols': cols, 'w': (time.sleep(0.2), 2)[1]}",
    "n = hgb['w'] + len(hgb['cols'])",
]
NAMES = ["time", "cols", "lr", "hgb", "n"]


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _statuses(magics) -> dict[str, str]:
    return {
        m.get("code"): m.get("status") for m in magics.cash_status("dict")["last_cell"]["statements"] if m.get("code")
    }


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: what a later kernel reads is what the disk tier kept."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _first_session(magics) -> None:
    for cell in CELLS[:4]:
        run_cash_cell(magics, cell, cells=CELLS)
    ns = magics.shell.user_ns
    assert ns["hgb"]["cols"] is ns["cols"] is ns["lr"]["cols"], "control: one list"


def _run(magics, *indices: int) -> dict[str, str]:
    """Run the cells at *indices*; the status of each statement of the last."""
    for i in indices:
        run_cash_cell(magics, CELLS[i], cells=CELLS)
    return _statuses(magics)


def test_a_holder_not_run_after_a_restart_does_not_refuse_the_entry(cash_magics, clean_backend):
    _first_session(cash_magics)
    _restart(cash_magics)

    statuses = _run(cash_magics, 0, 1, 3)

    assert statuses[CELLS[3]] == CacheStatus.RESTORED, statuses
    ns = cash_magics.shell.user_ns
    assert "lr" not in ns, "a restore does not bring back a variable no cell made"
    assert ns["hgb"] == {"cols": ["a", "b"], "w": 2}
    assert ns["hgb"]["cols"] is ns["cols"], "the restored entry holds the live list, as a run does"


def test_the_holder_s_cell_run_afterwards_shares_the_one_list(cash_magics, clean_backend):
    """``lr``'s cell run after the restored ``hgb``: a plain kernel builds
    ``lr`` around the live ``cols``, which ``hgb`` holds too."""
    _first_session(cash_magics)
    _restart(cash_magics)
    _run(cash_magics, 0, 1, 3, 2)

    ns = cash_magics.shell.user_ns
    assert ns["lr"]["cols"] is ns["cols"]
    assert ns["hgb"]["cols"] is ns["cols"]
    ns["cols"].append("c")
    assert ns["lr"]["cols"] == ns["hgb"]["cols"] == ["a", "b", "c"]


def test_a_holder_run_again_is_given_the_restored_list(cash_magics, clean_backend):
    """Control for the rule: a holder that is there is checked and restored
    with the outputs, so the three names hold one list."""
    _first_session(cash_magics)
    _restart(cash_magics)

    statuses = _run(cash_magics, 0, 1, 2, 3)

    assert statuses[CELLS[3]] == CacheStatus.RESTORED, statuses
    ns = cash_magics.shell.user_ns
    assert ns["hgb"]["cols"] is ns["cols"] is ns["lr"]["cols"]


def test_a_holder_that_is_there_but_changed_keeps_its_own_value(cash_magics, clean_backend):
    """``lr`` bound since the restart to something else holding ``cols``.
    ``hgb``'s statement leaves ``cols`` as it was, so the entry stores it as
    the variable (an anchor), not ``lr`` with it: the restore binds the live
    list and leaves ``lr`` alone, as a plain run does."""
    _first_session(cash_magics)
    _restart(cash_magics)
    _run(cash_magics, 0, 1)
    run_cash_cell(cash_magics, "lr = {'cols': cols, 'w': 9}", cells=CELLS)

    statuses = _run(cash_magics, 3)

    assert statuses[CELLS[3]] == CacheStatus.RESTORED, statuses
    ns = cash_magics.shell.user_ns
    assert ns["lr"]["w"] == 9, "the live holder is kept"
    assert ns["hgb"]["cols"] is ns["cols"] is ns["lr"]["cols"]


def test_a_cell_reading_it_alone_brings_it_back_without_the_holder(cash_magics, clean_backend):
    """Only the cell reading ``hgb`` runs after the restart: the upstream
    check restores ``hgb`` from its entry, without ``lr``."""
    _first_session(cash_magics)
    _restart(cash_magics)

    statuses = _run(cash_magics, 4)

    assert statuses[CELLS[3]] == CacheStatus.RESTORED, statuses
    ns = cash_magics.shell.user_ns
    assert ns["n"] == 4
    assert "lr" not in ns
