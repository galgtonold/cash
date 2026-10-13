"""A statement calling a notebook function keys the same after a restart,
when the function reads a global edited since its ``def`` last ran.

``def make()`` reads ``cols``; ``cols`` is edited and run, the ``def`` is
not. The call's key took the function's lineage, which folds ``cols`` as it
was when the ``def`` ran; after a restart the ``def`` ran with the new
``cols``, so the call missed with its code and data unchanged. It keys
the function by its code now, and ``cols`` as it is at the call.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from tests._cell_driver import run_cash_cell

COLS = "cols = ['a', 'b']"
EDITED = "cols = ['a', 'b', 'c']"
DEF = "def make():\n    return [c * 2 for c in cols]"
CALL = "r = make()"


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: what a later kernel reads is what a file tier keeps."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics, names) -> None:
    """What a kernel restart leaves: no session state, none of *names*; the
    cache itself stays."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in names:
        magics.shell.user_ns.pop(name, None)


def _statuses(magics) -> dict[str, str]:
    return {m.get("code"): str(m.get("status")) for m in magics.cash_status("dict")["last_cell"]["statements"]}


def test_the_call_is_restored_after_a_restart_in_a_run_without_the_notebook(cash_magics):
    cash_magics.cash_persist("on")
    for cell in [COLS, DEF, CALL, EDITED, CALL]:
        run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["r"] == ["aa", "bb", "cc"], "control: the call sees the edit"
    assert "COMPUTED" in _statuses(cash_magics)[CALL], "control: the edit re-keyed the call"

    _restart(cash_magics, ["cols", "make", "r"])
    for cell in [EDITED, DEF, CALL]:
        run_cash_cell(cash_magics, cell)

    assert cash_magics.shell.user_ns["r"] == ["aa", "bb", "cc"]
    assert "RESTORED" in _statuses(cash_magics)[CALL], _statuses(cash_magics)


def test_a_global_changed_after_the_restart_still_re_keys_the_call(cash_magics):
    cash_magics.cash_persist("on")
    for cell in [COLS, DEF, CALL]:
        run_cash_cell(cash_magics, cell)
    _restart(cash_magics, ["cols", "make", "r"])
    for cell in [COLS, DEF, CALL]:
        run_cash_cell(cash_magics, cell)
    assert "RESTORED" in _statuses(cash_magics)[CALL], "control: unchanged, it restores"

    run_cash_cell(cash_magics, EDITED)
    run_cash_cell(cash_magics, CALL)

    assert cash_magics.shell.user_ns["r"] == ["aa", "bb", "cc"]
    assert "COMPUTED" in _statuses(cash_magics)[CALL], _statuses(cash_magics)


def test_the_call_is_restored_after_a_restart_in_the_notebook(cash_magics):
    """With the notebook's cells the simulation keys the call before the
    ``def`` has run again; it reaches the key the run will."""
    cash_magics.cash_persist("on")
    cells = [COLS, DEF, CALL]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[0] = EDITED
    run_cash_cell(cash_magics, cells[0], cells=cells)
    run_cash_cell(cash_magics, CALL, cells=cells)
    assert cash_magics.shell.user_ns["r"] == ["aa", "bb", "cc"], "control: the call sees the edit"

    _restart(cash_magics, ["cols", "make", "r"])
    run_cash_cell(cash_magics, CALL, cells=cells)

    assert cash_magics.shell.user_ns["r"] == ["aa", "bb", "cc"]
    assert "RESTORED" in _statuses(cash_magics)[CALL], _statuses(cash_magics)
