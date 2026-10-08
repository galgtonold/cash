"""An edit ``%autoreload`` picks up reaches the cells that read the module.

``%load_ext autoreload`` / ``%autoreload 2`` / ``import mylib`` in one cell,
the usual first cell: IPython runs it on its own (it loads an extension), so
cash never saw the import and never tracked ``mylib``. The extension
reloaded the edited file from a ``pre_run_cell`` event, which fires after
the statements cash runs, so the reader below ran the old code and was keyed
as if nothing changed. The import in such a cell is now tracked, and the
extension's check runs before the cell, as in a plain kernel; a module it
reloads is taken in like an edit cash saw itself.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
from IPython.extensions.autoreload import ModuleReloader

from cash.tracking.function_tracker import FunctionTracker
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = f"import time\ndef get(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x + {{n}}\n"


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_autoreload_lib_{os.getpid()}_{id(tmp_path)}"
    path = tmp_path / f"{name}.py"
    path.write_text(LIB.format(n=0), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    sys.modules.pop(name, None)


def _edit(path, n):
    before = os.stat(path).st_mtime_ns
    path.write_text(LIB.format(n=n), encoding="utf-8")
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


def _autoreload_shell(reloader):
    """What run_autoreload_now reads of a shell with the extension loaded."""
    magics = types.SimpleNamespace(_reloader=reloader)
    return types.SimpleNamespace(magics_manager=types.SimpleNamespace(registry={"AutoreloadMagics": magics}))


def run_autoreload_now(shell):
    # Imported here, so the tests below collect against a cash without it.
    from cash.notebook.ipython.autoreload_hook import run_autoreload_now as run

    return run(shell)


def _reloader(name, check_all=False):
    reloader = ModuleReloader()
    reloader.enabled = True
    reloader.check_all = check_all
    if not check_all:
        reloader.modules[name] = True
    reloader.check()  # records the module's mtime, as the extension does once it is imported
    return reloader


def test_an_import_beside_magics_is_tracked(lib):
    name, _path = lib
    __import__(name)
    tracker = FunctionTracker()

    tracked = tracker.auto_track_local_imports(f"%load_ext autoreload\n%autoreload 2\nimport {name}", {})

    assert tracked == {name}


@pytest.mark.parametrize("check_all", [False, True], ids=["autoreload_1", "autoreload_2"])
def test_the_extension_s_reloads_are_reported(lib, check_all):
    name, path = lib
    module = __import__(name)
    reloader = _reloader(name, check_all)
    marked = dict(reloader.modules)
    shell = _autoreload_shell(reloader)
    assert run_autoreload_now(shell) == set()

    _edit(path, 1)

    assert run_autoreload_now(shell) == {name}
    assert module.get(10) == 11
    # The extension's own settings are as they were.
    assert (reloader.check_all, reloader.modules) == (check_all, marked)


def test_without_the_extension_nothing_runs():
    assert run_autoreload_now(types.SimpleNamespace()) == set()
    disabled = ModuleReloader()
    disabled.enabled = False
    assert run_autoreload_now(_autoreload_shell(disabled)) == set()


def test_a_reader_of_a_module_only_the_extension_reloads_runs_the_edit(
    cash_magics, statement_processor, mock_shell, lib
):
    """``%aimport mylib``: the module was never imported by a cell cash
    ran, so only the extension reloads it."""
    name, path = lib
    mock_shell.user_ns[name] = __import__(name)
    mock_shell.magics_manager.registry = {"AutoreloadMagics": types.SimpleNamespace(_reloader=_reloader(name))}
    cells = [f"x = {name}.get(10)"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    assert mock_shell.user_ns["x"] == 10

    _edit(path, 1)
    run_cash_cell(cash_magics, cells[0], cells=cells)

    assert mock_shell.user_ns["x"] == 11
    assert name in statement_processor.function_tracker.tracked_modules
