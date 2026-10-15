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


def _record_sources(reloader):
    """What the deduplicating reload records once a module is imported.
    IPython 8 (the last on Python 3.10) has no such reload: its
    ``%autoreload 2`` reloads the module whole, which the tests below
    expect of it all the same."""
    deduper = getattr(reloader, "deduper_reloader", None)
    if deduper is not None:
        assert deduper.enabled, "the extension's default these tests are about"
        deduper.update_sources()
    return deduper


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


# -- the module is reloaded whole --------------------------------------------
#
# IPython's deduplicating reload (``%autoreload 2``'s default) compiles each
# changed function from ``ast.unparse`` of its definition alone, with the
# module's file name: the code says it starts on line 1, so its source read as
# the module's docstring, and a function the edit added got a copy of the
# module's namespace for its globals. Keys built from that code differed from
# a fresh process's (an image notebook's 16 s preprocess missed after every
# restart), and a call that raised took its argument for changed, since the
# helper's source read as none (an Elo notebook's ``played`` lost its lineage
# and every run_elo after the edit ran uncached).

EDITED = (
    '"""The module."""\n'
    "import time\n\n\n"
    "def get(x):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return helper(x) + 1\n\n\n"
    "def helper(x):\n"
    "    return x * 2\n"
)


def test_a_module_the_extension_reloads_is_loaded_as_a_fresh_import_loads_it(lib):
    import inspect

    name, path = lib
    module = __import__(name)
    reloader = _reloader(name)
    deduper = _record_sources(reloader)
    shell = _autoreload_shell(reloader)

    before = os.stat(path).st_mtime_ns
    path.write_text(EDITED, encoding="utf-8")
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))

    assert run_autoreload_now(shell) == {name}
    assert module.get(10) == 21
    assert inspect.getsource(module.get).startswith("def get(x):")
    assert module.get.__code__.co_firstlineno == 5
    assert module.helper.__globals__ is module.__dict__
    assert deduper is None or deduper.enabled, "the extension's setting is as it was"


def test_a_module_cash_reloaded_is_left_to_cash(lib):
    from cash.notebook.ipython.autoreload_hook import run_autoreload_now as run

    name, path = lib
    module = __import__(name)
    reloader = _reloader(name)
    shell = _autoreload_shell(reloader)
    _edit(path, 1)
    import importlib

    importlib.reload(module)  # what cash's own check does first
    reloaded_get = module.get

    assert run(shell, {name}) == set()
    assert module.get is reloaded_get
    assert module.get(10) == 11
    # Control: the extension still reloads the next edit.
    _edit(path, 2)
    assert run(shell) == {name}
    assert module.get(10) == 12


ELO = (
    "import time\n\n\n"
    "def run_elo(matches, k=20.0):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return float(matches.sum()) * k{tail}\n"
)


def test_an_argument_to_a_call_that_raised_after_an_edit_keeps_its_lineage(
    cash_magics, statement_processor, mock_shell, tmp_path, monkeypatch
):
    name = f"_elo_{os.getpid()}_{id(tmp_path)}"
    path = tmp_path / f"{name}.py"
    path.write_text(ELO.format(tail=""), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        cells = [f"import numpy as np\nimport {name} as elo", "played = np.arange(4.0)", "hist = elo.run_elo(played)"]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        reloader = _reloader(name)
        _record_sources(reloader)
        mock_shell.magics_manager.registry = {"AutoreloadMagics": types.SimpleNamespace(_reloader=reloader)}
        state = cash_magics.tracking_state
        played = state.variable_lineage["played"]

        before = os.stat(path).st_mtime_ns
        path.write_text(ELO.format(tail=" + goal_diff"), encoding="utf-8")
        os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))
        with pytest.raises(NameError):
            run_cash_cell(cash_magics, cells[2], cells=cells)

        assert state.variable_lineage.get("played") == played, "run_elo cannot change its argument"
        path.write_text(ELO.format(tail=" + 1.0"), encoding="utf-8")
        os.utime(path, ns=(before + 4_000_000_000, before + 4_000_000_000))
        run_cash_cell(cash_magics, cells[2], cells=cells)
        assert mock_shell.user_ns["hist"] == 121.0
        assert "hist" in state.variable_lineage, "the fixed call was not cached"
    finally:
        sys.modules.pop(name, None)
