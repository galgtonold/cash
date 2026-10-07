"""After a restart, a cell run alone sees the state the cells above it set on a module.

``mylib.configure(5)`` in one cell, ``x = mylib.from_k(2)`` below it: after a
kernel restart, running only the reader rebuilt ``mylib`` from its import and
left the setting out, so the cell computed with the file's ``K`` (4, not 10),
silently. ``mylib.K = 5`` was rebuilt, because its text names the module; a
call is not, because with the module not loaded yet the simulation cannot
tell a setter (``configure``, ``set_k`` imported from the module, a notebook
helper that writes into it) from any other module function. Nor can it tell
a setting only running it shows (``globals()[name] = v``, ``global K`` in a
method).

The runtime records what each statement set state on when it runs -- what
the text says, and what it was seen rebinding -- and keeps it for a later
kernel; the simulation takes those names for the statement's outputs, as the
runtime did, so the reader's rebuild runs the setting too.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest

from cash.backends.file_backend import FileBackend
from cash.exceptions import CashWarning
from cash.notebook.cache_key import statement_source_hash
from tests._cell_driver import run_cash_cell

LIB = (
    "K = 2\n"
    "CFG = {'k': 2}\n"
    "def configure(k):\n    global K\n    K = k\n"
    "def set_cfg(k):\n    CFG['k'] = k\n"
    "def put(name, v):\n    globals()[name] = v\n"
    "def from_k(n):\n    return K * n\n"
    "def from_cfg(n):\n    return CFG['k'] * n\n"
    "class Settings:\n    def apply(self, k):\n        global K\n        K = k\n"
)


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend in place of the in-memory one: the record a later
    kernel reads is metadata alone, which only a file tier keeps."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_restart_state_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    (tmp_path / f"{name}_outer.py").write_text(f"import {name}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)
    sys.modules.pop(f"{name}_outer", None)


def _restart(magics, lib) -> None:
    """What a kernel restart leaves: no session state, no variables, the
    module not loaded."""
    magics.tracking_state.reset_session_state()
    simulator = magics._upstream_checker.simulator
    simulator.cache.reset()
    simulator.probe.reset()
    user_ns = magics.shell.user_ns
    for name in [n for n in user_ns if not n.startswith("_") and n not in ("get_ipython", "exit", "quit")]:
        user_ns.pop(name, None)
    sys.modules.pop(lib, None)
    sys.modules.pop(f"{lib}_outer", None)


def _state(lib) -> tuple[int, int]:
    """What the module holds: a reader restored from the cache would show
    the setting's value whatever the module holds."""
    module = sys.modules[lib]
    return module.K, module.CFG["k"]


def _codes(caught) -> list[str | None]:
    return [getattr(w.message, "code", None) for w in caught if isinstance(w.message, CashWarning)]


SETTERS = {
    "a_module_function_writing_a_global": (["import {m}", "{m}.configure(5)"], "x = {m}.from_k(2)"),
    "a_setter_imported_from_the_module": (
        ["import {m}\nfrom {m} import configure", "configure(5)"],
        "x = {m}.from_k(2)",
    ),
    "only_imported_from_the_module": (["from {m} import configure, from_k", "configure(5)"], "x = from_k(2)"),
    "a_reader_imported_from_the_module": (
        ["import {m}\nfrom {m} import from_k", "{m}.configure(5)"],
        "x = from_k(2)",
    ),
    "a_notebook_helper_writing_into_the_module": (
        ["import {m}", "def setup(k):\n    global seen\n    seen = k\n    {m}.K = k", "setup(5)"],
        "x = {m}.from_k(2)",
    ),
    "a_setter_changing_a_dict_of_the_module": (["import {m}", "{m}.set_cfg(5)"], "x = {m}.from_cfg(2)"),
    "a_setter_only_running_it_shows": (["import {m}", "{m}.put('K', 5)"], "x = {m}.from_k(2)"),
    "a_method_setting_a_global": (["import {m}\ns = {m}.Settings()", "s.apply(5)"], "x = {m}.from_k(2)"),
    "a_loop_of_a_method_setting_a_global": (
        ["import {m}\ns = {m}.Settings()", "for k in [5]:\n    s.apply(k)"],
        "x = {m}.from_k(2)",
    ),
}


@pytest.mark.parametrize("case", list(SETTERS), ids=list(SETTERS))
def test_a_reader_run_alone_after_a_restart_sees_the_setting(cash_magics, clean_backend, lib, case):
    above, reader = SETTERS[case]
    cells = [cell.format(m=lib) for cell in [*above, "y = 1", reader]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["x"] == 10, "control: the top-to-bottom run"
    state = _state(lib)

    _restart(cash_magics, lib)
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert _state(lib) == state
    assert cash_magics.shell.user_ns["x"] == 10


def test_the_record_names_what_the_setter_changed(cash_magics, clean_backend, lib):
    """``mylib.put('K', 5)`` says nothing in its text: the record is what
    running it showed, and the names that see the module."""
    from cash.notebook.cache_key import module_state_key

    cells = [f"import {lib}\nfrom {lib} import from_k", f"{lib}.put('K', 5)", "x = from_k(2)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    record = clean_backend.get_metadata(module_state_key(statement_source_hash(cells[1])))
    assert record is not None, "no record kept for a later kernel"
    assert record["modules"] == [lib]
    assert record["names"] == sorted([lib, "from_k"])
    assert clean_backend.get_metadata(module_state_key(statement_source_hash("x = from_k(2)"))) is None


def test_a_module_no_name_sees_is_reported_after_a_restart(cash_magics, clean_backend, lib):
    """Set through another module (``outer.mylib.configure(5)``): no name of
    the notebook sees ``mylib``, so nothing can rebuild what was set on it,
    and cash says so rather than compute on the file's ``K``."""
    outer = f"{lib}_outer"
    cells = [f"import {outer}", f"{outer}.{lib}.configure(5)", f"x = {outer}.{lib}.from_k(2)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["x"] == 10

    _restart(cash_magics, lib)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert "NOTEBOOK-RELOAD-STATE" in _codes(caught)


def test_a_setter_that_ran_in_this_kernel_is_not_reported(cash_magics, clean_backend, lib):
    outer = f"{lib}_outer"
    cells = [f"import {outer}", f"{outer}.{lib}.configure(5)", f"x = {outer}.{lib}.from_k(2)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    _restart(cash_magics, lib)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)

    assert "NOTEBOOK-RELOAD-STATE" not in _codes(caught)
    assert cash_magics.shell.user_ns["x"] == 10
