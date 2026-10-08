"""A setting on a module made without naming the module is rebuilt too.

``mylib.CONFIG["k"] = 5`` in one cell, ``x = mylib.from_k(2)`` below it:
after a kernel restart or an edit of ``mylib.py``, running only the reader
rebuilds ``mylib`` with the setting. The same setting made through a name
holding what the module holds (``from mylib import CONFIG; CONFIG["k"] = 5``,
``cfg = mylib.CONFIG; cfg["k"] = 5``, ``CFG.k = 5``), through a bare
decorator (``@mylib.register``), through an import whose module registers
itself, or through a method of the module's class (a class method doing
``cls.k = k`` whatever its name, a method doing ``type(self).k = k``) was
left out: the reader computed with the file's values, silently.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.callee_reach import module_state_writes
from tests._cell_driver import run_cash_cell

LIB = (
    "CONFIG = {'k': 1}\n"
    "REG = {}\n"
    "class Cfg:\n"
    "    k = 1\n"
    "    @classmethod\n    def tune(cls, k):\n        cls.k = k\n"
    "    def setk(self, k):\n        type(self).k = k\n"
    "    def own(self, k):\n        self.mine = k\n"
    "    def noop(self):\n        return self.k\n"
    "CFG = Cfg()\n"
    "def register(f):\n    REG[f.__name__] = f\n    return f\n"
    "def from_k(n):\n    return CONFIG['k'] * CFG.k * n\n"
    "def names():\n    return sorted(REG)\n"
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
    name = f"_named_state_{os.getpid()}_{id(tmp_path)}"
    path = tmp_path / f"{name}.py"
    path.write_text(LIB, encoding="utf-8")
    (tmp_path / f"{name}_plugin.py").write_text(
        f"import {name}\n@{name}.register\ndef plug():\n    return 1\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    for module in (name, f"{name}_plugin"):
        sys.modules.pop(module, None)


@pytest.fixture
def stores_everything(cash_instance):
    """Statements are stored however cheap, as on a slow machine: the cells
    above are then restored, not run, unless something says they set state."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _restart(magics, name) -> None:
    """What a kernel restart leaves: no session state, no variables, the
    modules not loaded."""
    magics.tracking_state.reset_session_state()
    simulator = magics._upstream_checker.simulator
    simulator.cache.reset()
    simulator.probe.reset()
    user_ns = magics.shell.user_ns
    for var in [n for n in user_ns if not n.startswith("_") and n not in ("get_ipython", "exit", "quit")]:
        user_ns.pop(var, None)
    for module in (name, f"{name}_plugin"):
        sys.modules.pop(module, None)


def _edit(path) -> None:
    before = os.stat(path).st_mtime_ns
    path.write_text(LIB + "def unrelated():\n    return 0\n", encoding="utf-8")
    # A reload is decided by the file's mtime, which must move.
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


READ_K = "x = {m}.from_k(2)"
READ_NAMES = "x = {m}.names()"


def _module_answer(name: str, reader: str):
    """What the module answers now: a reader restored from the cache shows
    the setting's value whatever the module holds."""
    module = sys.modules[name]
    return module.from_k(2) if reader == READ_K else module.names()


CASES = {
    "a_from_imported_dict": (["import {m}\nfrom {m} import CONFIG", "CONFIG['k'] = 5"], READ_K, 10),
    "an_alias_of_a_dict": (["import {m}", "cfg = {m}.CONFIG", "cfg['k'] = 5"], READ_K, 10),
    "a_method_changing_a_from_imported_dict": (
        ["import {m}\nfrom {m} import CONFIG", "CONFIG.update(k=5)"],
        READ_K,
        10,
    ),
    "a_from_imported_objects_attribute": (["import {m}\nfrom {m} import CFG", "CFG.k = 5"], READ_K, 10),
    "a_bare_decorator_on_a_def": (["import {m}", "@{m}.register\ndef alpha():\n    return 1"], READ_NAMES, ["alpha"]),
    "a_bare_decorator_on_a_class": (["import {m}", "@{m}.register\nclass Alpha:\n    pass"], READ_NAMES, ["Alpha"]),
    "a_class_method_named_anything": (["import {m}", "{m}.Cfg.tune(5)"], READ_K, 10),
    "a_method_setting_its_class": (["import {m}", "{m}.Cfg().setk(5)"], READ_K, 10),
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_a_reader_run_alone_after_a_restart_sees_the_setting(cash_magics, clean_backend, stores_everything, lib, case):
    name, _ = lib
    above, reader, expected = CASES[case]
    cells = [cell.format(m=name) for cell in [*above, "y = 1", reader]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["x"] == expected, "control: the top-to-bottom run"

    _restart(cash_magics, name)
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert _module_answer(name, reader) == expected
    assert cash_magics.shell.user_ns["x"] == expected


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_a_reader_run_alone_after_an_edit_of_the_module_sees_the_setting(cash_magics, stores_everything, lib, case):
    name, path = lib
    above, reader, expected = CASES[case]
    cells = [cell.format(m=name) for cell in [*above, "y = 1", reader]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["x"] == expected, "control: the top-to-bottom run"

    _edit(path)
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert _module_answer(name, reader) == expected
    assert cash_magics.shell.user_ns["x"] == expected


def test_an_import_whose_module_registers_itself_is_kept_as_a_setting_of_the_module(cash_magics, clean_backend, lib):
    """``import plugin`` where ``plugin.py`` does ``@mylib.register``: the
    import is one of the statements a restart's rebuild of ``mylib`` runs,
    kept for a later kernel as any setting is."""
    from cash.notebook.cache_key import module_state_key, statement_source_hash  # noqa: PLC0415

    name, _ = lib
    cells = [f"import {name}", f"import {name}_plugin", f"x = {name}.names()"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["x"] == ["plug"]

    record = clean_backend.get_metadata(module_state_key(statement_source_hash(cells[1])))
    assert record is not None, "no record kept for a later kernel"
    assert record["modules"] == [name]
    assert name in record["names"]


def _namespace(name: str) -> dict:
    module = __import__(name)
    return {
        "lib": module,
        "CONFIG": module.CONFIG,
        "CFG": module.CFG,
        "made": module.Cfg(),
        "mine": {"k": 1},
        "Cfg": module.Cfg,
    }


@pytest.mark.parametrize(
    "code",
    ["CONFIG['k'] = 5", "CONFIG.update(k=5)", "Cfg.k = 5", "CFG.own(5)", "made.setk(5)", "setattr(CFG, 'k', 5)"],
)
def test_a_store_into_what_the_module_holds_sets_its_state(lib, code):
    """A store into the module's own objects sets its state, wherever the
    name came from: a method storing on an instance the module holds, or on
    the class, too."""
    name, _ = lib
    assert module_state_writes(code, _namespace(name)) == {name}


@pytest.mark.parametrize("code", ["mine['k'] = 5", "made.k = 5", "made.own(5)", "lib.CFG.noop()", "lib.Cfg.noop(made)"])
def test_a_store_into_the_notebooks_own_objects_does_not(lib, code):
    """Controls: the notebook's own objects, an instance it made of the
    module's class included, and methods whose bodies store nothing."""
    name, _ = lib
    assert module_state_writes(code, _namespace(name)) == frozenset()


def test_an_import_sets_state_on_the_module_its_module_registers_into(lib):
    from cash.notebook.callee_reach import import_state_writes  # noqa: PLC0415 - added by the fix

    name, _ = lib
    __import__(f"{name}_plugin")
    namespace = _namespace(name)
    assert import_state_writes(f"import {name}_plugin", namespace) == {name}
    assert import_state_writes(f"import {name}", namespace) == frozenset()
