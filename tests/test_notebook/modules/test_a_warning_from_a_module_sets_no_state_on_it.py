"""A module function that warns sets no state on its module.

``y = mylib.load(1)`` with ``load`` calling ``warnings.warn``: the first
warning puts ``__warningregistry__`` in ``mylib``'s globals. That was taken
for the statement setting state on ``mylib``, so the statement was not
stored on its first run and missed ("input changed: mylib") on the second: a
slow loader ran twice before it was ever served. The globals the interpreter
itself adds are no state of the user's.
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings

import pytest

from cash.notebook.call_interception import CallSite
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _lib(counter: str) -> str:
    return (
        "import os, time, warnings\n"
        "def _count():\n"
        f"    fd = os.open({counter!r}, os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "def load(x):\n"
        "    _count()\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    warnings.warn('column dropped')\n"
        "    return x\n"
        "def load_quiet(x):\n"
        "    _count()\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return x\n"
        "K = 1\n"
        "def put(name, v):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    globals()[name] = v\n"
        "    return v\n"
    )


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_warning_lib_{os.getpid()}_{id(tmp_path)}"
    counter = tmp_path / "runs"
    (tmp_path / f"{name}.py").write_text(_lib(str(counter)), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, counter
    sys.modules.pop(name, None)


def _runs(counter) -> int:
    return len(counter.read_bytes()) if counter.exists() else 0


@pytest.mark.parametrize("fn", ["load", "load_quiet"])
@pytest.mark.parametrize("statement", ["y = {m}.{fn}(1)", "y = {m}.{fn}(1) + 0"], ids=["bare_call", "call_in_an_expression"])
def test_a_warning_call_is_served_on_the_second_run(cash_magics, lib, fn, statement):
    name, counter = lib
    cells = [f"import {name}", statement.format(m=name, fn=fn)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)

    assert cash_magics.shell.user_ns["y"] == 1
    assert _runs(counter) == 1
    assert name not in cash_magics.tracking_state.module_state_writers


def test_a_call_that_does_rebind_a_module_global_still_runs_again(cash_magics, lib):
    """Positive control: a real rebinding through ``globals()`` is still
    seen and the statement still runs on every run."""
    name, _counter = lib
    cells = [f"import {name}", f"{name}.K = 1", f"r = {name}.put('K', 5)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert sys.modules[name].K == 1

    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert sys.modules[name].K == 5


@pytest.mark.parametrize("fn", ["load", "load_quiet"])
def test_a_warning_call_unit_is_served_on_the_second_call(call_unit_harness, lib, fn):
    """The same for a call cached on its own (a call unit): its site was
    refused as rebinding a global of its module, so it ran every time."""
    name, counter = lib
    module = importlib.import_module(name)
    func = getattr(module, fn)
    unit = call_unit_harness(lineage={name: "hash-lib"}, user_ns={name: module})
    site = CallSite(source=f"{name}.{fn}(1)", free_names=frozenset({name}), occurrence_index=0)
    wrapped = unit.wrap(func, site)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert wrapped(1) == 1
        assert wrapped(1) == 1

    assert _runs(counter) == 1


def test_a_call_unit_rebinding_a_module_global_still_runs_every_time(call_unit_harness, lib):
    """Positive control: ``globals()[name] = v`` still refuses the site."""
    name, _counter = lib
    module = importlib.import_module(name)
    unit = call_unit_harness(lineage={name: "hash-lib"}, user_ns={name: module})
    site = CallSite(source=f"{name}.put('K', 5)", free_names=frozenset({name}), occurrence_index=0)
    wrapped = unit.wrap(module.put, site)

    wrapped("K", 5)
    module.K = 1
    wrapped("K", 5)

    assert module.K == 5
