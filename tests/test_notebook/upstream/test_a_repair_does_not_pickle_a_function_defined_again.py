"""A repair does not pickle a value holding a function a cell below it
defined again.

A plain run pickles the value above the cell defining the function again,
where its name still holds the function the value holds. Re-run below it,
pickle finds the new function under that name and refuses ("it's not the same
object as __main__.ratio"): the cell failed with ``UpstreamStateError``, and a
file writer left a half-written file. A file writer is refused before anything
runs; any other statement is refused when pickle refuses. Either way the cell
runs as written, with ``NOTEBOOK-REPAIR-GAP``.
"""

from __future__ import annotations

import functools
import sys
import types

import pytest

from cash.notebook.upstream.replay import UpstreamRepairGap
from tests._cell_driver import run_cash_cell

DEFS = "import pickle\ndef ratio(x, k):\n    return x * k"
# A tuple: a value nothing changes in place keeps the lineage it was made
# with, so the repair re-runs the writer on it as it is in memory. One that
# can change (a list, a dict, a partial) may be taken as changed by the dump
# on some Pythons, and the repair then makes it again first, under the
# ``ratio`` defined now, and the dump succeeds: right, but not this path.
MODEL = "model = (ratio, 2)"
AGAIN = "def ratio(x, k):\n    return x * k"


def _cells(writer: str, reader: str) -> list[str]:
    return [DEFS, MODEL, writer, AGAIN, reader]


def _run_all(cash_magics, cells: list[str]) -> None:
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)


def test_a_file_writer_is_not_re_run(cash_magics, tmp_path):
    path = tmp_path / "model.pkl"
    dump = f"with open({str(path)!r}, 'wb') as f:\n    pickle.dump(model, f{{}})"
    reader = f"n = len(open({str(path)!r}, 'rb').read())"
    _run_all(cash_magics, _cells(dump.format(""), reader))
    written = path.read_bytes()

    edited = _cells(dump.format(", protocol=4"), reader)
    with pytest.raises(UpstreamRepairGap, match="which would pickle .model."):
        run_cash_cell(cash_magics, reader, cells=edited)
    assert path.read_bytes() == written


def test_a_pickle_into_a_variable_fails_into_the_gap(cash_magics):
    blob = "blob = pickle.dumps(model{})"
    reader = "n = len(blob)"
    _run_all(cash_magics, _cells(blob.format(""), reader))

    with pytest.raises(UpstreamRepairGap, match="not the same object"):
        run_cash_cell(cash_magics, reader, cells=_cells(blob.format(", protocol=4"), reader))


def test_without_a_definition_below_the_repair_runs(cash_magics):
    """The positive control: the same edit with nothing defined again below
    the writer re-runs it, and the cell reads the new pickle."""
    blob = "blob = pickle.dumps(model{})"
    reader = "n = len(blob)"
    cells = [DEFS, MODEL, blob.format(""), "x = 1", reader]
    _run_all(cash_magics, cells)
    edited = [DEFS, MODEL, blob.format(", protocol=4"), "x = 1", reader]
    run_cash_cell(cash_magics, reader, cells=edited)
    ns = cash_magics.shell.user_ns
    assert ns["blob"] == __import__("pickle").dumps(ns["model"], protocol=4)


@pytest.fixture
def module(monkeypatch):
    """A module of the user's, as pickle and `named_now` find it."""
    mod = types.ModuleType("defined_again_module")
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return mod


def _define(module, code: str) -> None:
    exec(code, module.__dict__)


def _defined_again(value, module):
    from cash.notebook.held_functions import defined_again_held

    return defined_again_held(value, module.__dict__)


def test_a_function_defined_again_is_named(module):
    _define(module, "def f(x):\n    return x")
    held = {"steps": [module.f]}
    assert _defined_again(held, module) is None
    _define(module, "def f(x):\n    return x")
    assert _defined_again(held, module) == "defined_again_module.f"


def test_an_object_of_a_class_defined_again_is_named(module):
    _define(module, "class Step:\n    def run(self):\n        return 1")
    held = [module.Step()]
    assert _defined_again(held, module) is None
    _define(module, "class Step:\n    def run(self):\n        return 1")
    assert _defined_again(held, module) is not None


def test_a_function_its_wrapper_now_holds_is_not_named(module):
    _define(module, "def f(x):\n    return x")
    inner = module.f
    module.f = functools.wraps(inner)(lambda x: inner(x))
    assert _defined_again({"f": module.f}, module) is None
