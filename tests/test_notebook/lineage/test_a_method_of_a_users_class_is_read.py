"""A statement calling a method of the user's class is keyed on what the method reads.

``model = mylib.Model()`` and ``x = model.predict(2)``, ``predict`` returning
``SCALE * n``: the walk from the statement's names did not follow an
instance at all, and a class only to its ``__init__``, so neither statement
was keyed on ``mylib.SCALE`` or on an environment variable ``predict`` read.
Editing ``mylib.SCALE = 5`` to ``7`` served the old prediction, even after
Restart & Run All. A class defined in a cell is followed the same way.
"""

import os
import sys

import pytest

from cash.notebook.callee_reach import reached_user_code
from cash.notebook.lineage_formula import statement_environment_reads
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = (
    "import os, time\nSCALE = 1\n"
    "class Base:\n    def mode(self):\n        return os.environ.get('CASH_UT_METHOD_MODE', '-')\n"
    "class Model(Base):\n"
    f"    def predict(self, n):\n        time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n        return SCALE * n\n"
    "    @property\n    def scale(self):\n        return SCALE\n"
)


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_methodlib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("CASH_UT_METHOD_MODE", raising=False)
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize("code", ["x = model.predict(2)", "x = model.scale", "model = lib.Model()"])
def test_the_module_data_a_method_reads_is_reached(lib, code):
    module = __import__(lib)
    reach = reached_user_code(code, {"lib": module, "model": module.Model()})
    assert f"{lib}.SCALE" in dict(reach.data)


def test_an_inherited_method_s_environment_read_is_reached(lib):
    module = __import__(lib)
    reads = statement_environment_reads("m = model.mode()", {"model": module.Model()})
    assert ("env", "CASH_UT_METHOD_MODE") in reads


def test_a_cell_class_s_method_is_reached(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "import os\nclass Probe:\n    def mode(self):\n        return os.environ.get('X_PROBE')")
    run_cash_cell(cash_magics, "probe = Probe()")
    assert ("env", "X_PROBE") in statement_environment_reads("m = probe.mode()", mock_shell.user_ns)


def test_editing_the_setting_and_running_again_recomputes(cash_magics, mock_shell, lib):
    cells = [f"import {lib}", f"{lib}.SCALE = 5", f"model = {lib}.Model()", "x = model.predict(2)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["x"] == 10

    cells[1] = f"{lib}.SCALE = 7"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["x"] == 14
