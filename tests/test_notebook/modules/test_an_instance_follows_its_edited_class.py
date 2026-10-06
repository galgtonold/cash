"""An object built from a class in a local module is rebuilt after the class is edited.

Editing the class reloads the module, and the reload makes a new class
object. The instance a cell built from the old one kept it: its methods
returned the pre-edit results and ``isinstance(m, model.Model)`` turned False,
a state neither a plain kernel nor a top-to-bottom run has. A top-to-bottom
run builds the instance from the edited class, so cash re-runs the statement
that built it before a cell uses it.
"""

import os
import sys

import pytest

from tests._cell_driver import run_cash_cell

MODEL = "class Model:\n    def predict(self, v):\n        return v + {n}\n"


@pytest.fixture
def model_module(tmp_path, monkeypatch):
    name = f"_inst_model_{os.getpid()}_{id(tmp_path)}"
    path = tmp_path / f"{name}.py"
    path.write_text(MODEL.format(n=1), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    sys.modules.pop(name, None)


def _edit(path, text):
    before = os.stat(path).st_mtime_ns
    path.write_text(text, encoding="utf-8")
    # A reload is decided by the file's mtime, which must move.
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


def test_the_instance_is_rebuilt_from_the_edited_class(cash_magics, mock_shell, model_module):
    name, path = model_module
    cells = [f"import {name}", f"m = {name}.Model()", f"p = m.predict(10)\nok = isinstance(m, {name}.Model)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert (mock_shell.user_ns["p"], mock_shell.user_ns["ok"]) == (11, True)

    _edit(path, MODEL.format(n=100))
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert (mock_shell.user_ns["p"], mock_shell.user_ns["ok"]) == (110, True)
