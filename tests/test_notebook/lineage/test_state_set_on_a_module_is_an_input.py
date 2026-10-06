"""A statement reading a local module's data is keyed on what the data holds now.

``b = mylib.from_k(10)`` with ``from_k`` reading the module's ``K``: a cell
setting ``mylib.K = 7`` -- or ``mylib.CONFIG["k"] = 7``, ``mylib.set_k(7)``,
``setattr(mylib, "K", 7)`` -- was edited and the notebook run again, and the
old answer was served: the key was built from the module's file, which the
cell does not change. The values the statement's module functions read are
now folded, as the decorator folds a cached function's module globals.
"""

import os
import sys

import pytest

from cash.notebook.callee_reach import reached_user_code
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = (
    "import time\n"
    "K = 2\n"
    "CONFIG = {'k': 2}\n"
    "class Settings:\n    scale = 2\n"
    "def set_k(v):\n    global K\n    K = v\n"
    "def _k():\n    return K\n"
    f"def from_k(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x * _k()\n"
    f"def from_config(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x * CONFIG['k']\n"
    f"def from_settings(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x * Settings.scale\n"
)


@pytest.fixture
def state_module(tmp_path, monkeypatch):
    name = f"_statelib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("setter", "reader"),
    [
        ("{m}.K = {v}", "b = {m}.from_k(10)"),
        ("{m}.CONFIG['k'] = {v}", "b = {m}.from_config(10)"),
        ("{m}.CONFIG.update(k={v})", "b = {m}.from_config(10)"),
        ("{m}.set_k({v})", "b = {m}.from_k(10)"),
        ("setattr({m}, 'K', {v})", "b = {m}.from_k(10)"),
        ("{m}.Settings.scale = {v}", "b = {m}.from_settings(10)"),
        ("{m}.K = {v}", "b = {m}.K * 10 + ({m}.time.sleep(0.2) or 0)"),
        ("{m}.K = {v}", "def f(x):\n    {m}.time.sleep(0.2)\n    return x * {m}.K\nb = f(10)"),
    ],
    ids=[
        "attribute",
        "item",
        "method",
        "module_function",
        "setattr",
        "class_attribute",
        "direct_read",
        "notebook_function",
    ],
)
def test_editing_the_setting_and_running_again_recomputes(cash_magics, mock_shell, state_module, setter, reader):
    cells = [f"import {state_module}", setter.format(m=state_module, v=3), reader.format(m=state_module)]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["b"] == 30

    cells[1] = setter.format(m=state_module, v=7)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["b"] == 70


def test_a_function_reads_its_modules_data_through_its_helpers(state_module):
    module = __import__(state_module)
    labels = [label for label, _ in reached_user_code("b = lib.from_k(1)", {"lib": module}).data]
    assert labels == [f"{state_module}.K"]
