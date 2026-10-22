"""A function of the user's module held in a value is keyed by its module.

``big = [mylib.helper] + rows``, then ``t = big[0](3)``: the statement names
neither ``mylib`` nor ``helper``. A class of the user's module held this way
was keyed by its module's code and the module data its methods read; a
function of the module was not. ``mylib.K = 14`` in a cell then served the
old result, at every size, whenever the run-time trace was off (Python
3.10/3.11, ``CASH_TRACE=0``). The function is keyed as such a class is.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import held_functions, shadow
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SHOP = (
    "import os, time\n"
    "K = 2\n"
    "def _log(log):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "def helper(v, log):\n"
    "    _log(log)\n"
    "    return v * K\n"
)


@pytest.fixture(params=["0", "1"], ids=["trace_off", "trace_on"])
def shop(request, tmp_path, monkeypatch):
    name = f"_modfn_shop_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(SHOP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CASH_TRACE", request.param)
    if request.param == "0":
        monkeypatch.setattr(shadow, "SHADOW", None)
    users = held_functions._users_module_classes
    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [c for c in users() if c.__module__ == name])
    yield name, tmp_path
    sys.modules.pop(name, None)


@pytest.mark.parametrize("size", [10, 30_000])
def test_a_setting_of_its_module_reruns_the_call(cash_magics, mock_shell, shop, size):
    name, tmp_path = shop
    log = str(tmp_path / "runs.log")
    cells = [
        f"import types, {name}\nbig = [{name}.helper] + [types.SimpleNamespace(i=i) for i in range({size})]",
        f"t = big[0](3, {log!r})",
        f"{name}.K = 14",
    ]
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["t"] == 6
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert open(log, encoding="utf-8").read() == "x", "control: served when nothing changed"
    run_cash_cell(cash_magics, cells[2], cells=cells)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert mock_shell.user_ns["t"] == 42
