"""A cell changing a local module's data reaches a statement that reads it
only through objects of the module's classes.

``orders = pickle.load(f)`` holds ``mylib.Order`` objects whose ``total``
reads the module's ``K``. ``t = [o.total() for o in orders]`` names neither
``mylib`` nor ``K``, and the module's file is unchanged by ``mylib.K = 14``
in a cell: the key took the module's code (``heldmod:``) and not its data,
and the old ``t`` was served.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SHOP = (
    "import time\n"
    "K = 2\n"
    "TABLE = {'k': 1}\n"
    "def factor():\n"
    "    return K * TABLE['k']\n"
    "class Order:\n"
    "    scale = 1\n"
    "    def __init__(self, v):\n"
    "        self.v = v\n"
    "    def total(self):\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S / 3})\n"
    "        return self.v * factor() * self.scale\n"
    "    def counted(self, log):\n"
    "        import os\n"
    "        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "        os.write(fd, b'x')\n"
    "        os.close(fd)\n"
    "        return self.total()\n"
)


@pytest.fixture
def shop(tmp_path, monkeypatch):
    name = f"_data_shop_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(SHOP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, tmp_path / "orders.pkl"
    sys.modules.pop(name, None)


def _cells(name, pkl, change):
    return [
        f"import pickle, {name}\nwith open({str(pkl)!r}, 'wb') as f:\n"
        f"    pickle.dump([{name}.Order(i) for i in range(3)], f)",
        f"with open({str(pkl)!r}, 'rb') as f:\n    orders = pickle.load(f)",
        "t = [o.total() for o in orders]",
        "s = sum(t)",
        change.format(m=name),
    ]


@pytest.mark.parametrize(
    "change",
    ["{m}.K = 14", "{m}.TABLE['k'] = 7", "{m}.Order.scale = 7"],
    ids=["global", "global_in_place", "class_attribute"],
)
def test_the_reader_and_those_below_follow_the_change(cash_magics, mock_shell, shop, change):
    name, pkl = shop
    cells = _cells(name, pkl, change)
    for cell in cells[:4]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["t"] == [0, 2, 4]

    run_cash_cell(cash_magics, cells[4], cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    run_cash_cell(cash_magics, cells[3], cells=cells)

    assert mock_shell.user_ns["t"] == [0, 14, 28]
    assert mock_shell.user_ns["s"] == 42


def test_unchanged_the_reader_is_served(cash_magics, mock_shell, shop, tmp_path):
    """Control: the module's data is digested the same each time, so the
    reader is served while nothing changed."""
    name, pkl = shop
    log = tmp_path / "runs.log"
    cells = [*_cells(name, pkl, "pass")[:2], f"u = [o.counted({str(log)!r}) for o in orders]"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert log.read_text(encoding="utf-8") == "xxx"

    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert log.read_text(encoding="utf-8") == "xxx"
    assert mock_shell.user_ns["u"] == [0, 2, 4]
