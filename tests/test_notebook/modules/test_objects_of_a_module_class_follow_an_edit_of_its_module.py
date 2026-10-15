"""A statement calling into objects of a local module's class is keyed on that
module's code, however the objects were made.

``orders = pickle.load(f)`` holds ``mylib.Order`` objects, and
``t = [o.total() for o in orders]`` runs ``Order.total`` and the module's
``rate()``. Neither the statement nor the lineage of ``orders`` (the pickle
file's) names ``mylib``, so editing ``rate`` -- reloaded by cash's own
module check -- left the key where it was, and the old result was served.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

from cash.notebook.cache_key import unkeyed_reach
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SHOP = (
    "import time\n"
    "def rate():\n"
    "    return {rate}\n"
    "class Order:\n"
    "    def __init__(self, v):\n"
    "        self.v = v\n"
    "    def total(self):\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S / 3})\n"
    "        return self.v * {factor}\n"
    "    def counted(self, log):\n"
    "        import os\n"
    "        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "        os.write(fd, b'x')\n"
    "        os.close(fd)\n"
    "        return self.total()\n"
)


@pytest.fixture
def shop(tmp_path, monkeypatch):
    name = f"_held_shop_{os.getpid()}_{id(tmp_path)}"
    path = tmp_path / f"{name}.py"
    path.write_text(SHOP.format(rate=2, factor="rate()"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path, tmp_path / "orders.pkl"
    sys.modules.pop(name, None)


def _edit(path, text):
    before = os.stat(path).st_mtime_ns
    path.write_text(text, encoding="utf-8")
    # A reload is decided by the file's mtime, which must move.
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


def _cells(name, pkl):
    return [
        f"import pickle, {name}\nwith open({str(pkl)!r}, 'wb') as f:\n"
        f"    pickle.dump([{name}.Order(i) for i in range(3)], f)",
        f"with open({str(pkl)!r}, 'rb') as f:\n    orders = pickle.load(f)",
        "t = [o.total() for o in orders]",
    ]


def test_an_edit_of_the_module_reaches_a_reader_of_unpickled_objects(cash_magics, mock_shell, shop):
    name, path, pkl = shop
    cells = _cells(name, pkl)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["t"] == [0, 2, 4]

    _edit(path, SHOP.format(rate=5, factor="rate()"))
    run_cash_cell(cash_magics, cells[2], cells=cells)

    # What the objects run now: cash reloaded the module.
    assert sys.modules[name].rate() == 5
    assert mock_shell.user_ns["t"] == [0, 5, 10]


def test_unedited_the_reader_is_served(cash_magics, mock_shell, shop, tmp_path):
    """Control: the module's part of the key is the same each time, so the
    reader is served while nothing changed."""
    name, path, pkl = shop
    log = tmp_path / "runs.log"
    cells = [*_cells(name, pkl), f"u = [o.counted({str(log)!r}) for o in orders]"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert log.read_text(encoding="utf-8") == "xxx"

    run_cash_cell(cash_magics, cells[3], cells=cells)

    assert log.read_text(encoding="utf-8") == "xxx"
    assert mock_shell.user_ns["u"] == [0, 2, 4]


def test_a_module_edited_but_not_reloaded_is_keyed_on_what_it_runs(shop):
    """A module only a pickle loaded is not tracked, so cash does not reload
    it: after an edit it runs other code than its file says. Its part of the
    key is then neither the old one nor what a fresh import of the edited
    file gives, so neither entry is served for it."""
    from cash.notebook.held_modules import held_module_parts

    name, path, _pkl = shop
    module = __import__(name)
    before, unkeyed = held_module_parts([module.Order])
    assert before and not unkeyed

    _edit(path, SHOP.format(rate=5, factor="rate()"))
    stale, unkeyed = held_module_parts([module.Order])
    assert not unkeyed
    importlib.reload(module)
    fresh, _ = held_module_parts([module.Order])

    assert len({before[0], stale[0], fresh[0]}) == 3


def test_a_module_whose_file_is_gone_cannot_be_keyed(shop):
    name, path, _pkl = shop
    module = __import__(name)
    namespace = {"orders": [module.Order(i) for i in range(3)]}
    code = "t = [o.total() for o in orders]"
    assert unkeyed_reach(code, {"orders"}, namespace, {}).unkeyed_modules == set()

    path.unlink()

    assert unkeyed_reach(code, {"orders"}, namespace, {}).unkeyed_modules == {name}
