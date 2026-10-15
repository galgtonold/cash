"""With ``%autoreload`` on, cash reloads an edited module the extension's way:
objects made before the edit run the edited methods, as in a plain kernel.

cash's own module check reloads a tracked module before the extension's
check runs, which then takes it as current. A plain ``importlib.reload``
made a new class, so an object made before the edit kept the old one: its
methods ran the old code and ``type(orders[0]) is mylib.Order`` was False,
where a plain kernel with ``%autoreload 2`` updates the old class in place.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
from IPython.extensions.autoreload import ModuleReloader

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SHOP = (
    "import time\n"
    "class Order:\n"
    "    def __init__(self, v):\n"
    "        self.v = v\n"
    "    def total(self):\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S / 3})\n"
    "        return self.v * {factor}\n"
)


@pytest.fixture
def shop(tmp_path, monkeypatch):
    name = f"_ar_shop_{os.getpid()}_{id(tmp_path)}"
    path = tmp_path / f"{name}.py"
    path.write_text(SHOP.format(factor=2), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    sys.modules.pop(name, None)


def _edit(path, factor):
    before = os.stat(path).st_mtime_ns
    path.write_text(SHOP.format(factor=factor), encoding="utf-8")
    # A reload is decided by the file's mtime, which must move.
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


def _reloader(name, check_all=True):
    reloader = ModuleReloader()
    reloader.enabled = True
    reloader.check_all = check_all
    if not check_all:
        reloader.modules[name] = True
    reloader.check()  # records the module's mtime, as the extension does once it is imported
    return reloader


def _with_autoreload(mock_shell, reloader):
    mock_shell.magics_manager.registry = {"AutoreloadMagics": types.SimpleNamespace(_reloader=reloader)}


@pytest.mark.parametrize("made", ["in_a_cell", "from_a_pickle"])
def test_objects_made_before_the_edit_run_the_edited_method(cash_magics, mock_shell, shop, tmp_path, made):
    name, path = shop
    pkl = tmp_path / "orders.pkl"
    if made == "in_a_cell":
        build = [f"orders = [{name}.Order(i) for i in range(3)]"]
    else:
        build = [
            f"with open({str(pkl)!r}, 'wb') as f:\n    pickle.dump([{name}.Order(i) for i in range(3)], f)",
            f"with open({str(pkl)!r}, 'rb') as f:\n    orders = pickle.load(f)",
        ]
    cells = [f"import pickle, {name}", *build, "t = [o.total() for o in orders]"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["t"] == [0, 2, 4]
    _with_autoreload(mock_shell, _reloader(name))

    _edit(path, 3)
    # Without the notebook's cells (a headless run, a console), as with a
    # pickle: nothing rebuilds ``orders`` from the edited class first.
    run_cash_cell(cash_magics, cells[-1])

    module = sys.modules[name]
    orders = mock_shell.user_ns["orders"]
    assert type(orders[0]) is module.Order
    assert orders[1].total() == 3
    assert mock_shell.user_ns["t"] == [0, 3, 6]


def test_without_the_extension_cash_reloads_as_before(cash_magics, mock_shell, shop):
    """Control: with no ``%autoreload``, cash's reload makes a new class and
    leaves the objects made before it with the old one, as
    ``importlib.reload`` does."""
    name, path = shop
    cells = [f"import {name}", f"orders = [{name}.Order(i) for i in range(3)]"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    _edit(path, 3)
    run_cash_cell(cash_magics, "x = 1", cells=[*cells, "x = 1"])

    assert type(mock_shell.user_ns["orders"][0]) is not sys.modules[name].Order


def test_the_reload_is_the_extension_s_only_where_it_reloads(shop):
    # Imported here, so the tests above collect against a cash without it.
    from cash.notebook.ipython.autoreload_hook import in_place_reload

    name, _path = shop
    module = __import__(name)
    assert in_place_reload(types.SimpleNamespace()) is None
    disabled = ModuleReloader()
    disabled.enabled = False
    shell = types.SimpleNamespace(
        magics_manager=types.SimpleNamespace(registry={"AutoreloadMagics": types.SimpleNamespace(_reloader=disabled)})
    )
    assert in_place_reload(shell) is None

    # ``%autoreload 1`` reloads only what ``%aimport`` marked.
    reloader = _reloader("some_other_module", check_all=False)
    shell.magics_manager.registry["AutoreloadMagics"]._reloader = reloader
    calls = []
    assert in_place_reload(shell)(module, calls.append) is False
    assert calls == []
    reloader.modules[name] = True
    assert in_place_reload(shell)(module, lambda m: calls.append(m) or m) is True
    assert calls == [module]
