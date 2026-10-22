"""A user module's class held in a big value, with no object of it alive,
is keyed as in a small one.

``big = pickle.load(f)`` holding ``mylib.Order`` (the class, not an object
of it) and 30,000 other objects, then ``t = big[0](3).total()``. A walk of
``big`` stops early when nothing it could find exists (no notebook function,
no object of a class of the user's modules): it found nothing, so the key
lost ``mylib``'s code and data, and a cell's ``mylib.K = 14`` served the
old ``t``. With 10 objects the walk saw the class.

On Python 3.12+ the trace of what ran (``CASH_TRACE``) caught the stale hit;
the tests turn it off, as before 3.12.
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
    "class Order:\n"
    "    def __init__(self, v):\n"
    "        self.v = v\n"
    "    def total(self, log):\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "        os.write(fd, b'x')\n"
    "        os.close(fd)\n"
    "        return self.v * K\n"
)


@pytest.fixture
def shop(tmp_path, monkeypatch):
    name = f"_held_class_shop_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(SHOP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    # As before Python 3.12: no trace of what ran checks a hit there.
    monkeypatch.setenv("CASH_TRACE", "0")
    monkeypatch.setattr(shadow, "SHADOW", None)
    # The test harness's own classes (``tests.conftest.MockShell``) count as
    # a user module's, with objects alive: in a notebook only the shop's are.
    users = held_functions._users_module_classes
    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [c for c in users() if c.__module__ == name])
    yield name, tmp_path
    sys.modules.pop(name, None)


def _cells(name, tmp_path, size):
    pkl, log = tmp_path / "big.pkl", tmp_path / "runs.log"
    return [
        f"import pickle, types, {name}\nwith open({str(pkl)!r}, 'wb') as f:\n"
        f"    pickle.dump([{name}.Order] + [types.SimpleNamespace(i=i) for i in range({size})], f)",
        f"with open({str(pkl)!r}, 'rb') as f:\n    big = pickle.load(f)",
        f"t = [k(3).total({str(log)!r}) for k in big[:1]][0]",
    ]


@pytest.mark.parametrize("size", [10, 30_000])
def test_a_cell_changing_the_module_data_reruns_the_reader(cash_magics, mock_shell, shop, size):
    name, tmp_path = shop
    cells = _cells(name, tmp_path, size)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["t"] == 6

    run_cash_cell(cash_magics, f"{name}.K = 14", cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert mock_shell.user_ns["t"] == 42


@pytest.mark.parametrize("size", [10, 30_000])
def test_a_module_whose_file_is_gone_runs_the_reader_each_time(cash_magics, mock_shell, shop, size):
    name, tmp_path = shop
    cells = _cells(name, tmp_path, size)
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    (tmp_path / f"{name}.py").unlink()
    run_cash_cell(cash_magics, cells[2], cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert (tmp_path / "runs.log").read_text(encoding="utf-8") == "xx"


@pytest.mark.parametrize("size", [10, 30_000])
def test_unchanged_the_reader_is_served(cash_magics, mock_shell, shop, size):
    """Control: nothing changed, so the reader is served at either size."""
    name, tmp_path = shop
    cells = _cells(name, tmp_path, size)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert (tmp_path / "runs.log").read_text(encoding="utf-8") == "x"


@pytest.fixture
def traced_shop(tmp_path, monkeypatch):
    """The shop with the trace on (3.12+): a walk stops at an object of the
    class, and the hit is served only on its trace."""
    if not shadow.requested() or shadow.SHADOW is None:
        pytest.skip("the trace of what ran is not on here")
    name = f"_held_class_traced_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(SHOP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    users = held_functions._users_module_classes
    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [c for c in users() if c.__module__ == name])
    yield name, tmp_path
    sys.modules.pop(name, None)


def test_an_entry_whose_trace_was_dropped_runs_again(cash_magics, mock_shell, traced_shop):
    """Four traces are kept per statement: after six other inputs the first
    comes back, served on a key that left out the module, and its trace is
    gone. It runs again rather than be served on that key."""
    name, tmp_path = traced_shop
    log = tmp_path / "runs.log"
    cells = [
        f"import types, {name}",
        f"n = int(open({str(tmp_path / 'n.txt')!r}).read())",
        f"big = [{name}.Order] + [types.SimpleNamespace(i=i) for i in range(30_000)] + [n]",
        f"t = big[0](3).total({str(log)!r}) + big[-1]",
        f"{name}.K = 14",
    ]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    for v in [0, 1, 2, 3, 4, 5, 0]:
        (tmp_path / "n.txt").write_text(str(v), encoding="utf-8")
        for cell in cells[1:4]:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["t"] == 6
    run_cash_cell(cash_magics, cells[4], cells=cells)
    run_cash_cell(cash_magics, cells[3], cells=cells)

    assert mock_shell.user_ns["t"] == 42


@pytest.mark.parametrize("size", [10, 30_000])
def test_a_trace_asked_for_but_not_installed_walks_as_without_one(cash_magics, mock_shell, shop, monkeypatch, size):
    """``CASH_TRACE=1`` with the monitoring tool ids another's: nothing
    records what ran, so the walk takes what refers to the class as with
    the trace off."""
    name, tmp_path = shop
    monkeypatch.setenv("CASH_TRACE", "1")
    monkeypatch.setattr(shadow, "enable", lambda: None)
    cells = _cells(name, tmp_path, size)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, f"{name}.K = 14", cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert mock_shell.user_ns["t"] == 42
