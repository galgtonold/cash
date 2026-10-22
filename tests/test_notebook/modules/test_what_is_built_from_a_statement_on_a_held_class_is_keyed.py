"""What is built from a statement reading a big value that holds a class of
the user's module is keyed by that module too, with the what-ran trace on.

The walk of the big value stopped short when no object of the class existed,
leaning on the trace to check the hit; the trace checks only its own
statement. A cell below it (``s = t + 1``) was served 7 after ``t`` reran to
42; a call cached inside a statement was served when its own trace could not
be used (a process pool) or was dropped; after a restart the upstream check
restored ``t`` with no trace at all; and a statement whose trace could never
be used was never served.
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
    "def price(big, log):\n"
    "    return big[0](3).total(log)\n"
    "def price_pooled(big, log):\n"
    "    import multiprocessing\n"
    "    with multiprocessing.get_context('fork').Pool(1) as p:\n"
    "        p.map(abs, [1])\n"
    "    return big[0](3).total(log)\n"
)


@pytest.fixture
def shop(tmp_path, monkeypatch):
    if not shadow.requested() or shadow.SHADOW is None:
        pytest.skip("trace not on")
    name = f"_held_built_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(SHOP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    users = held_functions._users_module_classes
    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [c for c in users() if c.__module__ == name])
    yield name, tmp_path
    sys.modules.pop(name, None)


def _counts():
    s = shadow.SHADOW
    return {k: v for k, v in s.counts.items() if k.startswith(("stmt:", "call:"))} if s else None


# Shapes the call cache takes (``big[0](3).total(log)`` and ``k(3).total(log)``
# are not wrapped as calls, so they are left out).
CALLS = {
    "modfn": "{name}.price(big, {log!r})",
    "modfn_kw": "{name}.price(big=big, log={log!r})",
    "bound": "p(big, {log!r})",
}


@pytest.mark.parametrize("call", list(CALLS))
@pytest.mark.parametrize("size", [10, 30_000])
@pytest.mark.parametrize("variants", [1, 6])
@pytest.mark.parametrize("edit", [False, True], ids=["control", "edit"])
def test_call_hit_with_dropped_trace(cash_magics, mock_shell, shop, size, variants, call, edit):
    """The statement misses (``m`` is new) and runs; the call inside it is
    looked up in the call cache on a key that leaned on the trace, and that
    call's trace was dropped (4 are kept per call). ``big`` is loaded from a
    pickle, so ``mod.K = 14`` does not move its lineage."""
    name, tmp_path = shop
    log = str(tmp_path / "runs.log")
    d = str(tmp_path)
    cells = [
        f"import types, pickle, {name}\np = {name}.price\nk = {name}.Order\n"
        f"for _v in range(6):\n    pickle.dump([{name}.Order] + [types.SimpleNamespace(i=i) for i in range({size})] + [_v], open({d!r} + f'/big{{_v}}.pkl', 'wb'))",
        f"n = int(open({d!r} + '/n.txt').read())",
        f"m = int(open({d!r} + '/m.txt').read())",
        f"big = pickle.load(open({d!r} + f'/big{{n}}.pkl', 'rb'))",
        "t = " + CALLS[call].format(name=name, log=log) + " + m",
        f"{name}.K = 14",
    ]

    def run(c):
        run_cash_cell(cash_magics, c, cells=cells)

    run(cells[0])
    (tmp_path / "m.txt").write_text("0")
    run(cells[2])
    for v in range(variants):
        (tmp_path / "n.txt").write_text(str(v))
        run(cells[1])
        run(cells[3])
        run(cells[4])
    assert mock_shell.user_ns["t"] == 6
    if edit:
        run(cells[5])
    (tmp_path / "n.txt").write_text("0")
    run(cells[1])
    run(cells[3])
    (tmp_path / "m.txt").write_text("1")
    run(cells[2])
    before = open(log, encoding="utf-8").read()
    run(cells[4])
    if edit:
        assert mock_shell.user_ns["t"] == 43, (before, open(log, encoding="utf-8").read(), _counts())
    else:
        assert mock_shell.user_ns["t"] == 7
        assert open(log, encoding="utf-8").read() == before, "control: call not served"


@pytest.mark.parametrize("call", ["bound", "modfn"])
@pytest.mark.parametrize("size", [10, 30_000])
@pytest.mark.parametrize("edit", [False, True], ids=["control", "edit"])
def test_call_hit_after_traces_lost(cash_magics, mock_shell, shop, size, call, edit):
    """As in a restarted kernel whose traces file no longer has the call's
    trace (``disk: false`` traces are dropped at load; 4 per group are kept
    at load): the traces are cleared by hand, the entries stay."""
    name, tmp_path = shop
    log = str(tmp_path / "runs.log")
    d = str(tmp_path)
    cells = [
        f"import types, pickle, {name}\np = {name}.price\n"
        f"pickle.dump([{name}.Order] + [types.SimpleNamespace(i=i) for i in range({size})], open({d!r} + '/big.pkl', 'wb'))",
        f"m = int(open({d!r} + '/m.txt').read())",
        f"big = pickle.load(open({d!r} + '/big.pkl', 'rb'))",
        "t = " + CALLS[call].format(name=name, log=log) + " + m",
        f"{name}.K = 14",
    ]

    def run(c):
        run_cash_cell(cash_magics, c, cells=cells)

    (tmp_path / "m.txt").write_text("0")
    for c in cells[:4]:
        run(c)
    assert mock_shell.user_ns["t"] == 6
    shadow.SHADOW.traces.clear()
    shadow.SHADOW.groups.clear()
    if edit:
        run(cells[4])
    (tmp_path / "m.txt").write_text("1")
    run(cells[1])
    before = open(log, encoding="utf-8").read()
    run(cells[3])
    if edit:
        assert mock_shell.user_ns["t"] == 43, (before, open(log, encoding="utf-8").read(), _counts())
    else:
        assert mock_shell.user_ns["t"] == 7
        assert open(log, encoding="utf-8").read() == before, "control: call not served"


HELPERS = (
    "import os, time\n"
    "class Unused:\n"
    "    pass\n"
    "def mark(log):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    return 1\n"
    "def pooled():\n"
    "    import multiprocessing\n"
    "    with multiprocessing.get_context('fork').Pool(2) as p:\n"
    "        return sum(p.map(abs, range(4)))\n"
)


@pytest.mark.parametrize("size", [10, 30_000])
@pytest.mark.parametrize("pool", [False, True], ids=["nopool", "pool"])
@pytest.mark.parametrize("module", [False, True], ids=["nomodule", "module"])
def test_unusable_trace_cost(cash_magics, mock_shell, tmp_path, monkeypatch, size, pool, module):
    """Cost: a statement reading a big list of plain rows (no class of the
    user's in it), whose trace is unusable (a process pool ran), with a
    module of the user's loaded whose class has no object alive. Served on
    the second run?"""
    if not shadow.requested() or shadow.SHADOW is None:
        pytest.skip("trace not on")
    name = f"_zz3_helpers_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(HELPERS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    users = held_functions._users_module_classes
    monkeypatch.setattr(
        held_functions, "_users_module_classes", lambda: [c for c in users() if c.__module__ == name] if module else []
    )
    log = str(tmp_path / "runs.log")
    work = f"{name}.pooled()" if pool else "0"
    cells = [
        f"import multiprocessing, {name}",
        f"rows = [{{'i': i, 'v': float(i)}} for i in range({size})]",
        f"# @cash:no-cache-calls\nr = ({work}, len(rows), {name}.mark({log!r}))",
    ]
    try:
        for c in cells:
            run_cash_cell(cash_magics, c, cells=cells)
        run_cash_cell(cash_magics, cells[2], cells=cells)
        run_cash_cell(cash_magics, cells[2], cells=cells)
        run_cash_cell(cash_magics, cells[2], cells=cells)
        assert open(log, encoding="utf-8").read() == "x", (
            "not served (runs logged)",
            open(log, encoding="utf-8").read(),
            _counts(),
        )
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("size", [10, 30_000])
@pytest.mark.parametrize("edit", [False, True], ids=["control", "edit"])
def test_call_hit_with_unusable_trace(cash_magics, mock_shell, shop, size, edit):
    """No eviction: the cached call's own trace is unusable (it started a
    process pool), so its hit is served on its key alone, and that key
    leaned on the trace."""
    name, tmp_path = shop
    log = str(tmp_path / "runs.log")
    d = str(tmp_path)
    cells = [
        f"import types, pickle, {name}\np = {name}.price_pooled\n"
        f"pickle.dump([{name}.Order] + [types.SimpleNamespace(i=i) for i in range({size})], open({d!r} + '/big.pkl', 'wb'))",
        f"m = int(open({d!r} + '/m.txt').read())",
        f"big = pickle.load(open({d!r} + '/big.pkl', 'rb'))",
        f"t = p(big, {log!r}) + m",
        f"{name}.K = 14",
    ]

    def run(c):
        run_cash_cell(cash_magics, c, cells=cells)

    (tmp_path / "m.txt").write_text("0")
    for c in cells[:4]:
        run(c)
    assert mock_shell.user_ns["t"] == 6
    if edit:
        run(cells[4])
    (tmp_path / "m.txt").write_text("1")
    run(cells[1])
    before = open(log, encoding="utf-8").read()
    run(cells[3])
    if edit:
        assert mock_shell.user_ns["t"] == 43, (before, open(log, encoding="utf-8").read(), _counts())
    else:
        assert mock_shell.user_ns["t"] == 7
        assert open(log, encoding="utf-8").read() == before, ("control: call not served", _counts())


@pytest.mark.parametrize("size", [10, 30_000])
@pytest.mark.parametrize("how", ["run_all", "last_cell"])
@pytest.mark.parametrize("edit", [False, True], ids=["control", "edit"])
def test_restart_then_upstream_restore(tmp_path, monkeypatch, size, how, edit):
    """Restart (a second kernel on the same cache folder), the module's file
    edited in between (``K = 14``). Running only the last cell, the upstream
    check restores ``t`` from the cache (`CacheRestorer.try_virtual_restore`)
    with no trace check; Run All looks ``t`` up as a statement."""
    if not shadow.requested():
        pytest.skip("trace not on")
    from cash.core import Cash
    from cash.notebook.ipython.magics import CashMagics
    from tests.conftest import MockShell

    name = f"_held_restart_{os.getpid()}_{id(tmp_path)}"
    mod = tmp_path / f"{name}.py"
    mod.write_text(SHOP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    users = held_functions._users_module_classes
    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [c for c in users() if c.__module__ == name])
    import importlib
    import pickle
    import types

    importlib.invalidate_caches()
    m = importlib.import_module(name)
    pkl = tmp_path / "big.pkl"
    pkl.write_bytes(pickle.dumps([m.Order] + [types.SimpleNamespace(i=i) for i in range(size)]))
    sys.modules.pop(name, None)
    log = str(tmp_path / "runs.log")
    cells = [
        f"import pickle, types, {name}",
        f"with open({str(pkl)!r}, 'rb') as f:\n    big = pickle.load(f)",
        f"t = big[0](3).total({log!r})",
        "s = t + 0",
    ]

    def kernel():
        magics = CashMagics(MockShell(), Cash(cache_dir=str(tmp_path / "cache"), register_magic=False))
        magics.cash_on("")
        magics.cash_persist("on")
        magics.badges.mode = "off"
        return magics

    try:
        first = kernel()
        if shadow.SHADOW is None:
            pytest.skip("trace not installed")
        for c in cells:
            run_cash_cell(first, c, cells=cells)
        assert first.shell.user_ns["s"] == 6
        try:
            first._cell_executor.flush_writes()
        except Exception:
            pass
        import time

        time.sleep(1.0)
        if edit:
            mod.write_text(SHOP.replace("K = 2", "K = 14"), encoding="utf-8")
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        second = kernel()
        before = open(log, encoding="utf-8").read()
        if how == "run_all":
            for c in cells:
                run_cash_cell(second, c, cells=cells)
        else:
            run_cash_cell(second, cells[3], cells=cells)
        got = second.shell.user_ns.get("s")
        if edit:
            assert got == 42, (
                got,
                second.shell.user_ns.get("t"),
                before,
                open(log, encoding="utf-8").read(),
                _counts(),
            )
        else:
            assert got == 6
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("size", [10, 30_000])
@pytest.mark.parametrize("edit", ["cell", "file"])
@pytest.mark.parametrize("trace", ["on", "off"])
def test_downstream_of_a_trace_rerun(cash_magics, mock_shell, shop, size, edit, trace, monkeypatch):
    """Same kernel, no eviction. ``t`` is served on its trace, which sees
    the change and runs it again (42). Is ``s = t + 1``, below it, run
    again too?"""
    name, tmp_path = shop
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
        monkeypatch.setattr(shadow, "SHADOW", None)
    log = str(tmp_path / "runs.log")
    pkl = tmp_path / "big.pkl"
    cells = [
        f"import pickle, types, {name}\nwith open({str(pkl)!r}, 'wb') as f:\n"
        f"    pickle.dump([{name}.Order] + [types.SimpleNamespace(i=i) for i in range({size})], f)",
        f"with open({str(pkl)!r}, 'rb') as f:\n    big = pickle.load(f)",
        f"t = big[0](3).total({log!r})",
        "s = t + 1",
        f"{name}.K = 14",
    ]
    for c in cells[:4]:
        run_cash_cell(cash_magics, c, cells=cells)
    assert mock_shell.user_ns["s"] == 7
    if edit == "cell":
        run_cash_cell(cash_magics, cells[4], cells=cells)
    else:
        import importlib

        (tmp_path / f"{name}.py").write_text(SHOP.replace("K = 2", "K = 14"), encoding="utf-8")
        importlib.reload(sys.modules[name])
    run_cash_cell(cash_magics, cells[2], cells=cells)
    run_cash_cell(cash_magics, cells[3], cells=cells)
    assert (mock_shell.user_ns["t"], mock_shell.user_ns["s"]) == (42, 43), _counts()
