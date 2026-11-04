"""A ``functools.cache`` helper that reseeds ``random`` as it fills its memo
(``random_cities`` in Norvig's TSP notebook), called by the cells below it.

A statement that fills such a memo is stored with what it put into the
memo only when it took longer than the store floor, which a quick one does
on a loaded machine; on a first run in order the cells below have nothing
to replay. Yet before
each cell the upstream check found the stream's lineage moved and re-ran
statements above: the ``def`` of the helper counted as a draw and a seed
(its body runs when it is called, not when it is defined), the first call
keyed on the lineage its own seed gave the stream instead of an unseeded
one, and the lineage a seed inside the helper gives the stream (taken from
where the stream stands) was one the simulation could not reproduce. Every
``bench(...)`` cell re-ran the ones above it, and a draw re-run on a stream
already past it gave another number than a plain Run All.

Plain Run All is the reference: each cell runs once, and every value is the
one plain Python gives.
"""

from __future__ import annotations

import random

from tests._cell_driver import run_cash_cell

LOG = "bench_calls.log"

NOTE = (
    "def note(tag):\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (tag + '\\n').encode())\n"
    "    os.close(fd)"
)
HELPER = (
    "@functools.cache\n"
    "def random_cities(n, seed=42):\n"
    "    random.seed(seed + n)\n"
    "    return frozenset(round(random.random(), 6) for _ in range(n))"
)
CELLS = [
    f"import functools, os, random\nLOG = {LOG!r}",
    NOTE,
    HELPER,
    "a = sum(random_cities(5))",
    "def tour_a(c):\n    return sorted(c)\ndef rep(c):\n    return random.sample(sorted(c), len(c))",
    "tsps = (tour_a, rep)",
    "@functools.cache\ndef suite(n, k):\n    return tuple(random_cities(n, 42 + i) for i in range(k))",
    "def bench(fns, data):\n    note('bench')\n    return [round(sum(f(c)[0] for c in data), 6) for f in fns]",
    "r1 = bench(tsps, suite(5, 3))",
    "r2 = bench(tsps, suite(6, 3))",
    "x = random.random()",
    "y = x * 2",
]
NAMES = ("a", "r1", "r2", "x", "y")


def _plain(cells: list[str]) -> dict:
    """What restart and Run All in plain Python leaves, the log aside."""
    ns: dict = {"__name__": "__main__"}
    state = random.getstate()
    try:
        for cell in cells:
            exec(cell.replace(f"LOG = {LOG!r}", "LOG = os.devnull"), ns)
    finally:
        random.setstate(state)
    return {name: ns[name] for name in NAMES}


def _bench_calls() -> int:
    try:
        with open(LOG, encoding="utf-8") as fh:
            return len(fh.read().split())
    except FileNotFoundError:
        return 0


def test_a_first_run_in_order_runs_each_cell_once(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    ns = cash_magics.shell.user_ns
    # Each `bench(...)` cell ran once, and the helper's memo was filled once
    # per argument, as in a plain run: nothing above them was replayed (a
    # replayed `def` would have emptied the memo and filled it again). The
    # draw `x` is the plain run's: run again for `y` on a stream already
    # past it, it gave another number.
    assert _bench_calls() == 2
    assert ns["random_cities"].cache_info().misses == 7
    assert {name: ns[name] for name in NAMES} == _plain(CELLS)


def _statuses(magics) -> list[str]:
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def test_a_second_run_all_replays_nothing_either(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    second: list[str] = []
    for run_all in range(2):
        for cell in CELLS:
            run_cash_cell(cash_magics, cell, cells=CELLS)
            if run_all and "bench(tsps" in cell:
                second += _statuses(cash_magics)
    ns = cash_magics.shell.user_ns
    # Each `bench(...)` cell runs once more on the second Run All, unless it
    # was served: whether it was stored at all hangs on how long it took
    # against the store floor (10 ms), which a loaded machine crosses. A
    # statement above replayed would call `bench` again on top of that.
    assert set(second) <= {"COMPUTED", "RESTORED"}, second
    assert _bench_calls() == 2 + second.count("COMPUTED"), second
    assert {name: ns[name] for name in NAMES} == _plain(CELLS)


def _edited() -> list[str]:
    edited = list(CELLS)
    edited[2] = HELPER.replace("seed + n", "seed + 2 * n")
    assert _plain(edited)["a"] != _plain(CELLS)["a"]
    return edited


def test_an_edited_helper_reaches_every_value_below_it(cash_magics, tmp_path, monkeypatch):
    """Control: run in order from the edited helper down, every value is
    the edited plain run's, the draw below included."""
    monkeypatch.chdir(tmp_path)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    edited = _edited()
    for cell in edited[2:]:
        run_cash_cell(cash_magics, cell, cells=edited)
    ns = cash_magics.shell.user_ns
    assert {name: ns[name] for name in NAMES} == _plain(edited)


def test_the_last_cell_run_alone_after_an_edit_above_sees_the_edit(cash_magics, tmp_path, monkeypatch):
    """Control: after the helper is edited, running only the last cell
    re-runs what the edit reaches above it, the helper's callers and the
    draw below them."""
    monkeypatch.chdir(tmp_path)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    edited = _edited()
    run_cash_cell(cash_magics, edited[-1], cells=edited)
    ns = cash_magics.shell.user_ns
    assert {name: ns[name] for name in NAMES} == _plain(edited)
