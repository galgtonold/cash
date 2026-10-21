"""A statement whose outputs hold one of the process's standard streams keeps
what it was stored, served or run uncached as, and the warnings it gave.

A ``threading.Thread`` keeps the ``sys.stderr`` it was made under, a
``logging.StreamHandler`` and a ``tqdm`` bar the stream they write to. The
sharing check looked for the variables holding that stream up the heap's
referrers: past the stream, through what the test runner or the kernel keeps,
eight full heap scans for every such statement, and then gave up. A test
worker's heap made one statement take seconds, and the thread tests ran into
their 30 s timeout on Windows. The process's standard streams are no longer
climbed from; a file object of the user's still is.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook import shared_objects as so
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

#: A cost above what the disk tier keeps, so a statement can be stored.
SLOW = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
THREAD = [
    "import time, threading\nstop = threading.Event()",
    # A thread that ends at once: while one a statement started runs, no
    # statement is served (`thread_reach`), which is not what this checks.
    f"t = threading.Thread(target=stop.wait, args=(0,))\nt.start()\n{SLOW}",
    f"n = ({SLOW}, len([t]) + 1)[1]",
    f"pair = ({SLOW}, [t, 1])[1]",
]
HANDLER = [
    "import time, logging, sys",
    f"h = ({SLOW}, logging.StreamHandler(sys.stderr))[1]",
    "log = logging.getLogger('cash-pin')\nlog.addHandler(h)",
    f"k = ({SLOW}, h.level + 1)[1]",
    f"pair = ({SLOW}, [h, 1])[1]",
]
BAR = [
    "import time, sys\nfrom tqdm import tqdm",
    f"bar = ({SLOW}, tqdm(total=3, file=sys.stderr))[1]",
    "bar.update(1)",
    f"m = ({SLOW}, bar.n + 1)[1]",
    f"pair = ({SLOW}, [bar, 1])[1]",
]
NAMES = ["time", "pair", "threading", "stop", "t", "n", "logging", "sys", "h", "log", "k", "tqdm", "bar", "m"]


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: what a later kernel reads is what the disk tier kept."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _run(magics, cells: list[str]) -> list[tuple[list[str | None], list[str]]]:
    """Each cell's statement statuses and the warning codes it gave."""
    out = []
    for cell in cells:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_cash_cell(magics, cell, cells=cells)
        statuses = [s.get("status") for s in magics.cash_status("dict")["last_cell"]["statements"]]
        codes = sorted({c for w in caught if (c := getattr(w.message, "code", None)) and c != "NOTEBOOK-NOT-FOUND"})
        out.append((statuses, codes))
    return out


def _outcomes(magics, cells: list[str]) -> list:
    first = _run(magics, cells)
    again = _run(magics, cells)
    _restart(magics)
    after_restart = _run(magics, cells)
    return [first, again, after_restart]


def _stop(magics) -> None:
    ns = magics.shell.user_ns
    if "stop" in ns:
        ns["stop"].set()
        if "t" in ns:
            ns["t"].join(5)
    if "log" in ns and "h" in ns:
        ns["log"].removeHandler(ns["h"])
    if "bar" in ns:
        ns["bar"].close()


C, R, S = "COMPUTED", "RESTORED", "SKIPPED"
#: Each statement's status, cell by cell: the first run, a re-run of the
#: cells, and a run after a restart. What is restored is what a reader of the
#: object stored; the object itself, and a list holding it, always run.
#: Recorded before the streams were left out of the climb, and unchanged by
#: it.
EXPECTED = {
    "thread": [
        [[C, C], [C, C, C], [C], [C]],
        [[S, C], [C, C, R], [R], [C]],
        [[C, C], [C, C, R], [R], [C]],
    ],
    "stream handler": [
        [[C], [C], [C, C], [C], [C]],
        [[S], [C], [C, C], [R], [C]],
        [[C], [C], [C, C], [R], [C]],
    ],
    "tqdm bar": [
        [[C, C], [C], [C], [C], [C]],
        [[S, S], [C], [C], [R], [C]],
        [[C, C], [C], [C], [R], [C]],
    ],
}


@pytest.mark.parametrize("cells", [THREAD, HANDLER, BAR], ids=list(EXPECTED))
def test_what_is_stored_served_and_run_is_unchanged(cash_magics, cells, request):
    if cells is BAR:
        pytest.importorskip("tqdm")
    try:
        found = _outcomes(cash_magics, cells)
    finally:
        _stop(cash_magics)
    statuses = [[[str(getattr(s, "value", s)) for s in cell] for cell, _codes in run] for run in found]
    assert statuses == EXPECTED[request.node.callspec.id]
    assert all(codes == [] for run in found for _cell, codes in run), "no warning, as before"


def _inside(value) -> list:
    nodes = {id(value): value}
    inside, _owners = so._held_from_outside(nodes, {id(value): "v"}, so.VALUE_TYPES + so.library_value_types())
    return list(inside.values())


def test_a_thread_s_stderr_is_not_climbed_from():
    import threading

    stop = threading.Event()
    t = threading.Thread(target=stop.wait, args=(5,))
    try:
        assert t._stderr is sys.stderr, "control: the thread keeps the stream"
        assert not any(obj is sys.stderr for obj in _inside(t))
    finally:
        stop.set()


def test_a_handler_on_a_file_of_the_user_s_is_still_climbed_from(tmp_path):
    import logging

    with open(tmp_path / "log.txt", "w", encoding="utf-8") as f:
        handler = logging.StreamHandler(f)
        assert any(obj is f for obj in _inside(handler)), "the user's file is looked for"
        on_stderr = logging.StreamHandler(sys.stderr)
        assert not any(obj is sys.stderr for obj in _inside(on_stderr))
