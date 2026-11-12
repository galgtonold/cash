"""Counts a loop fills hold nothing of what it read, past the share budget too.

When the share check after a statement runs out of time, each output is
linked to every value the statement read: it may hold one of their objects,
and a write through it must move them on. ``trans_count`` -- a numpy array of
ints the loop fills from ``sessions`` -- holds no object at all, and nothing
but its own name refers to it, which its reference count shows. Linked to
``sessions`` all the same, each run of the loop moved ``sessions`` on and the
cell below rebuilt it from its parse. An output that does hold an object the
input holds, or one something the collector cannot see holds, stays linked.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash.notebook import shared_objects
from cash.notebook.statement import output_refusals, processor
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

PARSE = """import os, time, random
import numpy as np
def parse_log(n, log):
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b'p')
    os.close(fd)
    time.sleep(SLEEP)
    r = random.Random(0)
    return [{'id': i, 'actions': [r.randrange(8) for _ in range(6)]} for i in range(n)]"""

COUNTS = """trans_count = np.zeros((8, 8), dtype=int)
for session in sessions:
    for a, b in zip(session['actions'], session['actions'][1:]):
        trans_count[a, b] += 1"""


@pytest.fixture
def loop_check_runs_out(monkeypatch):
    """The share check of the loop runs out of time, and so does the search
    up that stands in for it: every output is linked to what it read."""
    real = processor.share_group

    def spent(names, *args, **kwargs):
        if "trans_count" in names:
            raise shared_objects.WalkBudgetExceeded
        return real(names, *args, **kwargs)

    monkeypatch.setattr(processor, "share_group", spent)
    monkeypatch.setattr(output_refusals, "share_group", spent)
    monkeypatch.setattr(processor, "referring_names", lambda *_args, **_kwargs: (set(), False))


def _cells(log, counts: str = COUNTS) -> list[str]:
    return [
        PARSE.replace("SLEEP", repr(ABOVE_PERSISTENCE_FLOOR_S)),
        f"sessions = parse_log(300, {str(log)!r})  # @cash:no-cache",
        counts,
        "total = int(trans_count.sum())",
    ]


def _partners(magics, name: str) -> set[str]:
    edges = magics.tracking_state.derivation_edges
    return set(edges.get(name, ())) | {src for src, targets in edges.items() if name in targets}


@pytest.mark.usefixtures("loop_check_runs_out")
def test_the_counts_are_not_linked_to_the_sessions(cash_magics, tmp_path):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "parse.log"
    cells = _cells(log)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert "sessions" in _partners(cash_magics, "session"), "control: the check was cut short and linked the rest"

    for index in (2, 3, 2, 3):
        run_cash_cell(cash_magics, cells[index], cells=cells)

    assert "trans_count" not in _partners(cash_magics, "sessions")
    assert log.read_bytes() == b"p"
    assert cash_magics.shell.user_ns["total"] == 300 * 5


@pytest.mark.usefixtures("loop_check_runs_out")
def test_counts_a_hidden_holder_refers_to_stay_linked(cash_magics, tmp_path):
    """Guard: an array of objects, which the collector does not see into,
    holds the counts: their reference count is more than their referrers
    show, so what holds them is not known and they stay linked to what the
    loop read."""
    counts = "box = np.empty(1, dtype=object)\n" + COUNTS + "\n        box[0] = trans_count"
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "parse.log"
    cells = _cells(log, counts)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert "sessions" in _partners(cash_magics, "trans_count")


@pytest.mark.usefixtures("loop_check_runs_out")
def test_a_list_of_what_it_read_stays_linked(cash_magics, tmp_path):
    """Guard: a list of the sessions themselves holds what ``sessions``
    holds, and a change through it moves ``sessions`` on."""
    counts = COUNTS + "\npicked = [s for s in sessions[:3]]"
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "parse.log"
    cells = _cells(log, counts)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert "sessions" in _partners(cash_magics, "picked")
    lineage = cash_magics.tracking_state.variable_lineage["sessions"]

    change = "picked[0]['id'] = 99"
    run_cash_cell(cash_magics, change, cells=[*cells, change])

    assert cash_magics.tracking_state.variable_lineage["sessions"] != lineage


def test_what_shares_nothing_is_told_by_its_count():
    def look() -> dict:
        ns = {
            "counts": np.zeros((3, 3), dtype=int),
            "pairs": {("a", "b"): 1},
            "rows": [[1]],
            "objs": np.empty(2, dtype=object),
        }
        ns["alias"] = ns["counts"]
        found = {name: shared_objects.held_only_by_names(name, ns) for name in ("counts", "pairs", "rows", "objs")}
        view = ns["counts"][0]
        found["counts with a view"] = shared_objects.held_only_by_names("counts", ns)
        del view
        holder = [ns["pairs"]]
        found["pairs held by a list"] = shared_objects.held_only_by_names("pairs", ns)
        del holder
        return found

    assert look() == {
        "counts": frozenset({"counts", "alias"}),
        "pairs": frozenset({"pairs"}),
        "rows": None,
        "objs": None,
        "counts with a view": None,
        "pairs held by a list": None,
    }
