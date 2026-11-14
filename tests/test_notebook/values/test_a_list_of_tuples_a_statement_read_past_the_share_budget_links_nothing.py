"""A list of tuples a statement read, past the share budget, links nothing.

When the share check after a statement runs out of time, each output it
could not clear is linked to every value the statement read. ``latest`` is
a session's list of actions, which ``sessions`` holds; ``time_intervals``,
read beside it, is a list of tuples of numbers a loop filled, which only its
own name refers to. Linked to ``latest`` all the same, the loop's next
``append`` moved ``latest`` on, and through it ``sessions``, so the cell
below rebuilt the sessions from their parse. A value read that does share an
object with an output, or that an output is bound to, stays linked.
"""

from __future__ import annotations

import pytest

from cash.notebook import shared_objects
from cash.notebook.statement import output_refusals, processor
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

PARSE = """import os, time, random
def parse_log(n, log):
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b'p')
    os.close(fd)
    time.sleep(SLEEP)
    r = random.Random(0)
    return [{'id': i, 'actions': [{'at': r.randrange(99)} for _ in range(6)]} for i in range(n)]"""

INTERVALS = """time_intervals = []
for session in sessions:
    for i in range(len(session['actions']) - 1):
        time_intervals.append((i, session['actions'][i + 1]['at'] - session['actions'][i]['at']))"""

LATEST = "latest, n = sessions[-1]['actions'], len(time_intervals)"


@pytest.fixture
def latest_check_runs_out(monkeypatch):
    """The share check of ``latest`` runs out of time, and so does the
    search up that stands in for it: its outputs are linked to what it
    read."""
    real = processor.share_group

    def spent(names, *args, **kwargs):
        if "latest" in names:
            raise shared_objects.WalkBudgetExceeded
        return real(names, *args, **kwargs)

    monkeypatch.setattr(processor, "share_group", spent)
    monkeypatch.setattr(output_refusals, "share_group", spent)
    monkeypatch.setattr(processor, "referring_names", lambda *_args, **_kwargs: (set(), False))


def _cells(log, latest: str = LATEST, intervals: str = INTERVALS) -> list[str]:
    return [
        PARSE.replace("SLEEP", repr(ABOVE_PERSISTENCE_FLOOR_S)),
        f"sessions = parse_log(20, {str(log)!r})  # @cash:no-cache",
        intervals,
        latest,
        "total = sum(a['at'] for s in sessions for a in s['actions']) + len(time_intervals)",
    ]


def _partners(magics, name: str) -> set[str]:
    edges = magics.tracking_state.derivation_edges
    return set(edges.get(name, ())) | {src for src, targets in edges.items() if name in targets}


def _run_all(magics, cells: list[str]) -> None:
    magics.cash_on("")
    magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


@pytest.mark.usefixtures("latest_check_runs_out")
def test_the_intervals_are_not_linked_to_what_shares_the_sessions(cash_magics, tmp_path):
    log = tmp_path / "parse.log"
    cells = _cells(log)
    _run_all(cash_magics, cells)
    assert "sessions" in _partners(cash_magics, "latest"), "control: the check was cut short and linked what it read"
    assert "time_intervals" not in _partners(cash_magics, "latest")

    more = "time_intervals.append((99, 0))"
    cells = [*cells[:4], more, cells[4]]
    for code in (more, cells[5]):
        run_cash_cell(cash_magics, code, cells=cells)

    assert log.read_bytes() == b"p"
    ns = cash_magics.shell.user_ns
    assert ns["total"] == sum(a["at"] for s in ns["sessions"] for a in s["actions"]) + 20 * 5 + 1


@pytest.mark.usefixtures("latest_check_runs_out")
def test_intervals_holding_a_session_stay_linked(cash_magics, tmp_path):
    """Guard: ``time_intervals`` holding a session's own dict shares it
    with ``sessions``, and with ``latest`` through it: a change through it
    moves ``sessions`` on."""
    intervals = INTERVALS + "\ntime_intervals.append(sessions[-1])"
    log = tmp_path / "parse.log"
    cells = _cells(log, intervals=intervals)
    _run_all(cash_magics, cells)
    assert "latest" in _partners(cash_magics, "time_intervals")
    lineage = cash_magics.tracking_state.variable_lineage["sessions"]

    change = "time_intervals[-1]['actions'][0]['at'] = 1000"
    look = "first = sessions[-1]['actions'][0]['at']"
    notebook = [*cells, change, look]
    run_cash_cell(cash_magics, change, cells=notebook)
    assert cash_magics.tracking_state.variable_lineage["sessions"] != lineage

    # Each cell gives what Restart + Run All gives it: ``total`` sits above
    # the change, so it sums the parse; ``look`` sits below it and sees 1000,
    # as plain Python does.
    ns = cash_magics.shell.user_ns
    for code in (look, cells[4], look):
        run_cash_cell(cash_magics, code, cells=notebook)
        if code == cells[4]:
            run_all: dict = {}
            for cell in cells:
                exec(cell.replace(repr(ABOVE_PERSISTENCE_FLOOR_S), "0"), run_all)
            assert ns["total"] == run_all["total"]
        else:
            assert ns["first"] == 1000


@pytest.mark.usefixtures("latest_check_runs_out")
def test_an_output_bound_to_the_intervals_stays_linked(cash_magics, tmp_path):
    """Guard: ``alias`` is the list ``time_intervals`` is: an append
    through it changes ``time_intervals``."""
    latest = "latest, n, alias = sessions[-1]['actions'], len(time_intervals), time_intervals"
    log = tmp_path / "parse.log"
    cells = _cells(log, latest=latest)
    _run_all(cash_magics, cells)
    assert "alias" in _partners(cash_magics, "time_intervals")
    lineage = cash_magics.tracking_state.variable_lineage["time_intervals"]

    change = "alias.append((0, 0))"
    run_cash_cell(cash_magics, change, cells=[*cells, change])

    assert cash_magics.tracking_state.variable_lineage["time_intervals"] != lineage


def test_what_shares_nothing_is_told_for_many_names_in_one_look():
    def look() -> tuple[dict, dict]:
        ns = {
            "counts": [(1, 2), (3, 4)],
            "pairs": {("a", "b"): 1},
            "rows": [[1]],
            "held": [1, 2],
            "number": 3,
        }
        ns["alias"] = ns["counts"]
        holder = [ns["held"]]
        many = shared_objects.names_holding_alone(list(ns), ns)
        one = {name: shared_objects.held_only_by_names(name, ns) for name in ns}
        del holder
        return many, one

    many, one = look()
    assert many == {
        "counts": frozenset({"counts", "alias"}),
        "alias": frozenset({"counts", "alias"}),
        "pairs": frozenset({"pairs"}),
    }
    assert {name: bound for name, bound in one.items() if bound is not None} == many
