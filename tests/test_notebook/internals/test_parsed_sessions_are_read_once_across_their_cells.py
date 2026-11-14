"""Parsed sessions are read whole once, by the call that built them, and
not again by the identity check or the loop below that watches them.

``sessions = parse_log(path)`` returns a list of dicts of plain records,
and the next cells loop over it to build ``time_intervals`` and
``trans_count``. Each cell paid for reading them again: the identity
check walked ``sessions`` and ``time_intervals`` an object at a time, and
the loop, which hands each session to numpy, fingerprinted ``sessions``
before and after it ran, each a pickle of every record. Now the identity
check reads plain records a level at a time
(`identity_holders._plain_tree_without_ids`), the call's own digest of it
is kept by the watchers (`CallRouting.watch_returned`), and the loop's
fingerprint arms it at once (`LoopWatch`, `value_watch.arm_at_first_sight`):
both fingerprints are handed back.

A session changed between cells is still seen: the loop runs again and
gives what plain Python gives.
"""

from __future__ import annotations

import json

import pytest

from cash import value_hash, value_watch
from cash.notebook.statement import identity_holders
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

SESSIONS, ACTIONS = 300, 60


def _cells(path: str) -> list[str]:
    return [
        "import json, time\nimport numpy as np\n"
        "def parse_log(path):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    out = {}\n"
        "    with open(path) as f:\n"
        "        for line in f:\n"
        "            d = json.loads(line)\n"
        "            out.setdefault(d['sid'], {'id': d['sid'], 'actions': []})['actions'].append(\n"
        "                {'name': d['name'], 'at': d['at'], 'tags': ('x', 'y')})\n"
        "    return list(out.values())",
        f"sessions = parse_log({path!r})",
        "time_intervals = []\n"
        "for session in sessions:\n"
        "    gaps = np.diff([a['at'] for a in session['actions']]).tolist()\n"
        "    time_intervals.extend(zip([a['name'] for a in session['actions'][:-1]], gaps))",
        "trans_count = np.zeros((8, 8), dtype=int)\n"
        "for session in sessions:\n"
        "    actions = session['actions']\n"
        "    for i in range(len(actions) - 1):\n"
        "        trans_count[int(actions[i]['name'][2:]), int(actions[i + 1]['name'][2:])] += 1",
    ]


@pytest.fixture
def log(tmp_path) -> str:
    path = tmp_path / "log.jsonl"
    with path.open("w") as f:
        for k in range(SESSIONS):
            for a in range(ACTIONS):
                f.write(json.dumps({"sid": k, "name": f"a_{(k * 7 + a * 3) % 8}", "at": k * 1000 + a * a}) + "\n")
    return str(path)


def _plain(cells: list[str], after: str = "") -> dict:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    if after:
        exec(after, ns)
        exec(cells[2], ns)
    return ns


@pytest.fixture
def reads(monkeypatch) -> dict[str, int]:
    """Walks an object at a time of ``sessions`` or ``time_intervals`` (a
    list of a record per action), and pickles of ``sessions`` for a
    digest."""
    counted = {"walked": 0, "pickled": 0}
    walk, pickle_digest = identity_holders._Walk.run, value_hash._pickle_digest

    def is_sessions(value) -> bool:
        # By its shape: the call's store checks it before the name binds it.
        return type(value) is list and len(value) == SESSIONS and type(value[0]) is dict and "actions" in value[0]

    def big(value) -> bool:
        return type(value) is list and len(value) >= SESSIONS

    def run(self, roots):
        if not self.quiet and any(map(big, roots)):
            counted["walked"] += 1
        return walk(self, roots)

    def digest(obj):
        if is_sessions(obj):
            counted["pickled"] += 1
        return pickle_digest(obj)

    monkeypatch.setattr(identity_holders._Walk, "run", run)
    monkeypatch.setattr(value_hash, "_pickle_digest", digest)
    return counted


def test_the_cells_give_what_plain_python_gives_reading_sessions_once(cash_magics, mock_shell, log, reads):
    cells = _cells(log)
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell)
    assert reads["walked"] == 0
    run_cash_cell(cash_magics, cells[2])
    run_cash_cell(cash_magics, cells[3])
    plain = _plain(cells)
    assert mock_shell.user_ns["time_intervals"] == plain["time_intervals"]
    assert (mock_shell.user_ns["trans_count"] == plain["trans_count"]).all()
    assert reads["walked"] == 0
    # Without the watchers (or verifying them) each fingerprint around the
    # loop reads it.
    assert reads["pickled"] <= (0 if value_watch.enabled() and not value_watch.VERIFY else 2), reads


def test_a_session_changed_between_cells_is_seen_by_the_loop(cash_magics, mock_shell, log):
    cells = _cells(log)
    for cell in cells[:3]:
        run_cash_cell(cash_magics, cell)
    change = "sessions[5]['actions'][2]['at'] += 999"
    run_cash_cell(cash_magics, change)
    run_cash_cell(cash_magics, cells[2])
    plain = _plain(cells[:3], after=change)
    assert mock_shell.user_ns["time_intervals"] == plain["time_intervals"]
    assert mock_shell.user_ns["time_intervals"] != _plain(cells[:3])["time_intervals"]


def test_a_session_changed_before_the_loop_first_runs_is_seen(cash_magics, mock_shell, log):
    """The digest the call kept is for the result as it returned: a change
    before any loop reads it is a change all the same."""
    cells = _cells(log)
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell)
    change = "sessions[0]['actions'].append({'name': 'a_1', 'at': -5, 'tags': ('x', 'y')})"
    run_cash_cell(cash_magics, change)
    run_cash_cell(cash_magics, cells[2])
    plain = _plain(cells[:2] + [change, cells[2]])
    assert mock_shell.user_ns["time_intervals"] == plain["time_intervals"]
    assert value_hash.compute_hash(mock_shell.user_ns["sessions"]) == value_hash._compute_hash(
        mock_shell.user_ns["sessions"]
    )
