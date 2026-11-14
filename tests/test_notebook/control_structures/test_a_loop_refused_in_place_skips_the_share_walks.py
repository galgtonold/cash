"""A loop whose in-place change cannot be restored walks no share check for it.

A loop over sessions growing a frame (``frames.append(new)``, ``df =
pd.concat([df, new])``) is run as one statement, and ``frames``, which it
changes in place, holds more objects than a restore checks
(`in_place.refusal`): the loop runs every time. That refusal is asked before
the share check of its outputs, whose identity and alias walks were for a
store that cannot happen, and the outputs its structure's check asks about
right after (``record_structure_share_edges``) are left to that one. What
plain Python gives, and the edges a later change moves a reader through,
stay the same, also after the loop stopped on an exception or an interrupt.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import in_place
from cash.notebook.statement import processor as statement_processor
from tests._cell_driver import run_cash_cell

pytest.importorskip("pandas")

SETUP = (
    "import pandas as pd\n"
    "logs = [{'uid': k, 'actions': [{'t': j, 'v': (k * 7 + j) % 11 * 0.5} for j in range(3)]} for k in range(400)]\n"
    "frames = []\n"
    "df = pd.DataFrame()\n"
    "stop, stop_with = None, ValueError"
)
LOOP = (
    "for i, s in enumerate(logs):\n"
    "    new = pd.DataFrame(s['actions'])\n"
    "    new['sid'] = i\n"
    "    if i == stop:\n"
    "        raise stop_with('stopped')\n"
    "    frames.append(new)\n"
    "    df = pd.concat([df, new])"
)
READ = (
    "# @cash:persist\nsummary = (len(df), len(frames), float(df['v'].sum()), float(sum(f['v'].sum() for f in frames)))"
)
READ_LOGS = "# @cash:persist\nlog_total = sum(a['v'] for r in logs for a in r['actions'])"
# Changes the session the loop left in ``s``, which ``logs`` holds.
CHANGE_SESSION = "s['actions'].append({'t': 99, 'v': 1000.0})"
# Changes the frame the loop left in ``new``, which ``frames`` holds too.
CHANGE = "new['v'] = new['v'] + 100.0"


class _Plain:
    """The same cells run by plain Python."""

    def __init__(self) -> None:
        self.ns: dict = {}

    def run(self, code: str) -> None:
        exec(compile(code, "<plain>", "exec"), self.ns)


def _summary(ns: dict) -> tuple:
    return ns["summary"]


def _walks(monkeypatch) -> dict[str, list]:
    """The identity walks (`unrestorable_output_reason`) and the outputs
    each statement's own share check asks about."""
    seen: dict[str, list] = {"identity": [], "shared": [], "unstored": []}
    real_reason = statement_processor.unrestorable_output_reason
    real_record = statement_processor.StatementProcessor._record_shared_object_edges

    def reason(outputs, *args, **kwargs):
        seen["identity"].append(set(outputs))
        return real_reason(outputs, *args, **kwargs)

    def record(self, run, captured_vars, *args, **kwargs):
        if "fallback_floor" not in kwargs:  # a statement's, not its structure's
            seen["shared"].append({name for name in run.outputs if name in captured_vars})
        return real_record(self, run, captured_vars, *args, **kwargs)

    real_unstore = statement_processor.StatementProcessor._unstore_in_place

    def unstore(self, run, why):
        seen["unstored"].append(why)
        return real_unstore(self, run, why)

    monkeypatch.setattr(statement_processor, "unrestorable_output_reason", reason)
    monkeypatch.setattr(statement_processor.StatementProcessor, "_unstore_in_place", unstore)
    monkeypatch.setattr(statement_processor.StatementProcessor, "_record_shared_object_edges", record)
    return seen


@pytest.fixture
def few_objects(monkeypatch):
    """`in_place.refusal` looks through 300 objects at most: 400 small
    frames are past it, as 3,000 frames are past the real bound."""
    monkeypatch.setattr(in_place, "_MAX_SEEN", 300)


def _start(cash_magics):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    plain = _Plain()
    run_cash_cell(cash_magics, SETUP)
    plain.run(SETUP)
    return plain


def test_a_refused_loop_walks_nothing_and_gives_what_plain_gives(cash_magics, few_objects, monkeypatch):
    ns = cash_magics.shell.user_ns
    plain = _start(cash_magics)
    walks = _walks(monkeypatch)
    for _ in range(3):
        walks["identity"].clear()
        walks["shared"].clear()
        walks["unstored"].clear()
        run_cash_cell(cash_magics, LOOP)
        plain.run(LOOP)
        assert any("frames (it holds too many objects" in why for why in walks["unstored"])
        # No identity walk; the statement's own share check leaves to its
        # structure's the names that one asks about.
        assert walks["identity"] == []
        assert walks["shared"] == [{"i"}]
        run_cash_cell(cash_magics, READ)
        plain.run(READ)
        assert _summary(ns) == _summary(plain.ns)
        assert len(ns["frames"]) == len(plain.ns["frames"])


def test_a_change_through_the_loops_last_session_reruns_its_reader(cash_magics, few_objects):
    """``s`` is ``logs[-1]``: the edge the structure's check records
    between them, which the statement left to it, moves ``logs`` when
    ``s`` changes, and its reader runs again."""
    ns = cash_magics.shell.user_ns
    plain = _start(cash_magics)
    for cell in (LOOP, READ_LOGS):
        run_cash_cell(cash_magics, cell)
        plain.run(cell)
    assert "logs" in cash_magics.tracking_state.derivation_edges.get("s", ())
    run_cash_cell(cash_magics, READ_LOGS)
    assert [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]] == ["RESTORED"]
    for cell in (CHANGE_SESSION, READ_LOGS):
        run_cash_cell(cash_magics, cell)
        plain.run(cell)
    assert ns["log_total"] == plain.ns["log_total"]


@pytest.mark.parametrize("stop_with", ["ValueError", "KeyboardInterrupt"])
def test_a_loop_stopped_part_way_leaves_what_plain_leaves(cash_magics, few_objects, stop_with):
    """Stopped by an exception or an interrupt, the loop records nothing of
    its own run, as before; what reads its names after is what plain gives,
    and so is a change through the frame it left in ``new``."""
    ns = cash_magics.shell.user_ns
    plain = _start(cash_magics)
    for cell in (LOOP, READ):
        run_cash_cell(cash_magics, cell)
        plain.run(cell)
    stop = f"stop, stop_with = 317, {stop_with}"
    run_cash_cell(cash_magics, stop)
    plain.run(stop)
    exc = KeyboardInterrupt if stop_with == "KeyboardInterrupt" else ValueError
    with pytest.raises(exc):
        run_cash_cell(cash_magics, LOOP)
    with pytest.raises(exc):
        plain.run(LOOP)
    for cell in (READ, CHANGE, READ, CHANGE_SESSION, READ_LOGS, "stop = None", LOOP, READ, READ_LOGS):
        run_cash_cell(cash_magics, cell)
        plain.run(cell)
        if cell == READ:
            assert _summary(ns) == _summary(plain.ns)
        if cell == READ_LOGS:
            assert ns["log_total"] == plain.ns["log_total"]
