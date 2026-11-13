"""The share check's give-up on a frame sharing its lists with the list it
was built from is taken again for the same objects, without walking them;
anything that changes those objects asks again.

student_8: ``df = pd.DataFrame(logs)`` shares each session's list with
``logs``, 87,000 of them. Each cheap statement on ``df`` (``df['n'] =
...``) walked the alias and identity checks past the 0.1 s budget only to
give up as the statement before had, a second a statement. Taken again,
the statement is refused the same, and the variables holding its objects
are still looked for from the outputs up, so a write through ``logs`` or
``df`` moves the other as before.

The digest of ``logs`` as a holder of ``df`` ran out of its budget each
statement too and was thrown away; known too costly, it is not read again
while ``logs`` is bound to the same list with the same lineage.

Pins the work, not only the behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

import time

import pytest

from cash import value_hash
from cash.analysis import analysis_budget
from cash.notebook.statement import output_refusals, processor
from cash.notebook.statement.share_giveups import CostlyDigests
from tests._cell_driver import run_cash_cell


def _busy(seconds: float) -> None:
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        pass


class _Walks:
    """Counts the share checks that walk the outputs' values; each one is
    made as slow as the alias check of student_8's sessions, past the
    share check's budget, so that it gives up."""

    def __init__(self, monkeypatch) -> None:
        self.count = 0
        real = output_refusals.identity_coupled_reason

        def slow(*args, **kwargs):
            self.count += 1
            _busy(0.3)
            return real(*args, **kwargs)

        monkeypatch.setattr(output_refusals, "identity_coupled_reason", slow)
        # The search from the outputs up that follows a give-up reads the
        # garbage collector's referrers, whose cost grows with every object
        # the process holds: in a worker that ran a few hundred tests it was
        # cut short before it found ``logs``, nothing was remembered, and
        # each statement walked again. That search's time limit is not what
        # this pins; the share check's own (0.1 s) stays real.
        search = processor.referring_names

        def unhurried(values, user_ns, budget, *args, **kwargs):
            return search(values, user_ns, max(budget, 60.0), *args, **kwargs)

        monkeypatch.setattr(processor, "referring_names", unhurried)
        monkeypatch.setattr(analysis_budget, "ANALYSIS_BUDGET_FLOOR_S", 60.0)


def _frame_over_logs(cash_magics) -> None:
    run_cash_cell(cash_magics, "import pandas as pd")
    run_cash_cell(cash_magics, "logs = [(f'u{i}', [i, i + 1]) for i in range(200)]")
    run_cash_cell(cash_magics, "df = pd.DataFrame(logs, columns=['user', 'actions'])")


def _taken_again(cash_magics, walks: _Walks) -> None:
    """``df`` changed in place twice: the second gives up without the walk."""
    run_cash_cell(cash_magics, "df['n'] = 1")
    before = walks.count
    run_cash_cell(cash_magics, "df['n2'] = 1")
    assert walks.count == before


@pytest.mark.real_time_limits
def test_a_statement_on_the_same_frame_gives_up_again_without_the_walk(cash_magics, monkeypatch):
    walks = _Walks(monkeypatch)
    _frame_over_logs(cash_magics)
    before = walks.count
    run_cash_cell(cash_magics, "df['n'] = 1")
    run_cash_cell(cash_magics, "df['m'] = 2")
    assert walks.count == before
    assert list(cash_magics.shell.user_ns["df"].columns) == ["user", "actions", "n", "m"]


@pytest.mark.real_time_limits
@pytest.mark.parametrize(
    "between",
    [
        pytest.param("logs = list(logs)", id="holder-rebound"),
        pytest.param("logs[0][1].append(9)", id="holder-changed-in-place"),
    ],
)
def test_a_change_to_a_holder_asks_again(cash_magics, monkeypatch, between):
    walks = _Walks(monkeypatch)
    _frame_over_logs(cash_magics)
    _taken_again(cash_magics, walks)
    run_cash_cell(cash_magics, between)
    before = walks.count
    run_cash_cell(cash_magics, "df['m'] = 2")
    assert walks.count == before + 1


@pytest.mark.real_time_limits
@pytest.mark.parametrize(
    "change",
    [
        pytest.param("df.sort_values('user', inplace=True)", id="columns-replaced"),
        pytest.param("df = df.copy()", id="output-rebound"),
        pytest.param("df['actions'] = 0", id="shared-column-replaced"),
    ],
)
def test_a_change_to_the_output_asks_again(cash_magics, monkeypatch, change):
    walks = _Walks(monkeypatch)
    _frame_over_logs(cash_magics)
    _taken_again(cash_magics, walks)
    before = walks.count
    run_cash_cell(cash_magics, change)
    assert walks.count == before + 1


@pytest.mark.real_time_limits
def test_an_edge_dropped_asks_again(cash_magics, monkeypatch):
    walks = _Walks(monkeypatch)
    _frame_over_logs(cash_magics)
    _taken_again(cash_magics, walks)
    edges = cash_magics.tracking_state.derivation_edges
    for src, dst in (("df", "logs"), ("logs", "df")):
        edges.get(src, set()).discard(dst)
    before = walks.count
    run_cash_cell(cash_magics, "df['m'] = 2")
    assert walks.count == before + 1


@pytest.mark.real_time_limits
@pytest.mark.parametrize(
    ("write", "moved"),
    [
        pytest.param("logs[0][1].append(9)", "df", id="through-logs"),
        pytest.param("df['actions'][0].append(9)", "logs", id="through-df"),
    ],
)
def test_a_write_through_either_still_moves_the_other(cash_magics, monkeypatch, write, moved):
    walks = _Walks(monkeypatch)
    _frame_over_logs(cash_magics)
    _taken_again(cash_magics, walks)
    before = walks.count
    run_cash_cell(cash_magics, "df['m'] = 2")
    assert walks.count == before  # taken again
    lineage = cash_magics.tracking_state.variable_lineage
    was = lineage[moved]
    run_cash_cell(cash_magics, write)
    assert lineage[moved] != was


class _Digests:
    """Counts the holder digests read within a budget, each one out of
    time, as the digest of student_8's ``logs`` was."""

    def __init__(self, monkeypatch) -> None:
        self.count = 0
        real = value_hash.budgeted_hash

        def out_of_time(*args, **kwargs):
            if kwargs.get("remember", True):
                return real(*args, **kwargs)
            self.count += 1
            return None

        monkeypatch.setattr(value_hash, "budgeted_hash", out_of_time)


def test_a_holder_digest_out_of_time_is_not_read_again(cash_magics, monkeypatch):
    digests = _Digests(monkeypatch)
    _frame_over_logs(cash_magics)
    run_cash_cell(cash_magics, "df['n'] = 1")
    first = digests.count
    assert first >= 1
    run_cash_cell(cash_magics, "df['m'] = 2")
    run_cash_cell(cash_magics, "df['k'] = 3")
    assert digests.count == first


def test_a_holder_digest_is_read_again_once_the_holder_changes(cash_magics, monkeypatch):
    digests = _Digests(monkeypatch)
    _frame_over_logs(cash_magics)
    run_cash_cell(cash_magics, "df['n'] = 1")
    first = digests.count
    run_cash_cell(cash_magics, "df['n2'] = 1")
    assert digests.count == first
    run_cash_cell(cash_magics, "logs[0][1].append(9)")
    before = digests.count
    run_cash_cell(cash_magics, "df['m'] = 2")
    assert digests.count > before


def test_a_holder_bound_to_another_object_is_not_known_costly():
    costly = CostlyDigests()
    logs, other = [1], [1]
    costly.begin()
    costly.note("logs", logs)
    costly.settle({"logs": logs}, {"logs": "L1"}, changed=())
    costly.begin()
    assert costly.known("logs", logs, "L1")
    assert not costly.known("logs", other, "L1")
    assert not costly.known("logs", logs, "L1")  # forgotten with it
