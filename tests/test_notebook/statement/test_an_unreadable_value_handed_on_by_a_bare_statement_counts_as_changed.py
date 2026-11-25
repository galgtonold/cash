"""A value cash cannot read, handed on by a statement that keeps nothing,
counts as changed.

An object holding a lock has no content to fingerprint before and after a
statement. Handed whole to a bare call (``bump(state)``) it was already
assumed changed. Handed on in a list or to a lambda given to ``map``
(``list(map(lambda s: s["tracker"].inc(), [state]))``) it was taken as only
read, as it is by a call whose result is kept (``rows = fetch(conn)``): the
change went unseen, and a reader below was served its old result.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SETUP = (
    "import threading\n"
    "class Tracker:\n"
    "    def __init__(self):\n"
    "        self.n = 0\n"
    "        self.lock = threading.Lock()\n"
    "    def inc(self):\n"
    "        self.n += 1\n"
    "def report(s):\n"
    "    return s['tracker'].n\n"
    "def fetch(s):\n"
    "    return s['tracker'].n\n"
    "state = {'tracker': Tracker()}"
)

CHANGERS = {
    "a lambda given to map over a list": "list(map(lambda s: s['tracker'].inc(), [state]))",
    "a lambda given to map over a repeated list": "list(map(lambda s: s['tracker'].inc(), [state] * 1))",
}


@pytest.fixture
def notebook(cash_magics, cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    run_cash_cell(cash_magics, SETUP)
    return cash_magics


@pytest.mark.parametrize("case", sorted(CHANGERS))
def test_a_reader_below_sees_the_change(notebook, case):
    changer = CHANGERS[case]
    run_cash_cell(notebook, changer)
    run_cash_cell(notebook, "out = report(state)")
    assert notebook.shell.user_ns["out"] == 1, "control: the first run"

    run_cash_cell(notebook, changer + "  # again")
    run_cash_cell(notebook, "out = report(state)")

    assert notebook.shell.user_ns["out"] == 2


def test_a_call_whose_result_is_kept_still_leaves_the_value_as_read(notebook, statement_processor):
    """Control: ``rows = fetch(state)`` keeps a result, so the unreadable
    value it is handed is not assumed changed and the statement is stored."""
    skipped: list[bool] = []
    post_execute = statement_processor._post_execute

    def noting(run, execution):
        post_execute(run, execution)
        skipped.append(bool(run.skip_cache))

    statement_processor._post_execute = noting

    run_cash_cell(notebook, "rows = fetch(state)")

    assert skipped == [False]
