"""A statement marked ``@cash:persist`` is stored even when the share check
of its outputs takes longer than the statement did.

Before a statement is stored, cash finds the other variables holding an
object of its outputs, within a budget of what the statement is worth
(`share_check_budget`: twice its time, at least a tenth of a second); past
it the statement is refused and runs again. A loop of 1,500 small frames ran
0.03 s and the walk over them took 0.10 to 0.16 s, so the loop, marked
``@cash:persist``, ran again three runs in four. Under ``@cash:persist`` the
user has said storing is worth its cost: the check gets at least
`SHARE_CHECK_PERSIST_FLOOR_S`.

The budget for an unmarked statement is pinned to nothing here, so its
check always runs out; the persisted statement's floor is what is left.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.annotations import get_statement_annotations
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell

SETUP = "rows = [{'n': i} for i in range(1500)]"
BUILD = "picked = [row for row in rows if row['n'] % 2]"


@pytest.fixture(autouse=True)
def no_time_for_the_share_check(monkeypatch):
    """Whatever the statement took, the share check of an unmarked
    statement has no time at all."""
    monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 0.0)
    monkeypatch.setattr(processor, "SHARE_CHECK_FACTOR", 0.0)


def _twice(cash_magics, code):
    """What running *code* as a statement says, both times."""
    run_cash_cell(cash_magics, SETUP)
    node = ast.parse(code).body[0]
    results = []
    for _ in range(2):
        cash_magics.shell.user_ns.pop("picked", None)
        results.append(
            cash_magics._statement_processor.process_statement(
                ast.unparse(node), annotation=get_statement_annotations(code, node)
            )
        )
        assert len(cash_magics.shell.user_ns["picked"]) == 750
    return results


def test_out_of_time_the_statement_runs_again(cash_magics):
    first, again = _twice(cash_magics, BUILD)

    assert "would take longer than" in str(first.get("uncacheable_reasons")), first
    assert again["status"] == CacheStatus.COMPUTED


def test_under_persist_it_is_stored(cash_magics):
    """The same statement, marked: the check gets its floor, and the
    second run is a hit."""
    first, again = _twice(cash_magics, BUILD + "  # @cash:persist")

    assert "would take longer than" not in str(first.get("uncacheable_reasons")), first
    assert again["status"] == CacheStatus.RESTORED
