"""A statement whose value costs more to store than to compute is not
stored, and runs again.

A loop building 6,000 three-row frames in 1.2 s had them copied into the
RAM tier (1.1 s) and pickled for disk (1.15 s, in the background, holding
the interpreter lock while the next statement ran): storing took twice
what running it again would. The store is now priced before anything is
copied, by a sample of the tables (`many_frames_cost`), and refused when
it costs more than the compute, with the reason said.
"""

from __future__ import annotations

import ast

import pytest

pytest.importorskip("pandas")

from cash.analysis.annotations import get_statement_annotations
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

SETUP = "import time\nimport pandas as pd\nt = pd.DataFrame({'user': ['s10'] * 3, 'n': [1, 2, 3]})"
BUILD = "frames = [t.copy() for _ in range(1000)]"


def _twice(cash_magics, code):
    """What running *code* as a statement says, the second time."""
    run_cash_cell(cash_magics, SETUP)
    node = ast.parse(code).body[0]
    results = []
    for _ in range(2):
        cash_magics.shell.user_ns.pop("frames", None)
        results.append(
            cash_magics._statement_processor.process_statement(
                ast.unparse(node), annotation=get_statement_annotations(code, node)
            )
        )
        assert len(cash_magics.shell.user_ns["frames"]) == 1000
    return results


def test_many_small_tables_cheaper_to_build_than_to_store_are_not_stored(cash_magics):
    first, again = _twice(cash_magics, BUILD)

    assert "tables would take" in (first.get("skipped_reason") or ""), first
    assert again["status"] == CacheStatus.COMPUTED


def test_under_persist_they_are_stored(cash_magics):
    """The control: ``@cash:persist`` keeps any value, so the same
    statement is stored."""
    first, again = _twice(cash_magics, BUILD + "  # @cash:persist")

    assert not first.get("skipped_reason"), first
    assert again["status"] == CacheStatus.RESTORED


def test_when_computing_them_costs_more_than_storing_they_are_stored(cash_magics):
    """The control: the same tables from a statement that took longer than
    storing them does."""
    first, again = _twice(cash_magics, "frames = time.sleep(1.5) or [t.copy() for _ in range(1000)]")

    assert not first.get("skipped_reason"), first
    assert again["status"] == CacheStatus.RESTORED
