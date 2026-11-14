"""A statement whose value costs more to store than to compute is not
stored, and runs again.

A loop building 6,000 three-row frames in 1.2 s had them copied into the
RAM tier (1.1 s) and pickled for disk (1.15 s, in the background, holding
the interpreter lock while the next statement ran): storing took twice
what running it again would. The store is now priced before anything is
copied, by a sample of the tables (`many_frames_cost`), and refused when
it costs more than the compute, with the reason said.

The sample is measured as the store runs; its price is pinned here
(`priced_at`), so a loaded machine, which slows the statement and the
sample unevenly, cannot move the verdict. That the sample finds the
tables is checked on every call.
"""

from __future__ import annotations

import ast

import pytest

pytest.importorskip("pandas")

from cash.analysis.annotations import get_statement_annotations
from cash.backends import sampled_cost
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement import store
from tests._cell_driver import run_cash_cell

SETUP = "import time\nimport pandas as pd\nt = pd.DataFrame({'user': ['s10'] * 3, 'n': [1, 2, 3]})"
BUILD = "frames = [t.copy() for _ in range(1000)]"


@pytest.fixture
def priced_at(monkeypatch):
    """Pin what storing the tables costs to *seconds*, copying and writing
    alike; returns the counts of tables the real sample found."""
    found: list[int] = []

    def pin(seconds: float) -> list[int]:
        def many_frames_cost(value):
            measured = sampled_cost.many_frames_cost(value)
            if measured is None:
                return None
            found.append(measured.items)
            return measured._replace(copy=seconds / 2, dump=seconds / 2)

        monkeypatch.setattr(store, "many_frames_cost", many_frames_cost)
        return found

    return pin


def _twice(cash_magics, code):
    """What running *code* as a statement says, both times."""
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


def test_many_small_tables_cheaper_to_build_than_to_store_are_not_stored(cash_magics, priced_at):
    found = priced_at(60.0)
    first, again = _twice(cash_magics, BUILD)

    assert found and set(found) == {1000}
    assert "tables would take" in (first.get("skipped_reason") or ""), first
    assert again["status"] == CacheStatus.COMPUTED


def test_under_persist_they_are_stored(cash_magics, priced_at):
    """The control: ``@cash:persist`` keeps any value, however dear its
    store, so the same statement is stored."""
    priced_at(60.0)
    first, again = _twice(cash_magics, BUILD + "  # @cash:persist")

    assert not first.get("skipped_reason"), first
    assert again["status"] == CacheStatus.RESTORED


def test_when_computing_them_costs_more_than_storing_they_are_stored(cash_magics, priced_at):
    """The control: the same tables, from a statement that took longer than
    storing them does."""
    found = priced_at(0.001)
    first, again = _twice(cash_magics, "frames = time.sleep(0.05) or [t.copy() for _ in range(1000)]")

    assert found and set(found) == {1000}
    assert not first.get("skipped_reason"), first
    assert again["status"] == CacheStatus.RESTORED
