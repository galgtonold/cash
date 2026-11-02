"""The debug line of a statement that runs every time names why.

The lookup it skips said "missing input lineage or @cash:no-cache" whatever
the reason, which sent a reader of ``%cash_debug`` looking for a lineage
that was there all along (``data.sort(key=...)``, skipped for its in-place
change).
"""

from __future__ import annotations

import logging

from tests._cell_driver import run_cash_cell

LOGGER = "cash.notebook.statement.processor"


def _skip_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Skipping cache lookup" in r.getMessage()]


def test_an_in_place_change_is_named_as_the_reason(cash_magics, caplog):
    run_cash_cell(cash_magics, "data = [3, 1, 2]")
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        # Positive control: a statement that is looked up logs no skip.
        run_cash_cell(cash_magics, "y = sorted(data)")
        assert not _skip_lines(caplog)
        run_cash_cell(cash_magics, "data.sort(key=lambda x: -x)")
    lines = _skip_lines(caplog)
    assert lines, "the in-place sort skips its lookup"
    assert all("missing input lineage" not in line for line in lines)
    assert any("In-place mutation on: data" in line for line in lines)


def test_a_no_cache_statement_says_so(cash_magics, caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        run_cash_cell(cash_magics, "x = 1  # @cash:no-cache")
    lines = _skip_lines(caplog)
    assert lines
    assert all("missing input lineage" not in line for line in lines)
    assert any("@cash:no-cache" in line for line in lines)
