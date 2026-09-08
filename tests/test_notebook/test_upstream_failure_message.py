"""An upstream failure message must point at the remedy that exists.

Round-14 gate finding, the second half of the BLOCKING plot-cell issue. When
reconstruction re-executed a statement whose input was missing, the user got:

    ... failed during auto-re-execution: NameError: name 'sub' is not defined.
    Cash stopped instead of running this cell against stale upstream state -
    fix the upstream cell and re-run.

There was nothing to fix. The cell was fine; it had simply never run in that
kernel. The reporter had five cells blocked at once and spent the time reading
a cell that turned out to be correct, because the message sent them there.
"""
from __future__ import annotations

import pytest

pytest.importorskip("IPython")

from cash.notebook.upstream.checker import UpstreamChecker


def _msg(stmt: str, error: str) -> str:
    return UpstreamChecker._format_upstream_failure(stmt, error)


def test_a_missing_name_points_at_the_cell_that_defines_it():
    m = _msg("ax.plot(sub['month'], sub['margin_pct'])",
             "NameError: name 'sub' is not defined")
    assert "run the cell that defines 'sub'" in m
    assert "fix the upstream cell" not in m, (
        "the old advice is the misleading one for a NameError -- there is "
        "usually nothing to fix"
    )


def test_a_real_failure_still_says_to_fix_the_cell():
    """The old advice is correct when the statement genuinely raised."""
    m = _msg("df = load()", "ValueError: bad column")
    assert "fix the upstream cell and re-run" in m
    assert "run the cell that defines" not in m


def test_the_message_names_the_statement_and_the_error():
    m = _msg("ax.plot(sub['month'])", "NameError: name 'sub' is not defined")
    assert "ax.plot(sub['month'])" in m
    assert "NameError: name 'sub' is not defined" in m


def test_the_message_stays_embeddable():
    """It is re-raised inside the user's cell via a generated triple-quoted
    ``raise ...('''<msg>''')``, so it must stay one line and carry no ``'''``."""
    m = _msg("x = f('''weird''')\nsecond line",
             "NameError: name 'sub' is not defined")
    assert "\n" not in m
    assert "'''" not in m
