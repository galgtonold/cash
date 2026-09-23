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

import ast

import pytest

pytest.importorskip("IPython")

from cash.notebook.upstream.checker import UpstreamChecker


def _msg(stmt: str, error: str) -> str:
    return UpstreamChecker._format_upstream_failure(stmt, error)


def test_a_missing_name_points_at_the_cell_that_defines_it():
    m = _msg("ax.plot(sub['month'], sub['margin_pct'])", "NameError: name 'sub' is not defined")
    assert "run the cell that defines 'sub'" in m
    assert "fix the upstream cell" not in m, (
        "the old advice is the misleading one for a NameError -- there is usually nothing to fix"
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
    m = _msg("x = f('''weird''')\nsecond line", "NameError: name 'sub' is not defined")
    assert "\n" not in m
    assert "'''" not in m


class _Checker(UpstreamChecker):
    """Just enough of a checker to exercise the planning-gap detection."""

    def __init__(self, executed_cell_codes):  # noqa: D107 - test double
        self.executed_cell_codes = executed_cell_codes
        self.debug = False
        self.simulator = type("_Sim", (), {"get_cached_ast": staticmethod(ast.parse)})()


def test_a_missing_name_with_a_known_producer_names_it_as_a_cash_gap():
    """The distinction the old message could not draw.

    A NameError here has two causes that look identical to the reader: the cell
    genuinely never ran (their problem, run it), or cash scheduled the statement
    that READS a name without the one that WRITES it (cash's problem, running
    cells will not help). When the evidence is unambiguous, say which.
    """
    checker = _Checker({"sub": "sub = mm[mm['category'] == 'Electronics']"})
    gap = checker._planning_gap_for(
        NameError("name 'sub' is not defined"),
        already_scheduled=[],
    )
    assert gap is not None
    assert "'sub' is set by" in gap
    assert "gap in cash's re-execution plan" in gap


def test_no_gap_is_claimed_when_nothing_produces_the_name():
    """The ordinary case -- the user really has not run anything defining it."""
    checker = _Checker({"other": "other = 1"})
    assert checker._planning_gap_for(NameError("name 'sub' is not defined"), already_scheduled=[]) is None


def test_no_gap_is_claimed_when_the_producer_was_already_scheduled():
    """If cash DID schedule it, the failure is something else; do not misblame."""
    producer = "sub = mm[mm['category'] == 'Electronics']"
    checker = _Checker({"sub": producer})
    assert checker._planning_gap_for(NameError("name 'sub' is not defined"), already_scheduled=[producer]) is None


def test_a_non_nameerror_never_claims_a_gap():
    checker = _Checker({"sub": "sub = 1"})
    assert checker._planning_gap_for(ValueError("bad column"), already_scheduled=[]) is None


def test_the_gap_note_rides_along_in_the_message():
    m = UpstreamChecker._format_upstream_failure(
        "ax.plot(sub['month'])",
        "NameError: name 'sub' is not defined",
        planning_gap="NOTE: 'sub' is set by 'sub = mm[...]'",
    )
    assert "NOTE: 'sub' is set by" in m
    assert "\n" not in m and "'''" not in m


# --- Round 25: the advice reaches the user on the path failures really take ---
#
# A repaired statement runs silently, so the processor reports its failure in
# ``result['error']`` rather than raising. That path formatted ``str(error)`` --
# ``'f1'`` for a KeyError -- with no planning-gap check, and all four round-25
# repair failures (r25s1, r25s2 x2, r25s3) told the tester to "fix the upstream
# cell", which had nothing wrong in it.

from cash.exceptions import UpstreamStateError


def _fail_with(checker, stmt, error, cells, scheduled=()):
    def process(code, ttl, silent, **kw):
        return {"error": error} if code == stmt else {"status": "computed"}

    with pytest.raises(UpstreamStateError) as raised:
        checker._reexecute_statements([*scheduled, stmt], process, None, notebook_cells=cells)
    return str(raised.value)


def test_a_repair_nameerror_names_the_line_cash_did_not_re_run():
    cell = "file_median = 5\nin_cents = file_median > 1\nsales.loc[in_cents, ['price']] /= 100"
    m = _fail_with(
        _Checker({}),
        "sales.loc[in_cents, ['price']] /= 100",
        NameError("name 'in_cents' is not defined"),
        ["import cash", cell],
    )
    assert "NameError: name 'in_cents' is not defined" in m, m
    assert "fix the upstream cell" not in m, m
    assert "'in_cents' is set by 'in_cents = file_median > 1' (cell 2)" in m, m
    assert "run cell 2" in m and "report" in m, m


def test_a_repair_keyerror_names_the_column_write_above_it():
    cell = "results['f1'] = 2 * results['p']\nbest = results.sort_values(['f1'])"
    m = _fail_with(_Checker({}), "best = results.sort_values(['f1'])", KeyError("f1"), [cell])
    assert "KeyError: 'f1'" in m, m
    assert "is set by \"results['f1'] = 2 * results['p']\" (cell 1)" in m, m
    assert "fix the upstream cell" not in m, m


def test_a_repair_keyerror_names_the_loop_that_fills_the_dict():
    cells = ["results = {}\nfor name in ['logreg']:\n    results[name] = name", "fig = draw(results['logreg'])"]
    m = _fail_with(_Checker({}), "fig = draw(results['logreg'])", KeyError("logreg"), cells, scheduled=["results = {}"])
    assert "'logreg' is set by \"for name in ['logreg']:\" (cell 1)" in m, m


def test_a_repair_attributeerror_names_the_column_it_reads():
    cells = ["sales['week'] = sales['date']", "weekly = sales[sales.week <= 3]"]
    m = _fail_with(
        _Checker({}),
        "weekly = sales[sales.week <= 3]",
        AttributeError("'DataFrame' object has no attribute 'week'"),
        cells,
    )
    assert "'week' is set by \"sales['week'] = sales['date']\" (cell 1)" in m, m


def test_a_statement_that_ran_before_is_not_blamed_on_the_user():
    m = _fail_with(
        _Checker({"best": "best = pick(results)"}), "best = pick(results)", KeyError("x"), ["best = pick(results)"]
    )
    assert "ran without error before" in m and "fix the upstream cell" not in m, m


def test_a_repair_that_really_fails_still_says_to_fix_the_cell():
    m = _fail_with(_Checker({}), "df = load()", ValueError("bad column"), ["df = load()"])
    assert "ValueError: bad column" in m and "fix the upstream cell and re-run" in m, m
    assert "NOTE" not in m, m


def test_a_writer_below_the_failing_statement_is_not_blamed():
    cell = "best = results.sort_values(['f1'])\nresults['f1'] = 1"
    m = _fail_with(_Checker({}), "best = results.sort_values(['f1'])", KeyError("f1"), [cell])
    assert "NOTE" not in m, m
