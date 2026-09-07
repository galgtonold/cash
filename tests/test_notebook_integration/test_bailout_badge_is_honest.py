"""When cash steps aside, the badge must not claim the cell executed.

Reported symptom: a badge that is "just empty and says executed which is
wrong", appearing while work was still going on.

Mechanism. Six paths in ``cell_executor`` abort cash's pipeline and publish an
empty badge to clear the progress placeholder (lines 817, 950, 1327, 1332,
1347, 1351) -- a SyntaxError from the AST parse, a SyntaxError out of the
upstream simulation, and ``Cash auto-caching failed ... falling back to normal
execution``. Each then hands the cell to ``original_run_cell``.

An empty badge had no state of its own, so it fell through the summary
renderer to the EXECUTED default and rendered ``EXECUTED 0.00s``. The cell's
real work then began -- so the badge announced completion *before the work
started*, and simultaneously hid the fact that caching had not run at all.

A SyntaxError cell is the cheapest way to reach an abort path from outside.
"""
import re

import pytest

pytestmark = pytest.mark.timeout(180)

SETUP = (
    "import cash\n"
    "%cash_on"
)

# Reaches CodeAnalyzer._parse_cell's SyntaxError branch, which publishes the
# empty badge and then falls back to normal execution.
SYNTAX_ERROR_CELL = "x = (1 +\n"

GOOD_CELL = "import time\nval = [time.sleep(0.4), 7][1]\nprint('ok', val)\n"


def _badge_text(runner, cell_num):
    """Rendered badge markup for the cell, stylesheet stripped.

    Read off the cell's stored ``text/html`` outputs, not ``get_raw_output``,
    which returns only stream text -- the badge is a display_data payload.
    """
    cell = runner.get_cell(cell_num)
    parts = []
    for out in cell.get("outputs", []):
        html = (out.get("data") or {}).get("text/html", "")
        if isinstance(html, list):
            html = "".join(html)
        if html:
            parts.append(html)
    return re.sub(r"<style>.*?</style>", "", "".join(parts), flags=re.DOTALL)


def test_a_bailed_out_cell_does_not_report_that_it_executed(nb_runner):
    r = nb_runner.create_notebook([SETUP, SYNTAX_ERROR_CELL])
    r.start_kernel()
    # The cell legitimately raises once cash hands it back to IPython, and the
    # runner propagates that. The outputs -- including the badge cash already
    # published on the way out -- are stored before it does, which is what
    # this test reads.
    try:
        r.run_all()
    except Exception:
        pass

    badge = _badge_text(r, 2)
    if "c3-card" not in badge:
        pytest.skip("no badge rendered for this abort path in this harness")

    labels = re.findall(r'c3-summary-label[^>]*>([^<]+)<', badge)
    assert "EXECUTED" not in labels, (
        f"cash bailed out and then ran the cell normally, but the badge "
        f"reported EXECUTED: {labels}"
    )


def test_a_normal_cell_still_reports_that_it_executed(nb_runner):
    """The control arm.

    Without it, the assertion above passes just as well if EXECUTED stops
    being rendered anywhere -- which would hide every real execution instead
    of only the false one.
    """
    r = nb_runner.create_notebook([SETUP, GOOD_CELL])
    r.start_kernel()
    r.run_all()

    labels = re.findall(
        r'c3-summary-label[^>]*>([^<]+)<', _badge_text(r, 2))
    assert "EXECUTED" in labels, (
        f"a normal cell should still summarise as EXECUTED, got {labels}")
