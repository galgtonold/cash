"""When cash steps aside, it must say so where the user can see it.

Reported: running the demo tour's cells repeatedly produced a cell with an
empty badge and no output. The badge half is fixed (it no longer claims
EXECUTED), but the badge was only ever the symptom -- the *cause* was written
with ``logger.error("Cash auto-caching failed: ...")`` to a logger that nothing
in a notebook surfaces. A user hitting this saw a cell do nothing and had no
way to find out why, and the bug behind it left no trace anyone could report.

The bail-out now emits ``NOTEBOOK-BAILOUT`` as a real warning, so the exception
that caused it lands next to the cell.

Forcing it: ``CodeAnalyzer._parse_cell`` is called early in the pipeline inside
a ``try/except SyntaxError``. Raising something that is NOT a SyntaxError --
and not one of the three exception types with their own branches -- lands in
``_handle_pipeline_exception``'s final else, which is the bail-out under test.
"""
import pytest

pytestmark = pytest.mark.timeout(180)

SETUP = (
    "import cash\n"
    "%cash_on\n"
    "import warnings\n"
    "warnings.simplefilter('always')"
)

BREAK_THE_PIPELINE = (
    "from cash.notebook.analysis import CodeAnalyzer\n"
    "CodeAnalyzer._parse_cell = staticmethod(\n"
    "    lambda *a, **k: (_ for _ in ()).throw(ValueError('probe: forced internal failure')))\n"
    "print('pipeline broken')\n"
)

WORK = "answer = 6 * 7\nprint('answer', answer)\n"


@pytest.mark.fresh_kernel
def test_a_bailout_tells_the_user_why(nb_runner):
    """Needs its own kernel: it patches ``CodeAnalyzer._parse_cell`` at class
    level, and a warm kernel would carry that break into the next test -- as
    it did, failing the control arm below with a leaked ValueError."""
    r = nb_runner.create_notebook([SETUP, BREAK_THE_PIPELINE, WORK])
    r.start_kernel()
    r.run_all()

    out = r.get_raw_output(3)

    # The cell's own work must still happen -- cash steps aside, it does not
    # eat the cell. This is the half the user noticed was missing.
    assert "answer 42" in out, (
        f"the cell did not run after cash bailed out: {out!r}")

    assert "NOTEBOOK-BAILOUT" in out, (
        f"cash bailed out silently -- no diagnostic reached the cell: {out!r}")
    assert "ValueError" in out, (
        f"the diagnostic did not name the exception that caused it: {out!r}")
    assert "probe: forced internal failure" in out, (
        f"the diagnostic dropped the underlying message: {out!r}")


def test_a_healthy_cell_says_nothing_about_bailing_out(nb_runner):
    """The control arm.

    Without it, this suite passes just as well if NOTEBOOK-BAILOUT were
    emitted on every cell.
    """
    r = nb_runner.create_notebook([SETUP, WORK])
    r.start_kernel()
    r.run_all()

    out = r.get_raw_output(2)
    assert "answer 42" in out
    assert "NOTEBOOK-BAILOUT" not in out, (
        f"a healthy cell reported a bail-out: {out!r}")
