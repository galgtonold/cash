"""CAS-189 verification: does a leading ``# @cash:no-cache`` cover the WHOLE cell?

Empirical, external-counter based. Each statement in the annotated cell calls
``bump(tag)``, which appends a line to a file on disk. The file is read from the
TEST side (never by a notebook cell), so cash's file-tracking machinery cannot
double-count it and the badge is never trusted as evidence.

If the directive covered the whole cell, an isolated re-run of that cell would
append 3 more lines (all statements live). If it covers only the NEXT statement,
the re-run appends 1 line and statements 2/3 replay from cache.
"""
import pytest

pytestmark = [pytest.mark.timeout(180)]

SETUP = (
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print\n"
    "import time"
)


def _bump_def(sink: str) -> str:
    return (
        f"SINK = r'{sink}'\n"
        "def bump(tag):\n"
        "    time.sleep(0.05)\n"
        "    with open(SINK, 'a') as f:\n"
        "        f.write(tag + '\\n')\n"
        "    return tag"
    )


def _tags(sink):
    return [ln for ln in sink.read_text().splitlines() if ln.strip()]


def test_leading_no_cache_covers_whole_cell(nb_runner, tmp_path):
    sink = tmp_path / "bumps.txt"
    sink.write_text("")
    sink_s = str(sink).replace("\\", "/")

    nb_runner.create_notebook([
        SETUP,
        _bump_def(sink_s),
        # The natural spelling from the ticket: ONE directive at the top of the
        # cell, followed by three top-level statements.
        "# @cash:no-cache\n"
        "a = bump('s1')\n"
        "b = bump('s2')\n"
        "c = bump('s3')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    first = _tags(sink)
    assert first == ["s1", "s2", "s3"], first

    # Isolated warm re-runs of the annotated cell. A whole-cell no-cache means
    # every statement re-executes every time.
    for run in range(2, 5):
        nb_runner.run_cell(3)
        got = _tags(sink)
        expected = ["s1", "s2", "s3"] * run
        assert got == expected, (
            f"warm re-run #{run}: leading '# @cash:no-cache' did NOT cover the "
            f"whole cell.\n  expected {expected}\n  got      {got}\n"
            f"  -> statements that did NOT re-execute: "
            f"{sorted(set(['s1','s2','s3']) - set(got[len(expected) - 3:]))}\n"
            f"  cell output: {nb_runner.get_output(3)!r}"
        )


def test_statement_adjacent_no_cache_still_scoped(nb_runner, tmp_path):
    """Control: a directive directly above ONE mid-cell statement stays scoped
    to that statement (this is the behaviour any fix must preserve)."""
    sink = tmp_path / "bumps2.txt"
    sink.write_text("")
    sink_s = str(sink).replace("\\", "/")

    nb_runner.create_notebook([
        SETUP,
        _bump_def(sink_s),
        "a = bump('s1')\n"
        "# @cash:no-cache\n"
        "b = bump('s2')\n"
        "c = bump('s3')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _tags(sink) == ["s1", "s2", "s3"], _tags(sink)

    nb_runner.run_cell(3)
    got = _tags(sink)
    # Record whatever actually happens; s2 must at minimum re-fire.
    assert "s2" in got[3:], (
        f"the statement-adjacent no-cache statement did not re-execute: {got}"
    )
    print(f"[CAS-189 control] after warm re-run, sink = {got}")
