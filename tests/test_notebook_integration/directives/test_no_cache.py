"""``# @cash:no-cache``: what it covers and what re-runs because of it.

``# @cash: no-cache`` makes a statement behave as if cash weren't installed:
re-running ADVANCES state (like plain Jupyter), for both reassignment and
in-place mutation. Previously the in-place case was wrongly reset to its
cell-entry base.

Root cause (found with the upstream-trace harness): a no-cache statement still
bumps its var's runtime lineage, so pass 2 of the simulation flagged the var
stale (runtime lineage advanced past the simulation's) and re-executed its
producer -- resetting it. The no-cache exclusion only covered the
stale-value guard's self-write sets, not the pass-2 lineage mismatch. The fix
drops no-cache-written vars from ``broken_vars`` before producer scheduling.

Verification: does a leading ``# @cash:no-cache`` cover the WHOLE cell?

Empirical, external-counter based. Each statement in the annotated cell calls
``bump(tag)``, which appends a line to a file on disk. The file is read from the
TEST side (never by a notebook cell), so cash's file-tracking machinery cannot
double-count it and the badge is never trusted as evidence.

If the directive covered the whole cell, an isolated re-run of that cell would
append 3 more lines (all statements live). If it covers only the NEXT statement,
the re-run appends 1 line and statements 2/3 replay from cache.
"""

import pytest


def _last(out: str) -> str:
    return out.strip().splitlines()[-1].strip()


@pytest.mark.integration
@pytest.mark.upstream
def test_nocache_inplace_mutation_advances(nb_runner):
    nb_runner.create_notebook(
        [
            "log = []",
            "# @cash: no-cache\nlog.append(len(log))\nprint(len(log))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "2", nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.upstream
def test_nocache_reassignment_advances(nb_runner):
    nb_runner.create_notebook(
        [
            "counter = 0",
            "# @cash: no-cache\ncounter = counter + 1\nprint(counter)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "2", nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.upstream
def test_cached_inplace_mutation_still_resets(nb_runner):
    """Control: WITHOUT no-cache, the run-from-start guarantee still resets the
    in-place mutation on an isolated re-run (the fix is scoped to no-cache)."""
    nb_runner.create_notebook(
        [
            "log = []",
            "log.append(len(log))\nprint(len(log))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "1", nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.upstream
def test_nocache_inplace_does_not_reexecute_producer(upstream_trace):
    """The producer of a no-cache-mutated var must not be scheduled for re-run."""
    t = upstream_trace(
        ["log = []", "# @cash: no-cache\nlog.append(len(log))\nprint(len(log))"],
        lambda r: r.run_cell(2),
    )
    assert "log = []" not in t.scheduled(), t.scheduled()
    assert t.events("broken_drop_nocache"), "expected log dropped from broken_vars"


SETUP = "import cash\n%cash_on\n%cash_badge print\nimport time"


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
    return [ln for ln in sink.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.mark.timeout(180)
def test_leading_no_cache_covers_whole_cell(nb_runner, tmp_path):
    sink = tmp_path / "bumps.txt"
    sink.write_text("", encoding="utf-8")
    sink_s = str(sink).replace("\\", "/")

    nb_runner.create_notebook(
        [
            SETUP,
            _bump_def(sink_s),
            # The natural spelling from the ticket: ONE directive at the top of the
            # cell, followed by three top-level statements.
            "# @cash:no-cache\na = bump('s1')\nb = bump('s2')\nc = bump('s3')",
        ]
    )
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
            f"{sorted(set(['s1', 's2', 's3']) - set(got[len(expected) - 3 :]))}\n"
            f"  cell output: {nb_runner.get_output(3)!r}"
        )


@pytest.mark.timeout(180)
def test_statement_adjacent_no_cache_still_scoped(nb_runner, tmp_path):
    """Control: a directive directly above ONE mid-cell statement stays scoped
    to that statement (this is the behaviour any fix must preserve)."""
    sink = tmp_path / "bumps2.txt"
    sink.write_text("", encoding="utf-8")
    sink_s = str(sink).replace("\\", "/")

    nb_runner.create_notebook(
        [
            SETUP,
            _bump_def(sink_s),
            "a = bump('s1')\n# @cash:no-cache\nb = bump('s2')\nc = bump('s3')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _tags(sink) == ["s1", "s2", "s3"], _tags(sink)

    nb_runner.run_cell(3)
    got = _tags(sink)
    # Record whatever actually happens; s2 must at minimum re-fire.
    assert "s2" in got[3:], f"the statement-adjacent no-cache statement did not re-execute: {got}"
    print(f"[no-cache control] after warm re-run, sink = {got}")
