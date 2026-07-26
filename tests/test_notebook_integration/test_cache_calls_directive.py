"""End-to-end sub-expression caching under ``# @cash:cache-calls`` (CAS-243).

The two shapes the feature exists for, both of which statement-level caching
cannot help with:

- ``out.append(compute(t))`` — skip-cached because the append is a mutation, so
  today it re-runs in full on every execution.
- ``s += compute(x)`` — cached, but keyed on the running prefix, so reordering
  the iterable re-runs everything after the first change.

Under the directive the expensive call is cached and the cheap wrapper still
executes, which is both faster *and* more correct than skip-caching: the
mutation genuinely happens every run.

Counted, never timed — wall-clock cannot distinguish "recomputed" from
"restored but slow".
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Above the cost model's floor, or nothing is stored and every assertion below
# holds whether or not the feature works.
_SLEEP = 0.2


def _helpers(log):
    return (
        "import time, pathlib\n"
        f"LOG = pathlib.Path(r'{log}')\n"
        "def compute(x):\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write(str(x) + '\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return x + 1\n"
    )


def _n(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_append_loop_caches_the_call_under_the_directive(nb_runner, tmp_path):
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "out = []",
        "# @cash:cache-calls\nfor t in [1, 2]:\n    out.append(compute(t))\nprint('OUT', out)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 2, "baseline did not run both iterations"
    assert "OUT [2, 3]" in nb_runner.get_output(3)

    nb_runner.run_cell(3)
    assert _n(log) == 2, (
        "compute() re-ran; the directive did not cache the call inside the append"
    )
    assert "OUT [2, 3]" in nb_runner.get_output(3), (
        "the append stopped executing -- the mutation must still happen every run"
    )


def test_without_the_directive_nothing_changes(nb_runner, tmp_path):
    """Positive control: the gate must actually gate.

    Without this, a bug that switched interception on unconditionally would
    still pass the test above.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "out = []",
        "for t in [1, 2]:\n    out.append(compute(t))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 2

    nb_runner.run_cell(3)
    assert _n(log) == 4, "interception engaged without the directive"


def test_reordering_an_accumulator_fold_costs_nothing(nb_runner, tmp_path):
    """The user-reported case: a reorder must stop re-running the tail.

    Note where the directive sits. ``@cash:`` directives attach to the
    statement *below* them and the backwards scan stops at the first
    non-comment line, so on the cell's first line it would scope to ``s = 0``
    and never reach the loop. It has to be on the loop header.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "s = 0\n# @cash:cache-calls\nfor x in [1, 10, 5]:\n    s += compute(x)\nprint('SUM', s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 3
    assert "SUM 19" in nb_runner.get_output(2)

    nb_runner.set_cell_source(
        2, "s = 0\n# @cash:cache-calls\nfor x in [5, 10, 1]:\n    s += compute(x)\nprint('SUM', s)"
    )
    nb_runner.run_cell(2)
    assert "SUM 19" in nb_runner.get_output(2)
    assert _n(log) == 3, (
        "a reorder re-ran compute(); the call cache is order-independent by "
        "construction and must absorb it"
    )

    # A genuinely new value costs exactly one call, and no more.
    nb_runner.set_cell_source(
        2, "s = 0\n# @cash:cache-calls\nfor x in [5, 10, 1, 7]:\n    s += compute(x)\nprint('SUM', s)"
    )
    nb_runner.run_cell(2)
    assert "SUM 27" in nb_runner.get_output(2)
    assert _n(log) == 4, "adding one item cost more than one call"
