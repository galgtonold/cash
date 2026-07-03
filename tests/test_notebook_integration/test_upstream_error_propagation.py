"""CAS-87: a failing upstream statement must fail the user's cell loudly.

Before the fix, exceptions during upstream auto-re-execution were logged and
swallowed; the downstream cell then computed against the STALE pre-edit value
and printed a result as if nothing happened — a silent wrong-data situation.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.timeout(120), pytest.mark.upstream]


def _setup(nb_runner):
    nb_runner.create_notebook([
        "x = 10",
        "y = x * 2",
        "print('y=' + str(y))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y=20" in nb_runner.get_output(3)


def test_failing_upstream_edit_fails_the_cell(nb_runner):
    _setup(nb_runner)
    # Edit the producer so it now raises a NameError (a typo'd name).
    nb_runner.set_cell_source(1, "x = undefined_name_xyz * 2")

    with pytest.raises(CellExecutionError) as excinfo:
        nb_runner.run_cell(3)
    text = str(excinfo.value)
    assert "UpstreamStateError" in text, (
        f"expected the upstream failure to surface in the cell, got: {text[:400]}"
    )
    assert "undefined_name_xyz" in text, (
        f"error should name the failing upstream statement: {text[:400]}"
    )


def test_fixing_upstream_recovers(nb_runner):
    _setup(nb_runner)
    nb_runner.set_cell_source(1, "x = undefined_name_xyz * 2")
    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(3)

    # Fix the producer: the same isolated re-run must now compute fresh.
    nb_runner.set_cell_source(1, "x = 50")
    nb_runner.run_cell(3)
    assert "y=100" in nb_runner.get_output(3)


def test_unchanged_rerun_still_clean(nb_runner):
    _setup(nb_runner)
    nb_runner.run_cell(3)
    assert "y=20" in nb_runner.get_output(3)
