"""A call site follows a function it calls, whether the function is defined
above or BELOW the caller.

A function whose body calls a name bound in a LATER cell has a dependency
pointing down the notebook: ``def a`` in cell 2 calls ``b`` from cell 3.
Lineage flows upward, so the call site must learn about ``b`` from the call
itself. The tests run the same notebook with the callee above and below the
caller and check that:

* an edit to the callee refreshes the call site, on Run All and on a rerun of
  the call site alone;
* removing the callee and restarting raises, as a plain kernel does, instead of reprinting the old number for code that no longer runs.

Mutual recursion always has one function defined below the other, which is how
this first surfaced.
"""

import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"
CALLER = "def a(n):\n    return b(n) * 2"
CALLEE = "def b(n):\n    return n + 1"
CALLSITE = "r = a(3)\nprint(f'r={r}')"


def test_callee_above_caller_propagates_an_edit(nb_runner):
    """The control: with the callee ABOVE, editing it refreshes the call site."""
    nb_runner.create_notebook([C_ON, CALLEE, CALLER, CALLSITE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "r=8" in nb_runner.get_output(4)

    nb_runner.set_cell_source(2, "def b(n):\n    return n + 10")
    nb_runner.run_cell(4)
    assert "r=26" in nb_runner.get_output(4), f"callee above the caller must propagate: {nb_runner.get_output(4)!r}"


def test_callee_above_caller_surfaces_its_removal(nb_runner):
    """The control: with the callee ABOVE, deleting it surfaces the NameError."""
    nb_runner.create_notebook([C_ON, CALLEE, CALLER, CALLSITE])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "r=8" in nb_runner.get_output(4)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.set_cell_source(2, "pass")  # b no longer exists anywhere
    with pytest.raises(Exception):
        nb_runner.run_all()


def test_callee_below_caller_propagates_an_edit_on_run_all(nb_runner):
    """Callee BELOW: editing it and re-running the notebook refreshes the call site."""
    nb_runner.create_notebook([C_ON, CALLER, CALLEE, CALLSITE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "r=8" in nb_runner.get_output(4)

    nb_runner.set_cell_source(3, "def b(n):\n    return n + 10")
    nb_runner.run_all()
    assert "r=26" in nb_runner.get_output(4), (
        f"callee below the caller must propagate on Run All: {nb_runner.get_output(4)!r}"
    )


def test_callee_below_caller_propagates_an_edit_on_isolated_rerun(nb_runner):
    """Callee BELOW: editing it and re-running ONLY the call site."""
    nb_runner.create_notebook([C_ON, CALLER, CALLEE, CALLSITE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "r=8" in nb_runner.get_output(4)

    nb_runner.set_cell_source(3, "def b(n):\n    return n + 10")
    nb_runner.run_cell(4)
    assert "r=26" in nb_runner.get_output(4), f"callee below the caller must propagate: {nb_runner.get_output(4)!r}"


def test_callee_below_caller_surfaces_its_removal(nb_runner):
    """Same deletion, callee BELOW: a fresh run raises, so cash must not serve."""
    nb_runner.create_notebook([C_ON, CALLER, CALLEE, CALLSITE])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "r=8" in nb_runner.get_output(4)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.set_cell_source(3, "pass")  # b no longer exists anywhere
    with pytest.raises(Exception):
        nb_runner.run_all()
