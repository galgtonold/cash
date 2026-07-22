"""A call site does not depend on functions defined BELOW it (CAS-232).

cash reconstructs dependencies upward. A function whose body calls a name bound
in a LATER cell therefore has a dependency pointing down the notebook, and the
call site never learns about it.

This is about DIRECTION, not recursion — isolated by moving one cell:

* callee defined ABOVE the caller -> everything works, because ``def a`` names
  ``b`` as an ordinary input and lineage flows normally;
* callee defined BELOW the caller -> the call site is stale, with no cycle
  anywhere.

Mutual recursion always contains one (each function references the other before
it exists), which is how this first surfaced.

The consequence is worse than staleness, which is why the forward case is
``xfail`` rather than pinned: cash can serve a cached value for code that no
longer runs at all. Remove the callee, restart, and the call site reprints its
old number where a plain kernel raises ``NameError`` — a masked error, not a
stale one.

Fixing it means resolving a function's free names at CALL time and folding the
callees into the call site's key: a second dependency mechanism alongside the
existing input-lineage one, with real over-invalidation risk. Deliberately not
attempted as a patch.
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
    assert "r=26" in nb_runner.get_output(4), (
        f"callee above the caller must propagate: {nb_runner.get_output(4)!r}"
    )


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN, the remaining half of CAS-232. The call site's KEY now carries the "
        "callee's lineage, which is enough whenever the callee cell actually runs "
        "(Run All, or a restart). Re-running ONLY the call site does not re-execute "
        "the edited callee, so user_ns and variable_lineage still hold the OLD b "
        "and the key legitimately does not change. Closing it means putting the "
        "callee into the statement's INPUTS so upstream reconstruction re-executes "
        "its cell — inputs also drive cacheability and mutation analysis, which is "
        "why it is not a one-line follow-on. strict=True: global xfail_strict is OFF."
    ),
)
def test_callee_below_caller_propagates_an_edit_on_isolated_rerun(nb_runner):
    """Callee BELOW: editing it and re-running ONLY the call site."""
    nb_runner.create_notebook([C_ON, CALLER, CALLEE, CALLSITE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "r=8" in nb_runner.get_output(4)

    nb_runner.set_cell_source(3, "def b(n):\n    return n + 10")
    nb_runner.run_cell(4)
    assert "r=26" in nb_runner.get_output(4), (
        f"callee below the caller must propagate: {nb_runner.get_output(4)!r}"
    )


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
