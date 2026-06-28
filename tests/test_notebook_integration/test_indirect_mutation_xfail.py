"""Known indirect-mutation channels not yet reset on isolated re-run (CAS-61).

Same root family as CAS-58/CAS-60: an object reachable from an upstream variable
is mutated in the cell, but the mutation is attributed to a different name, so
the upstream holder is never reset and the value doubles on re-run. CAS-60 fixed
the bare `Name = Name` alias channel (incl. DataFrame aliases). These four remain
as tracked limitations; each xfail flips to XPASS when its channel is fixed.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


@pytest.mark.xfail(reason="CAS-61: attribute-store alias not tracked", strict=False)
def test_alias_via_attribute(nb_runner):
    _rerun(nb_runner,
           "class Box:\n    pass\nb = Box()\nx = [1, 2, 3]",
           "b.ref = x\nb.ref.append(99)\nprint(x)", "[1, 2, 3, 99]")


@pytest.mark.xfail(reason="CAS-61: container-element aliasing not tracked", strict=False)
def test_tuple_holds_mutable(nb_runner):
    _rerun(nb_runner, "lst = [1, 2]",
           "t = (lst,)\nt[0].append(3)\nprint(lst)", "[1, 2, 3]")


@pytest.mark.xfail(reason="CAS-61: alias bound inside loop body not scanned", strict=False)
def test_alias_in_for_loop(nb_runner):
    _rerun(nb_runner, "x = [1, 2, 3]",
           "for _ in range(1):\n    y = x\n    y.append(99)\nprint(x)", "[1, 2, 3, 99]")


@pytest.mark.xfail(reason="CAS-61: depth-2 function-arg mutation not propagated", strict=False)
def test_nested_function_depth2_arg(nb_runner):
    _rerun(nb_runner,
           "def inner(z):\n    z.append(9)\ndef outer(y):\n    inner(y)\ndata = [1]",
           "outer(data)\nprint(data)", "[1, 9]")


@pytest.mark.xfail(reason="CAS-61: walrus-as-method-receiver not attributed", strict=False)
def test_walrus_alias_mutate(nb_runner):
    # (y := x).append(..) — the NamedExpr receiver is not surfaced as a mutated
    # name, so the alias y->x is never resolved.
    _rerun(nb_runner, "x = [1, 2]", "(y := x).append(3)\nprint(x)", "[1, 2, 3]")


@pytest.mark.xfail(reason="CAS-61: ternary alias is flow-sensitive (two sources)", strict=False)
def test_conditional_alias(nb_runner):
    _rerun(nb_runner, "x = [1, 2]\nz = [9]",
           "y = x if True else z\ny.append(3)\nprint(x)", "[1, 2, 3]")
