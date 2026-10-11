"""Indirect-mutation channels on an isolated re-run.

An object reachable from an upstream variable is mutated in the cell through
another name. The statement that made the two names share the object links
them, so the upstream holder is reset and the value does not double on
re-run: an attribute store (``b.ref = x``), a tuple holding a list, a
conditional alias. The walrus receiver is still a tracked limitation; its
xfail flips to XPASS when its channel is fixed.
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


def test_alias_via_attribute(nb_runner):
    _rerun(
        nb_runner,
        "class Box:\n    pass\nb = Box()\nx = [1, 2, 3]",
        "b.ref = x\nb.ref.append(99)\nprint(x)",
        "[1, 2, 3, 99]",
    )


def test_tuple_holds_mutable(nb_runner):
    _rerun(nb_runner, "lst = [1, 2]", "t = (lst,)\nt[0].append(3)\nprint(lst)", "[1, 2, 3]")


@pytest.mark.xfail(reason="walrus-as-method-receiver not attributed")
def test_walrus_alias_mutate(nb_runner):
    # (y := x).append(..) — the NamedExpr receiver is not surfaced as a mutated
    # name, so the alias y->x is never resolved.
    _rerun(nb_runner, "x = [1, 2]", "(y := x).append(3)\nprint(x)", "[1, 2, 3]")


def test_conditional_alias(nb_runner):
    _rerun(nb_runner, "x = [1, 2]\nz = [9]", "y = x if True else z\ny.append(3)\nprint(x)", "[1, 2, 3]")
