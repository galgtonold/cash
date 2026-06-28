"""Mutating an upstream object through a bare ``y = x`` alias must reset the
upstream holder on an isolated re-run, not accumulate (CAS-60).

`y = x` shares x's object, so `y.append(99)` also mutates x. The mutation was
attributed to the in-cell alias `y` (which has no producer to restore from), so
x was never marked broken and the shared object kept the mutation — an isolated
re-run doubled it (`[1, 2, 3, 99]` -> `[1, 2, 3, 99, 99]`). The fix resolves the
mutated alias back through the (transitive) alias map and marks the source x
mutated, so x's producer restores its cell-entry base. Copies (`y = x.copy()`,
`y = x[:]`) are not aliases and keep their cache.
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


def test_alias_append(nb_runner):
    _rerun(nb_runner, "x = [1, 2, 3]", "y = x\ny.append(99)\nprint(x)", "[1, 2, 3, 99]")


def test_alias_aug_assign_subscript(nb_runner):
    _rerun(nb_runner, "x = [10, 20]", "y = x\ny[0] += 5\nprint(x)", "[15, 20]")


def test_alias_multiple_mutations(nb_runner):
    _rerun(nb_runner, "data = [1]", "ref = data\nref.append(2)\nref.append(3)\nprint(data)", "[1, 2, 3]")


def test_alias_chain(nb_runner):
    _rerun(nb_runner, "x = [1, 2]", "y = x\nz = y\nz.append(3)\nprint(x)", "[1, 2, 3]")


def test_alias_set_mutation(nb_runner):
    _rerun(nb_runner, "s = {1, 2}", "t = s\nt.add(3)\nprint(sorted(s))", "[1, 2, 3]")


def test_direct_mutation_control(nb_runner):
    # Control: direct in-place mutation (no alias) already reset correctly.
    _rerun(nb_runner, "x = [1, 2, 3]", "x.append(99)\nprint(x)", "[1, 2, 3, 99]")


def test_copy_is_not_alias_preserved(nb_runner):
    # A real copy is independent; mutating it must NOT reset the source, and the
    # copy itself is idempotent across re-runs.
    _rerun(nb_runner, "x = [1, 2, 3]", "y = x.copy()\ny.append(99)\nprint(y, x)",
           "[1, 2, 3, 99] [1, 2, 3]")
