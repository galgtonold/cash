"""In-place mutation of a function-call ARGUMENT must not accumulate on an
isolated re-run (CAS-58).

A user-defined helper that mutates its parameter in place (``def add(x):
x.append(1)``) called as a bare statement (``add(data)``) mutates the caller's
variable, but the cell never names the mutation. cash resolves the helper's
source from the notebook cells, statically detects which parameters it mutates
(one level deep), maps them back to the call arguments, and resets those vars on
an isolated re-run. A pure helper passed the same variable is NOT a mutation and
must keep its cache (no over-invalidation).

Distinct from the CAS-49 hidden-state limitations (globals / default args /
generators) where the mutated state is not visible in the cell.
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


def test_list_append_via_helper(nb_runner):
    _rerun(nb_runner, "data = [1, 2, 3]\ndef append_one(x):\n    x.append(99)",
           "append_one(data)\nprint(len(data))", "4")


def test_dict_mutation_via_helper(nb_runner):
    _rerun(nb_runner, "d = {'n': 0}\ndef bump(x):\n    x['n'] += 1",
           "bump(d)\nprint(d['n'])", "1")


def test_df_row_via_helper(nb_runner):
    _rerun(nb_runner,
           "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\n"
           "def add_row(d):\n    d.loc[len(d)] = [99]",
           "add_row(df)\nprint(len(df))", "4")


def test_keyword_arg_mutation(nb_runner):
    _rerun(nb_runner, "data = [1, 2, 3]\ndef fill(*, target):\n    target.append(0)",
           "fill(target=data)\nprint(len(data))", "4")


def test_only_mutated_arg_reset(nb_runner):
    """two(a, b) mutates only `a`; `b` (a pure read) keeps working."""
    _rerun(nb_runner,
           "rows = [1, 2, 3]\nkeep = [9]\ndef two(a, b):\n    a.append(b[0])",
           "two(rows, keep)\nprint(len(rows), len(keep))", "4 1")


def test_del_via_helper(nb_runner):
    _rerun(nb_runner, "d = {'a': 1, 'b': 2}\ndef drop_b(x):\n    del x['b']",
           "drop_b(d)\nprint(sorted(d))", "['a']")


def test_pure_helper_not_over_invalidated(nb_runner):
    """A pure helper passed a var must NOT be treated as a mutation."""
    _rerun(nb_runner, "data = [1, 2, 3]\ndef peek(x):\n    return len(x)",
           "peek(data)\nprint(len(data))", "3")


def test_reassigning_helper_not_a_mutation(nb_runner):
    """A helper that rebinds its param to a new local object does not mutate the
    caller -> not flagged, keeps cache (no over-invalidation)."""
    _rerun(nb_runner, "data = [1, 2, 3]\ndef grow(x):\n    x = x + [0]\n    return x",
           "grow(data)\nprint(len(data))", "3")


def test_depth2_mutation_via_nested_helper(nb_runner):
    """outer(y) mutates y only by calling inner(y); the interprocedural analysis
    propagates the mutation back through the nested call (CAS-61)."""
    _rerun(nb_runner,
           "data = [1]\ndef inner(z):\n    z.append(9)\ndef outer(y):\n    inner(y)",
           "outer(data)\nprint(data)", "[1, 9]")


def test_depth3_mutation_via_nested_helpers(nb_runner):
    _rerun(nb_runner,
           "data = [1]\ndef inner(z):\n    z.append(9)\ndef mid(b):\n    inner(b)\ndef outer(y):\n    mid(y)",
           "outer(data)\nprint(data)", "[1, 9]")


def test_depth2_pure_chain_not_over_invalidated(nb_runner):
    """A nested chain that never mutates (only returns) must keep its cache."""
    _rerun(nb_runner,
           "data = [1, 2, 3]\ndef inner_pure(z):\n    return len(z)\ndef outer_pure(y):\n    return inner_pure(y)",
           "outer_pure(data)\nprint(len(data))", "3")
