"""The rewrite of a cell's calls works on a copy of its tree and leaves the
original as it was.

The runtime compiles the copy while the analysis and the cache key read the
original; a rewrite that touched the original would key one source and run
another. The copy shares the constants of a long literal with the original
rather than copying each one.
"""

from __future__ import annotations

import ast

from cash.notebook.call_interception import wrap_eligible_calls

SOURCE = (
    "data = ["
    + ", ".join(f"({i}, 'x{i}')" for i in range(500))
    + "]\nrows = [score(d) for d in data]\ntotal = sum(rows) + score(1)\n"
)


def test_the_original_tree_is_left_unchanged():
    tree = ast.parse(SOURCE)
    before = ast.dump(tree, include_attributes=True)

    rewritten, sites = wrap_eligible_calls(tree)

    assert sites, "the cell has calls to wrap"
    assert ast.dump(tree, include_attributes=True) == before
    assert "__cash_call__" in ast.unparse(rewritten)
    assert "__cash_call__" not in ast.unparse(tree)


def test_a_cell_with_nothing_to_wrap_comes_back_equal_and_as_a_new_tree():
    tree = ast.parse("a = [(1, 'x'), (2, 'y')]\nb = a[0]\n")

    rewritten, sites = wrap_eligible_calls(tree)

    assert sites == []
    assert rewritten is not tree and rewritten.body[0] is not tree.body[0]
    assert ast.dump(rewritten, include_attributes=True) == ast.dump(tree, include_attributes=True)


def test_the_rewritten_tree_compiles_and_runs_as_the_original_would():
    rewritten, _sites = wrap_eligible_calls(ast.parse(SOURCE))
    namespace = {"score": lambda d: 1, "__cash_call__": lambda fn, index: fn}

    exec(compile(rewritten, "<cell>", "exec"), namespace)

    assert namespace["total"] == 501
