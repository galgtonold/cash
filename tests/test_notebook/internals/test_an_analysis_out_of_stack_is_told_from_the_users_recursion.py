"""The upstream check tells a RecursionError of cash's own walk of a syntax
tree from one the user's code raised: the first makes the cell run uncached,
the second is the cell's error."""

from __future__ import annotations

import ast
import sys

from cash.analysis.code_analyzer import CodeAnalyzer
from cash.notebook.ipython.upstream_phase import _analysis_ran_out_of_stack


def _deep_tree(terms: int = 3000) -> ast.Module:
    total: ast.expr = ast.Name(id="a", ctx=ast.Load())
    for _ in range(terms - 1):
        total = ast.BinOp(left=total, op=ast.Add(), right=ast.Name(id="a", ctx=ast.Load()))
    return ast.Module(
        body=[ast.Assign(targets=[ast.Name(id="x", ctx=ast.Store())], value=total, lineno=1)], type_ignores=[]
    )


def _caught(run) -> BaseException:
    try:
        run()
    except BaseException as error:  # the error is what is looked at
        return error
    raise AssertionError("nothing raised")


def test_running_out_of_stack_in_pythons_own_tree_walk_is_the_analysis():
    error = _caught(lambda: ast.unparse(_deep_tree()))
    assert isinstance(error, RecursionError)
    assert _analysis_ran_out_of_stack(error)


def test_running_out_of_stack_in_cashs_code_is_the_analysis():
    def walk(node):  # stands for a recursive visitor of cash's
        return CodeAnalyzer.strip_magics("x") and walk(node)

    walk.__code__ = walk.__code__.replace(co_filename=CodeAnalyzer.strip_magics.__code__.co_filename)
    assert _analysis_ran_out_of_stack(_caught(lambda: walk(None)))


def test_a_recursion_of_the_users_code_is_not():
    namespace: dict = {}
    exec(compile("def down(n):\n    return down(n + 1)", "<cell>", "exec"), namespace)
    error = _caught(lambda: namespace["down"](0))
    assert isinstance(error, RecursionError)
    assert not _analysis_ran_out_of_stack(error)


def test_another_error_is_not():
    assert not _analysis_ran_out_of_stack(_caught(lambda: 1 / 0))
    assert not _analysis_ran_out_of_stack(RecursionError("never raised"))


def test_the_statement_analysis_itself_walks_a_tree_this_deep():
    tree = _deep_tree(2000)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(250)
    try:
        inputs, outputs = CodeAnalyzer.analyze_code_block("", tree=tree)
    finally:
        sys.setrecursionlimit(limit)
    assert (inputs, outputs) == ({"a"}, {"x"})
