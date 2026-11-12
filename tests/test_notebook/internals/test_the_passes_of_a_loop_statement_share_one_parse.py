"""Each pass of a loop body statement is the statement's text under a comment
naming the pass, and each was parsed on its own: what cash keeps about a
tree (its walks, the calls it makes) was worked out again for every pass.
The passes now share one tree, which is the tree of each pass's own text,
positions included. Pins how the work is done: a refactor may expect this
to fail.
"""

from __future__ import annotations

import ast

from cash.analysis.ast_util import parse_cached
from cash.control_markers import mark_control, mark_iteration

BODY = "x = rows.append(v)\nprint(x)"


def test_two_passes_share_one_tree():
    first = parse_cached(mark_iteration(BODY, "a1b2c3"))
    second = parse_cached(mark_iteration(BODY, "d4e5f6"))
    assert first is second


def test_the_shared_tree_is_each_passs_own():
    for text in (mark_iteration(BODY, "d4e5f6"), mark_iteration(mark_control(BODY, "77"), "d4e5f6")):
        assert ast.dump(parse_cached(text), include_attributes=True) == ast.dump(
            ast.parse(text), include_attributes=True
        )


def test_a_marker_like_line_inside_the_statement_is_its_own():
    first = parse_cached(mark_iteration("s = '''\n# __iteration_context__: aa\n'''", "1"))
    second = parse_cached(mark_iteration("s = '''\n# __iteration_context__: bb\n'''", "1"))
    assert first is not second
    assert first.body[0].value.value != second.body[0].value.value
