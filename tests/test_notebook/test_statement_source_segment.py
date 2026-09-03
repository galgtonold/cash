"""The badge shows a statement the way it was written, not the way it is keyed.

``ast.unparse`` normalizes a statement onto one logical line -- it is what the
cache key needs and what gets compiled. For the badge we want the user's own
text back, which is what ``ast.get_source_segment`` returns.
"""
from __future__ import annotations

import ast

from cash.notebook.ipython.cell_executor import _statement_source


def test_a_top_level_statement_comes_back_verbatim():
    cell = (
        'category_stats = (\n'
        '    transactions\n'
        '    .groupby("category")\n'
        '    .sort_values("total_spend", ascending=False)\n'
        ')\n'
    )
    node = ast.parse(cell).body[0]
    assert _statement_source(cell, node) == cell.rstrip("\n")


def test_a_nested_statement_is_dedented_relative_to_itself():
    """``get_source_segment`` returns the first line flush and the rest at their
    ABSOLUTE file indentation, which reads as ragged in a badge row.

    ``textwrap.dedent`` cannot fix it: the first line shares no common prefix
    with the others.
    """
    cell = (
        'for k in keys:\n'
        '    total = (\n'
        '        df[df.k == k]\n'
        '        .amount.sum()\n'
        '    )\n'
    )
    inner = ast.parse(cell).body[0].body[0]

    raw = ast.get_source_segment(cell, inner)
    assert raw.splitlines()[1].startswith("        "), "premise: raw is ragged"

    assert _statement_source(cell, inner) == (
        'total = (\n'
        '    df[df.k == k]\n'
        '    .amount.sum()\n'
        ')'
    )


def test_a_single_line_statement_is_unchanged():
    cell = "x = 1\n"
    node = ast.parse(cell).body[0]
    assert _statement_source(cell, node) == "x = 1"


def test_an_unrecoverable_segment_returns_none():
    """A node with no position info cannot be located in the source.

    The caller falls back to the unparsed form, so this must not raise.
    """
    node = ast.parse("x = 1").body[0]
    del node.lineno
    assert _statement_source("x = 1\n", node) is None
