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
    """A node checked against a mismatched source can't be located in it.

    ``get_source_segment`` indexes into the source's lines by the node's
    ``lineno``. A node parsed from a three-line source, then looked up
    against a shorter, mismatched ``raw_cell``, pushes that index past the
    end of the line list -- ``get_source_segment`` raises ``IndexError``,
    which is what exercises our own ``except`` clause.

    (A node with its ``lineno`` deleted entirely takes a different path:
    ``get_source_segment`` catches ``AttributeError`` internally and returns
    ``None`` without ever raising, so a test built on that would still pass
    with our ``except`` deleted -- it would not prove this function's guard
    does anything.)

    The caller falls back to the unparsed form, so this must not raise.
    """
    cell = "x = 1\ny = 2\nz = 3\n"
    node = ast.parse(cell).body[2]
    assert _statement_source("x = 1\n", node) is None


def test_a_top_level_function_definition_returns_none():
    """A ``def`` only BINDS the name -- the body never runs, so the body is
    not "the code that ran" the way it is for every other captured statement.
    Showing it in full would also make the badge very tall in any notebook
    that defines functions, so capture is withheld and the caller falls back
    to the unparsed form (today's clipped ``def foo(x):`` + "... +N lines").
    """
    cell = "def foo(x):\n    y = x + 1\n    return y\n"
    node = ast.parse(cell).body[0]
    assert _statement_source(cell, node) is None


def test_a_top_level_async_function_definition_returns_none():
    """``AsyncFunctionDef`` is a distinct node type from ``FunctionDef`` --
    checked separately so an ``async def`` doesn't slip through the guard."""
    cell = "async def foo(x):\n    y = x + 1\n    return y\n"
    node = ast.parse(cell).body[0]
    assert _statement_source(cell, node) is None


def test_a_top_level_class_definition_returns_none():
    cell = "class Foo:\n    x = 1\n"
    node = ast.parse(cell).body[0]
    assert _statement_source(cell, node) is None


def test_a_top_level_match_statement_is_captured_verbatim():
    """The def/class exclusion is about BINDING vs. EXECUTING, not about
    "is it multi-line". A ``match`` is just as multi-line as a def/class,
    but it genuinely executes its matched branch, and (unlike an
    if/for/while/with/try) it is not a control structure the runtime
    splits into per-branch rows -- it is cached and executed as ONE unit,
    so its full source IS "the code that ran". It must NOT be excluded the
    way def/class are.
    """
    cell = (
        'match command:\n'
        '    case "go":\n'
        '        result = 1\n'
        '    case _:\n'
        '        result = 0\n'
    )
    node = ast.parse(cell).body[0]
    assert _statement_source(cell, node) == cell.rstrip("\n")


def test_a_vertical_tab_inside_a_string_literal_is_not_truncated():
    """The single-line fast path used to slice raw_cell.splitlines(keepends=
    True) directly. str.splitlines() treats a vertical tab (U+000B) as a
    line break; the CPython parser does not -- it recognizes only CRLF, CR
    and LF. So this whole assignment is still ONE line to ast (lineno ==
    end_lineno == 1). Splitting on the vertical tab desyncs the fast path's
    line index from that numbering: lines[0] becomes only the fragment up
    to it, so the slice silently drops everything after (here, the closing
    B') instead of raising.

    Built with chr() rather than an escape literal so the exact code point
    is unambiguous on the page.
    """
    vertical_tab = chr(0x0B)
    cell = f"x = 'A{vertical_tab}B'\n"
    node = ast.parse(cell).body[0]
    assert _statement_source(cell, node) == ast.get_source_segment(cell, node)


def test_a_form_feed_does_not_corrupt_the_following_statement():
    """A form feed (U+000C) the parser folds into the middle of line 1
    still reads as a line break to str.splitlines(), so that split produces
    an extra list entry the parser's own numbering has no room for. Every
    lines[node.lineno - 1] lookup for a LATER statement is then off by one
    -- so the corruption lands on the statement AFTER the one containing
    the form feed, not on that statement itself.
    """
    form_feed = chr(0x0C)
    cell = f"x = 1{form_feed}\ny = 2\n"
    tree = ast.parse(cell)
    assert len(tree.body) == 2, "premise: two top-level statements"
    assert [_statement_source(cell, node) for node in tree.body] == [
        ast.get_source_segment(cell, node) for node in tree.body
    ]


def test_the_fast_path_agrees_with_get_source_segment_for_every_parser_incompatible_linebreak():
    """The property that matters is agreement with the slow path on every
    input, not the exact output for the two reproductions above.
    str.splitlines() breaks on eight characters the parser does not:
    vertical tab (U+000B), form feed (U+000C), FILE/GROUP/RECORD SEPARATOR
    (U+001C-U+001E), NEL (U+0085), LINE SEPARATOR (U+2028) and PARAGRAPH
    SEPARATOR (U+2029). This cell puts one of each inside its own
    single-line statement, so a fix that only special-cases the two
    reproduced characters (rather than fixing the general splitting rule)
    would still fail here -- and fail on every statement after the first
    offender, not just the one that contains it.

    Built with chr() rather than escape literals so the exact code points
    are unambiguous on the page.
    """
    offenders = "".join(
        chr(c) for c in (0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029)
    )
    cell = "\n".join(
        f"x{i} = 'A{ch}B'" for i, ch in enumerate(offenders)
    ) + "\n"
    tree = ast.parse(cell)
    assert len(tree.body) == len(offenders), "premise: one statement per offender"
    for node in tree.body:
        assert _statement_source(cell, node) == ast.get_source_segment(cell, node)
