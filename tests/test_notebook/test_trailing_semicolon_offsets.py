"""``_expr_has_trailing_semicolon`` must locate the ``;`` the way the parser
locates the node it follows.

CAS-96 recovers a trailing ``;`` (IPython display suppression) from the raw
cell, because ``ast.unparse`` drops it. It did so by indexing
``raw_cell.splitlines()`` with ``node.end_lineno`` and then slicing that line
with ``node.end_col_offset``. Two independent mismatches with the parser's own
coordinates made it silently answer ``False`` — dropping the suppression and
echoing a repr the user asked it not to:

* ``str.splitlines()`` breaks on characters the CPython tokenizer does not
  (vertical tab, form feed, NEL, ...), so one of them anywhere earlier in the
  cell shifts every later line index.
* ``end_col_offset`` is a UTF-8 *byte* offset, but was used as a character
  index, so any non-ASCII character earlier on the same line slid the slice
  past the ``;``.

Both are silent: no exception, so nothing falls back. The oracle here is
``ast.get_source_segment`` — whatever text really follows the node — which
makes these tests a property of the function rather than a pin on six inputs.
"""
import ast

import pytest

from cash.notebook.ipython.cell_executor import CellExecutor

VT = chr(0x0B)  # vertical tab
FF = chr(0x0C)  # form feed
NEL = chr(0x85)  # next line


def semicolon_follows(source: str, node: ast.stmt) -> bool:
    """What the answer must be, derived from the parser's own segment."""
    segment = ast.get_source_segment(source, node, padded=True)
    assert segment is not None, "oracle needs a locatable node"
    tail = source[source.index(segment) + len(segment):]
    return tail.lstrip().startswith(";")


@pytest.mark.parametrize(
    "label, source",
    [
        # Controls: these already passed, and must keep passing.
        ("plain", "'hello';\n"),
        ("plain, no semicolon", "'hello'\n"),
        ("second statement", "x = 1\n'hello';\n"),
        # Line-index desync: a parser-incompatible break earlier in the cell.
        ("vertical tab earlier", f"x = 'A{VT}B'\n'hello';\n"),
        ("form feed earlier", f"x = 1{FF}\n'hello';\n"),
        ("NEL earlier", f"x = '{NEL}'\n'hello';\n"),
        # Byte-vs-character offset: non-ASCII earlier on the SAME line.
        ("non-ascii, semicolon", "'héllo';\n"),
        ("non-ascii, no semicolon", "'héllo'\n"),
        ("emoji, semicolon", "'\U0001f600';\n"),
        # Both mismatches at once.
        ("vertical tab AND non-ascii", f"x = 'A{VT}B'\n'héllo';\n"),
    ],
)
def test_matches_the_parsers_own_coordinates(label, source):
    node = ast.parse(source).body[-1]
    assert CellExecutor._expr_has_trailing_semicolon(
        source, node
    ) == semicolon_follows(source, node), label


def test_a_non_expression_statement_is_never_suppressed():
    """The ``isinstance(node, ast.Expr)`` guard is load-bearing: an assignment
    followed by ``;`` has no repr to suppress."""
    source = "x = 1;\n"
    node = ast.parse(source).body[0]
    assert CellExecutor._expr_has_trailing_semicolon(source, node) is False
