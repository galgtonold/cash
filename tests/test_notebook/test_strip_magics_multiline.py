"""CAS-163: ``strip_magics`` must not shred a multi-line statement whose
continuation line begins with ``%`` (modulo / ``%``-format) or ``!``.

The magic filter used to test *every* physical line and drop any whose stripped
form started with ``%``/``!``.  A valid multi-line statement such as::

    print("Asian call = %.4f\\n"
          "European   = %.4f"
          % (a, b))

has a continuation line (``      % (a, b))``) that starts with ``%``.  The old
filter deleted it, turning valid Python into ``'(' was never closed`` — a
*fictional* SyntaxError.  Inside the notebook simulator that aborted the
upstream replay and silently disabled cache restore for the cell.

These are fast, direct tests of the splitter: feed it the source and assert the
result is ONE statement that parses, not a fragment that raises.
"""
import ast

from cash.notebook.analysis import CodeAnalyzer


# A literal backslash-n INSIDE the string (built with chr to avoid any
# ambiguity between an escape and a real newline in the test source itself).
_BS_N = chr(92) + "n"

# Form that actually triggered the bug: the ``%`` operator begins a
# continuation line of a parenthesised, implicitly-concatenated string.
MULTILINE_PCT_OP_AT_LINE_START = (
    'print("Asian call = %.4f' + _BS_N + '"\n'
    '      "European   = %.4f"\n'
    '      % (a, b))'
)

# The headline two-line form from the report (``%`` mid-line): valid Python
# that must also survive untouched.
MULTILINE_PCT_MID_LINE = (
    'print("Asian call = %.4f' + _BS_N + '"\n'
    '      "European   = %.4f" % (a, b))'
)


def _parse_single_call(cleaned: str) -> ast.AST:
    """Assert *cleaned* parses to exactly one top-level ``print(...)`` call."""
    tree = ast.parse(cleaned)  # must NOT raise
    assert len(tree.body) == 1, f"expected one statement, got {len(tree.body)}"
    node = tree.body[0]
    assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Call), (
        f"expected a bare call expression, got {ast.dump(node)}"
    )
    return node


def test_pct_operator_at_continuation_line_start_survives():
    """The regression trigger: a ``%`` opening a continuation line is preserved."""
    cleaned = CodeAnalyzer.strip_magics(MULTILINE_PCT_OP_AT_LINE_START)
    # Before the fix this raised: ``'(' was never closed``.
    _parse_single_call(cleaned)
    # The format operator must still be present (line was NOT deleted).
    assert "% (a, b)" in cleaned


def test_pct_mid_line_multiline_print_survives():
    """The two-line headline form (``%`` mid-line) is likewise untouched."""
    cleaned = CodeAnalyzer.strip_magics(MULTILINE_PCT_MID_LINE)
    _parse_single_call(cleaned)
    assert cleaned == MULTILINE_PCT_MID_LINE  # byte-identical: nothing stripped


def test_real_magic_still_stripped_alongside_multiline_pct():
    """A genuine cell magic is still removed even when the same cell also holds
    a multi-line ``%``-statement whose continuation starts with ``%``."""
    code = (
        "loop_result = sum(range(3))\n"
        "%matplotlib inline\n"
        + MULTILINE_PCT_OP_AT_LINE_START
    )
    cleaned = CodeAnalyzer.strip_magics(code)
    ast.parse(cleaned)  # must not raise
    assert "%matplotlib inline" not in cleaned      # magic dropped
    assert "% (a, b)" in cleaned                     # operator preserved
    assert "loop_result = sum(range(3))" in cleaned


def test_bare_line_magic_and_shell_escape_still_stripped():
    """Plain line magics / shell escapes at logical-line start still go."""
    code = "%time f()\nx = 1\n!ls -la\ny = 2"
    cleaned = CodeAnalyzer.strip_magics(code)
    assert cleaned == "x = 1\ny = 2"
    ast.parse(cleaned)


def test_pct_inside_triple_quoted_string_is_not_a_magic():
    """A ``%`` opening a line INSIDE a triple-quoted string is data, not a magic."""
    code = 'doc = """\n% not a magic — just text\n"""\nz = 3'
    cleaned = CodeAnalyzer.strip_magics(code)
    assert "% not a magic" in cleaned
    ast.parse(cleaned)


def test_genuine_syntax_error_still_degrades():
    """A real typo must remain a SyntaxError (graceful degrade, no CAS-156 regression)."""
    code = "x = = 1"
    cleaned = CodeAnalyzer.strip_magics(code)
    try:
        ast.parse(cleaned)
        raised = False
    except SyntaxError:
        raised = True
    assert raised, "a genuine typo must still fail to parse"


def test_no_magic_cell_returned_unchanged():
    """Ordinary multi-statement cells are passed through byte-for-byte."""
    code = "a = 1\nb = 2\nc = a + b"
    assert CodeAnalyzer.strip_magics(code) == code


def test_indented_magic_does_not_empty_its_block():
    """A magic that is the only statement in a block must not be *deleted*.

    ``if IN_COLAB:`` with a lone ``%pip install`` under it: deleting the magic
    line leaves ``if IN_COLAB:`` followed by a dedent — 'expected an indented
    block', a fictional SyntaxError. cash then flags the cell as broken and stops
    dependency-tracking everything that reads from it. The magic is replaced by
    ``pass`` so the suite still has a body.
    """
    code = (
        "if IN_COLAB:\n"
        "    %pip install -q cash-lib\n"
        "\n"
        "import cash"
    )
    cleaned = CodeAnalyzer.strip_magics(code)
    ast.parse(cleaned)                       # must NOT raise
    assert "%pip" not in cleaned             # the magic itself is gone
    assert "    pass" in cleaned             # ...replaced by pass at its indent
    assert "import cash" in cleaned


def test_magic_as_sole_function_body_survives():
    """The same guarantee inside a ``def`` (another empty-suite trap)."""
    cleaned = CodeAnalyzer.strip_magics("def setup():\n    %matplotlib inline")
    ast.parse(cleaned)
    assert "def setup():" in cleaned and "pass" in cleaned
