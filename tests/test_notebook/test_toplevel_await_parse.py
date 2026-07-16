"""CAS-164: cash's cell-level parse must tolerate a top-level ``await``.

A module-level ``await`` is a SyntaxError to plain ``ast.parse`` (it needs
``PyCF_ALLOW_TOP_LEVEL_AWAIT``), so an async cell under ``%cash_on`` used to
raise inside ``analyze_code_block`` and be silently skipped — the coroutine
never ran, output was swallowed, no error, no badge. This pins the parse point
directly; the end-to-end no-op only reproduces against a LIVE Jupyter server,
which the nbclient unit harness lacks (CAS-136).
"""
import pytest

from cash.notebook.analysis import CodeAnalyzer


def test_parse_cell_tolerates_top_level_await():
    """The fix: the cell parser accepts a module-level await (no SyntaxError)."""
    tree = CodeAnalyzer._parse_cell("result = await fetch(url)")
    assert tree is not None


def test_analyze_code_block_handles_await_cell():
    """Before the fix this raised ``SyntaxError: 'await' outside function``."""
    inputs, outputs = CodeAnalyzer.analyze_code_block(
        "import asyncio\nresult = await fetch(url)\nprint(result)"
    )
    assert "result" in outputs, outputs


def test_strip_magics_survives_await_cell():
    """An await cell has no magics and must pass through strip_magics intact."""
    code = "x = 1\nresult = await fetch(x)"
    assert CodeAnalyzer.strip_magics(code) == code


def test_genuine_syntax_error_still_raises():
    """A real typo is still a SyntaxError (CAS-156 clean-traceback path)."""
    with pytest.raises(SyntaxError):
        CodeAnalyzer._parse_cell("x = (1 + ")
