"""
Unit tests for control structure body statement extraction and badge rendering.
"""
import ast

from cash.notebook.badge_renderer import render_control_body_html


class TestRenderControlBodyHtml:
    """Tests for the render_control_body_html helper."""

    def test_empty_list(self):
        assert render_control_body_html([]) == ""

    def test_if_else_body(self):
        stmts = [
            "if x > 0:",
            "  y = x * 2",
            "  z = y + 1",
            "else:",
            "  y = 0",
            "  z = -1",
        ]
        html = render_control_body_html(stmts)
        assert "<details" in html
        assert "<summary" in html
        assert "if x &gt; 0:" in html  # header with HTML escaping
        assert "y = x * 2" in html
        assert "z = y + 1" in html
        assert "else:" in html
        assert "y = 0" in html

    def test_single_statement(self):
        stmts = ["if True:"]
        html = render_control_body_html(stmts)
        assert "<details" in html
        assert "if True:" in html

    def test_html_escaping(self):
        stmts = [
            "if a < b:",
            "  c = a & b",
        ]
        html = render_control_body_html(stmts)
        assert "&lt;" in html  # < is escaped in header
        assert "&amp;" in html  # & is escaped in body

    def test_long_header_truncation(self):
        long_cond = "if " + "x" * 100 + ":"
        stmts = [long_cond, "  pass"]
        html = render_control_body_html(stmts)
        assert "…" in html  # truncation marker


class TestExtractBodyStatements:
    """Tests for control_structure_helpers.extract_body_statements."""

    def _parse_and_extract(self, code):
        """Helper to parse code and extract body statements."""
        from cash.notebook.control_structure_helpers import extract_body_statements
        node = ast.parse(code).body[0]
        return extract_body_statements(node)

    def test_if_else(self):
        code = "if x > 0:\n    y = 1\n    z = 2\nelse:\n    y = -1"
        stmts = self._parse_and_extract(code)
        assert stmts[0].startswith("if ")
        assert any("y = 1" in s for s in stmts)
        assert any("z = 2" in s for s in stmts)
        assert any("else:" in s for s in stmts)
        assert any("y = -1" in s or "y = (-1)" in s or "y = -1" in s for s in stmts)

    def test_if_elif_else(self):
        code = "if x > 0:\n    y = 1\nelif x == 0:\n    y = 0\nelse:\n    y = -1"
        stmts = self._parse_and_extract(code)
        assert any("if " in s for s in stmts)
        assert any("elif " in s for s in stmts)
        assert any("else:" in s for s in stmts)

    def test_while(self):
        code = "while i < 10:\n    i += 1\n    total += i"
        stmts = self._parse_and_extract(code)
        assert stmts[0].startswith("while ")
        assert any("i += 1" in s for s in stmts)

    def test_with(self):
        code = "with open('f') as fp:\n    data = fp.read()"
        stmts = self._parse_and_extract(code)
        assert stmts[0].startswith("with ")
        assert any("fp.read()" in s for s in stmts)

    def test_try_except(self):
        code = "try:\n    result = compute()\nexcept ValueError as e:\n    result = default"
        stmts = self._parse_and_extract(code)
        assert any("try:" in s for s in stmts)
        assert any("compute()" in s for s in stmts)
        assert any("except" in s and "ValueError" in s for s in stmts)

    def test_try_except_finally(self):
        code = "try:\n    f = open('x')\nexcept IOError:\n    pass\nfinally:\n    cleanup()"
        stmts = self._parse_and_extract(code)
        assert any("try:" in s for s in stmts)
        assert any("finally:" in s for s in stmts)
        assert any("cleanup()" in s for s in stmts)

    def test_non_control_returns_empty(self):
        """Non-control structure returns empty list."""
        from cash.notebook.control_structure_helpers import extract_body_statements
        node = ast.parse("x = 1").body[0]
        assert extract_body_statements(node) == []
