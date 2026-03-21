"""
Batch 320: textwrap and string formatting patterns with caching.
Tests textwrap.dedent, textwrap.fill, indent, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestTextwrapFormat:
    """Test textwrap formatting operation caching."""

    def test_dedent_basic(self, nb_runner):
        """Dedent indented text, verify caching."""
        nb_runner.create_notebook([
            "import textwrap",
            "raw = '    line1\\n    line2\\n    line3'",
            "cleaned = textwrap.dedent(raw)\nlines = cleaned.strip().split('\\n')",
            "print(f'count={len(lines)} first={lines[0]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=3" in out
        assert "first=line1" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "count=3" in out2

    def test_fill_wrap_edit(self, nb_runner):
        """textwrap.fill with width edit."""
        nb_runner.create_notebook([
            "import textwrap",
            "text = 'The quick brown fox jumps over the lazy dog near the river bank'",
            "width = 20",
            "wrapped = textwrap.fill(text, width=width)\nline_count = len(wrapped.split('\\n'))",
            "print(f'lines={line_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        lines_narrow = int(out.split("lines=")[1].strip())

        nb_runner.set_cell_source(3, "width = 40")
        nb_runner.run_all()
        out2 = nb_runner.get_output(5)
        lines_wide = int(out2.split("lines=")[1].strip())
        assert lines_wide < lines_narrow

    def test_indent_pattern(self, nb_runner):
        """textwrap.indent with prefix."""
        nb_runner.create_notebook([
            "import textwrap",
            "text = 'line1\\nline2\\nline3'",
            "indented = textwrap.indent(text, '>>> ')\nfirst_line = indented.split('\\n')[0]",
            "print(f'first={first_line}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "first=>>> line1" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "first=>>> line1" in out2
