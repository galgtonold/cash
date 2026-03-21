"""Batch 360: textwrap, dedent, and multi-line string formatting."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTextwrapDedent:
    def test_wrap_basic(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\nlong_text = 'The quick brown fox jumps over the lazy dog and then runs away'",
            "wrapped = textwrap.fill(long_text, width=30)\nlines = wrapped.count('\\n') + 1\nprint(f'lines={lines}')\nprint(f'wrapped={repr(wrapped)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lines=" in out

    def test_dedent_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\ntext = '    line1\\n    line2\\n    line3'",
            "dedented = textwrap.dedent(text)\nfirst = dedented.split('\\n')[0]\nprint(f'first={first}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=line1" in nb_runner.get_output(2)
        # Edit text
        nb_runner.set_cell_source(1, "import textwrap\ntext = '        hello\\n        world'")
        nb_runner.run_all()
        assert "first=hello" in nb_runner.get_output(2)

    def test_indent(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\ntext = 'line1\\nline2\\nline3'",
            "indented = textwrap.indent(text, '>>> ')\nprint(f'indented={repr(indented)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert ">>> line1" in nb_runner.get_output(2)
