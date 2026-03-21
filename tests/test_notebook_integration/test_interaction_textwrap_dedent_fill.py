"""Batch 411: textwrap module dedent and fill."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTextwrapDedentFill:
    def test_dedent(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\nraw = '    hello\\n    world\\n    foo'",
            "cleaned = textwrap.dedent(raw)\nlines = cleaned.strip().split('\\n')\nprint(f'lines={lines}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lines=['hello', 'world', 'foo']" in nb_runner.get_output(2)

    def test_fill(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\ntext = 'The quick brown fox jumps over the lazy dog and runs away'",
            "wrapped = textwrap.fill(text, width=20)\nline_count = len(wrapped.split('\\n'))\nprint(f'line_count={line_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        count = int(nb_runner.get_output(2).split("line_count=")[1].strip())
        assert count >= 3

    def test_textwrap_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\ntext = 'hello world test'",
            "shortened = textwrap.shorten(text, width=12, placeholder='...')\nprint(f'short={shortened}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "short=hello..." in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import textwrap\ntext = 'foo bar baz qux'")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "short=foo bar..." in out
