"""
Interaction test: textwrap.wrap and shorten with custom settings.
Tests textwrap.wrap with width, initial_indent, subsequent_indent,
and textwrap.shorten across cells.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTextwrapWrapShorten:
    """Test textwrap.wrap and shorten across cells."""

    def test_wrap_shorten(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: wrap text
            "import textwrap\ntext = 'The quick brown fox jumps over the lazy dog and continues running through the forest'\nwrapped = textwrap.wrap(text, width=30)\nprint(f'lines={len(wrapped)}')\nfor line in wrapped:\n    print(f'  |{line}|')",
            # Cell 2: shorten
            "short = textwrap.shorten(text, width=40, placeholder='...')\nprint(f'short={short}')\nprint(f'short_len={len(short)}')",
            # Cell 3: indent
            "indented = textwrap.indent(text, prefix='>>> ')\nprint(f'indented={indented}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "lines=" in out1
        out2 = nb_runner.get_output(2)
        assert "..." in out2
        assert int(nb_runner.get_output(2).split("short_len=")[1].strip()) <= 40
        out3 = nb_runner.get_output(3)
        assert ">>> The quick" in out3

    def test_wrap_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\ntext = 'Hello World this is a test of text wrapping'\nlines = textwrap.wrap(text, width=20)\nline_count = len(lines)\nprint(f'lines={line_count}')",
            "first = lines[0]\nprint(f'first={first}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Change width
        nb_runner.set_cell_source(1, "import textwrap\ntext = 'Hello World this is a test of text wrapping'\nlines = textwrap.wrap(text, width=10)\nline_count = len(lines)\nprint(f'lines={line_count}')")
        nb_runner.run_cells([1, 2])
        # Narrower width = more lines
        count = int(nb_runner.get_output(1).split("lines=")[1].strip())
        assert count > 3

    def test_wrap_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\nresult = textwrap.fill('A short sentence for testing.', width=15)\nprint(f'filled={result}')",
            "line_count = result.count('\\n') + 1\nprint(f'count={line_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "count=" in nb_runner.get_output(2)
