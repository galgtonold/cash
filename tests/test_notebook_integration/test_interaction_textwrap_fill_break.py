"""
Interaction test: textwrap fill and shorten with break_on_hyphens.
Tests textwrap.fill with break_long_words, break_on_hyphens,
shorten with placeholder, and cross-cell text pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTextwrapFillBreak:
    """Test textwrap fill with break options across cells."""

    def test_textwrap_fill_break(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: fill with break options
            "import textwrap\ntext = 'This is a very-long-hyphenated-word-that-should-break and some more text after it'\nfilled_break = textwrap.fill(text, width=30, break_on_hyphens=True)\nfilled_no_break = textwrap.fill(text, width=30, break_on_hyphens=False)\nbreak_lines = len(filled_break.split('\\n'))\nno_break_lines = len(filled_no_break.split('\\n'))\nprint(f'break_lines={break_lines}')\nprint(f'no_break_lines={no_break_lines}')",
            # Cell 2: shorten
            "long = 'The quick brown fox jumps over the lazy dog near the river'\nshort = textwrap.shorten(long, width=30, placeholder='...')\nprint(f'shortened={short}')\nprint(f'short_len={len(short)}')",
            # Cell 3: combine
            "combo = textwrap.shorten(text, width=40, placeholder=' [...]')\nprint(f'combo={combo}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "break_lines=" in out1
        assert "no_break_lines=" in out1
        out2 = nb_runner.get_output(2)
        assert "shortened=" in out2
        assert len(nb_runner.get_output(2).split("shortened=")[1].split("\n")[0]) <= 30
        out3 = nb_runner.get_output(3)
        assert "combo=" in out3

    def test_textwrap_fill_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\nparagraph = 'Python is a versatile language used for web development data science and automation'\nfilled = textwrap.fill(paragraph, width=25)\nline_count = len(filled.split('\\n'))\nprint(f'lines={line_count}')",
            "first = filled.split('\\n')[0]\nprint(f'first_line={first}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(1)
        assert "lines=" in out

        # Edit width
        nb_runner.set_cell_source(1, "import textwrap\nparagraph = 'Python is a versatile language used for web development data science and automation'\nfilled = textwrap.fill(paragraph, width=50)\nline_count = len(filled.split('\\n'))\nprint(f'lines={line_count}')")
        nb_runner.run_cells([1, 2])
        out = nb_runner.get_output(2)
        assert "first_line=" in out

    def test_textwrap_fill_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap\nmsg = 'Hello World'\nresult = textwrap.shorten(msg, width=20, placeholder='...')\nprint(f'result={result}')",
            "is_truncated = '...' in result\nprint(f'truncated={is_truncated}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Hello World" in nb_runner.get_output(1)
        assert "truncated=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "truncated=False" in nb_runner.get_output(2)
