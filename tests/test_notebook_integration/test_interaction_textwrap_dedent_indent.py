"""Batch 498: textwrap dedent indent and fill wrapping."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTextwrapDedentIndent:
    def test_dedent(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap",
            "text = '''\n        Hello World\n        This is indented\n        Three lines\n    '''\ndedented = textwrap.dedent(text).strip()\nlines = dedented.split('\\n')\nprint(f'lines={len(lines)} first={lines[0]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lines=3" in out
        assert "first=Hello World" in out

    def test_fill_width(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap",
            "text = 'The quick brown fox jumps over the lazy dog near the river'\nfilled = textwrap.fill(text, width=30)\nlines = filled.split('\\n')\nprint(f'line_count={len(lines)}')\nprint(f'max_len={max(len(l) for l in lines)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "max_len=" in out

    def test_wrap_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import textwrap",
            "text = 'hello world foo bar'\nwrapped = textwrap.wrap(text, width=12)\nprint(f'parts={len(wrapped)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "text = 'a b c d e f g h'\nwrapped = textwrap.wrap(text, width=8)\nprint(f'parts={len(wrapped)}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "parts=" in out2
