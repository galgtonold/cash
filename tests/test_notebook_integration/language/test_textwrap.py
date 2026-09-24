"""textwrap and whitespace handling across cells."""

import pytest


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapFormat:
    """Test textwrap formatting operation caching."""

    def test_dedent_basic(self, nb_runner):
        """Dedent indented text, verify caching."""
        nb_runner.create_notebook(
            [
                "import textwrap",
                "raw = '    line1\\n    line2\\n    line3'",
                "cleaned = textwrap.dedent(raw)\nlines = cleaned.strip().split('\\n')",
                "print(f'count={len(lines)} first={lines[0]}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = 'The quick brown fox jumps over the lazy dog near the river bank'",
                "width = 20",
                "wrapped = textwrap.fill(text, width=width)\nline_count = len(wrapped.split('\\n'))",
                "print(f'lines={line_count}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = 'line1\\nline2\\nline3'",
                "indented = textwrap.indent(text, '>>> ')\nfirst_line = indented.split('\\n')[0]",
                "print(f'first={first_line}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "first=>>> line1" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "first=>>> line1" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapDedent:
    """textwrap, dedent, and multi-line string formatting."""

    def test_wrap_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nlong_text = 'The quick brown fox jumps over the lazy dog and then runs away'",
                "wrapped = textwrap.fill(long_text, width=30)\nlines = wrapped.count('\\n') + 1\nprint(f'lines={lines}')\nprint(f'wrapped={repr(wrapped)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lines=" in out

    def test_dedent_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = '    line1\\n    line2\\n    line3'",
                "dedented = textwrap.dedent(text)\nfirst = dedented.split('\\n')[0]\nprint(f'first={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=line1" in nb_runner.get_output(2)
        # Edit text
        nb_runner.set_cell_source(1, "import textwrap\ntext = '        hello\\n        world'")
        nb_runner.run_all()
        assert "first=hello" in nb_runner.get_output(2)

    def test_indent(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = 'line1\\nline2\\nline3'",
                "indented = textwrap.indent(text, '>>> ')\nprint(f'indented={repr(indented)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert ">>> line1" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapDedentFill:
    """textwrap module dedent and fill."""

    def test_dedent(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nraw = '    hello\\n    world\\n    foo'",
                "cleaned = textwrap.dedent(raw)\nlines = cleaned.strip().split('\\n')\nprint(f'lines={lines}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lines=['hello', 'world', 'foo']" in nb_runner.get_output(2)

    def test_fill(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = 'The quick brown fox jumps over the lazy dog and runs away'",
                "wrapped = textwrap.fill(text, width=20)\nline_count = len(wrapped.split('\\n'))\nprint(f'line_count={line_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        count = int(nb_runner.get_output(2).split("line_count=")[1].strip())
        assert count >= 3

    def test_textwrap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = 'hello world test'",
                "shortened = textwrap.shorten(text, width=12, placeholder='...')\nprint(f'short={shortened}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "short=hello..." in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import textwrap\ntext = 'foo bar baz qux'")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "short=foo bar..." in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapDedentIndent:
    """textwrap dedent indent and fill wrapping."""

    def test_dedent(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = '''\n        Hello World\n        This is indented\n        Three lines\n    '''\ndedented = textwrap.dedent(text).strip()\nlines = dedented.split('\\n')\nprint(f'lines={len(lines)} first={lines[0]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lines=3" in out
        assert "first=Hello World" in out

    def test_fill_width(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = 'The quick brown fox jumps over the lazy dog near the river'\nfilled = textwrap.fill(text, width=30)\nlines = filled.split('\\n')\nprint(f'line_count={len(lines)}')\nprint(f'max_len={max(len(l) for l in lines)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "max_len=" in out

    def test_wrap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap",
                "text = 'hello world foo bar'\nwrapped = textwrap.wrap(text, width=12)\nprint(f'parts={len(wrapped)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "text = 'a b c d e f g h'\nwrapped = textwrap.wrap(text, width=8)\nprint(f'parts={len(wrapped)}')"
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "parts=" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapFillBreak:
    """Test textwrap fill with break options across cells."""

    def test_textwrap_fill_break(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: fill with break options
                "import textwrap\ntext = 'This is a very-long-hyphenated-word-that-should-break and some more text after it'\nfilled_break = textwrap.fill(text, width=30, break_on_hyphens=True)\nfilled_no_break = textwrap.fill(text, width=30, break_on_hyphens=False)\nbreak_lines = len(filled_break.split('\\n'))\nno_break_lines = len(filled_no_break.split('\\n'))\nprint(f'break_lines={break_lines}')\nprint(f'no_break_lines={no_break_lines}')",
                # Cell 2: shorten
                "long = 'The quick brown fox jumps over the lazy dog near the river'\nshort = textwrap.shorten(long, width=30, placeholder='...')\nprint(f'shortened={short}')\nprint(f'short_len={len(short)}')",
                # Cell 3: combine
                "combo = textwrap.shorten(text, width=40, placeholder=' [...]')\nprint(f'combo={combo}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import textwrap\nparagraph = 'Python is a versatile language used for web development data science and automation'\nfilled = textwrap.fill(paragraph, width=25)\nline_count = len(filled.split('\\n'))\nprint(f'lines={line_count}')",
                "first = filled.split('\\n')[0]\nprint(f'first_line={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(1)
        assert "lines=" in out

        # Edit width
        nb_runner.set_cell_source(
            1,
            "import textwrap\nparagraph = 'Python is a versatile language used for web development data science and automation'\nfilled = textwrap.fill(paragraph, width=50)\nline_count = len(filled.split('\\n'))\nprint(f'lines={line_count}')",
        )
        nb_runner.run_cells([1, 2])
        out = nb_runner.get_output(2)
        assert "first_line=" in out

    def test_textwrap_fill_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nmsg = 'Hello World'\nresult = textwrap.shorten(msg, width=20, placeholder='...')\nprint(f'result={result}')",
                "is_truncated = '...' in result\nprint(f'truncated={is_truncated}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Hello World" in nb_runner.get_output(1)
        assert "truncated=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "truncated=False" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTextwrapWrapShorten:
    """Test textwrap.wrap and shorten across cells."""

    def test_wrap_shorten(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: wrap text
                "import textwrap\ntext = 'The quick brown fox jumps over the lazy dog and continues running through the forest'\nwrapped = textwrap.wrap(text, width=30)\nprint(f'lines={len(wrapped)}')\nfor line in wrapped:\n    print(f'  |{line}|')",
                # Cell 2: shorten
                "short = textwrap.shorten(text, width=40, placeholder='...')\nprint(f'short={short}')\nprint(f'short_len={len(short)}')",
                # Cell 3: indent
                "indented = textwrap.indent(text, prefix='>>> ')\nprint(f'indented={indented}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import textwrap\ntext = 'Hello World this is a test of text wrapping'\nlines = textwrap.wrap(text, width=20)\nline_count = len(lines)\nprint(f'lines={line_count}')",
                "first = lines[0]\nprint(f'first={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Change width
        nb_runner.set_cell_source(
            1,
            "import textwrap\ntext = 'Hello World this is a test of text wrapping'\nlines = textwrap.wrap(text, width=10)\nline_count = len(lines)\nprint(f'lines={line_count}')",
        )
        nb_runner.run_cells([1, 2])
        # Narrower width = more lines
        count = int(nb_runner.get_output(1).split("lines=")[1].strip())
        assert count > 3

    def test_wrap_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import textwrap\nresult = textwrap.fill('A short sentence for testing.', width=15)\nprint(f'filled={result}')",
                "line_count = result.count('\\n') + 1\nprint(f'count={line_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "count=" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStringExpandtabs:
    """string expandtabs and whitespace handling."""

    def test_expandtabs(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'col1\\tcol2\\tcol3'",
                "expanded = text.expandtabs(8)\ncols = expanded.split()\nprint(f'cols={cols}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cols=['col1', 'col2', 'col3']" in nb_runner.get_output(2)

    def test_strip_variations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "s = '  hello  '",
                "l = s.lstrip()\nr = s.rstrip()\nb = s.strip()\nprint(f'l=[{l}] r=[{r}] b=[{b}]')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "l=[hello  ]" in out
        assert "r=[  hello]" in out
        assert "b=[hello]" in out

    def test_whitespace_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "raw = '  spaces  and\\ttabs  '",
                "cleaned = ' '.join(raw.split())\nprint(f'cleaned={cleaned}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=spaces and tabs" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "raw = '\\t\\thello\\t\\tworld\\t\\t'")
        nb_runner.run_all()
        assert "cleaned=hello world" in nb_runner.get_output(2)
