"""re: search, findall, sub and compiled patterns across cells."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestRegexEdits:
    """Editing regex patterns."""

    def test_edit_regex_pattern(self, nb_runner):
        """Edit the regex pattern in a search."""
        nb_runner.create_notebook(
            [
                "import re",
                "text = 'Hello World 123'  # regex source",
                "match = re.findall(r'\\d+', text)\nprint(f'match = {match}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "match = ['123']" in nb_runner.get_output(3)

        # Change pattern to words
        nb_runner.set_cell_source(3, "match = re.findall(r'[A-Z][a-z]+', text)\nprint(f'match = {match}')")
        nb_runner.run_all()
        assert "match = ['Hello', 'World']" in nb_runner.get_output(3)

    def test_edit_regex_substitution(self, nb_runner):
        """Edit regex substitution."""
        nb_runner.create_notebook(
            [
                "import re",
                "text = 'foo bar baz'  # sub source",
                "result = re.sub(r'\\s+', '-', text)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = foo-bar-baz" in nb_runner.get_output(3)

        # Change replacement
        nb_runner.set_cell_source(3, "result = re.sub(r'\\s+', '_', text)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = foo_bar_baz" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.integration
class TestRegexPatterns:
    """Regular expression caching scenarios."""

    def test_regex_substitution_propagation(self, nb_runner):
        """Regex substitution with upstream change propagation."""
        nb_runner.create_notebook(
            [
                "replacement = 'REDACTED'",
                textwrap.dedent("""\
                import re
                text = "SSN: 123-45-6789, Phone: 555-1234"
                ssn_pat = re.compile(r'\\d{3}-\\d{2}-\\d{4}')
                cleaned = ssn_pat.sub(replacement, text)
            """),
                "print(f'cleaned={cleaned}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "REDACTED" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "replacement = '***-**-****'")
        nb_runner.run_cells([1, 2, 3])
        assert "***-**-****" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestCompiledRegexChange:
    """Test regex patterns across cells."""

    def test_regex_change_propagation(self, nb_runner):
        """Change regex pattern → downstream updates."""
        nb_runner.create_notebook(
            [
                "import re",
                "pattern = re.compile(r'\\b[A-Z][a-z]+\\b')",
                textwrap.dedent("""\
                text = "Hello World foo Bar"
                matches = pattern.findall(text)
                print(matches)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Hello" in output
        assert "World" in output
        assert "Bar" in output

        # Change to only match 5+ char capitalized words
        nb_runner.set_cell_source(2, "pattern = re.compile(r'\\b[A-Z][a-z]{4,}\\b')")
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert "Hello" in output2
        assert "World" in output2
        # "Bar" is only 3 chars, should NOT match
        assert "Bar" not in output2


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestRegexInteraction:
    """Test regex compilation and matching with cache invalidation."""

    def test_regex_pattern_edit(self, nb_runner):
        """Editing a regex pattern should invalidate match results."""
        nb_runner.create_notebook(
            [
                "import re\npattern = re.compile(r'\\d+')",
                "text = 'abc 123 def 456'",
                "matches = pattern.findall(text)",
                "result = ','.join(matches)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=123,456" in out

        # Change pattern to match words instead
        nb_runner.set_cell_source(1, "import re\npattern = re.compile(r'[a-z]+')")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=abc,def" in out

    def test_regex_sub_edit(self, nb_runner):
        """Editing substitution pattern should propagate."""
        nb_runner.create_notebook(
            [
                "import re\nreplacer = re.compile(r'\\s+')",
                "text = 'hello   world   python'",
                "cleaned = replacer.sub('-', text)",
                "print(f'cleaned={cleaned}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "cleaned=hello-world-python" in out

        # Change to replace with underscore
        nb_runner.set_cell_source(1, "import re\nreplacer = re.compile(r'\\s+')")
        nb_runner.set_cell_source(3, "cleaned = replacer.sub('_', text)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "cleaned=hello_world_python" in out

    def test_regex_groups_edit(self, nb_runner):
        """Editing regex with groups should propagate captured groups."""
        nb_runner.create_notebook(
            [
                "import re\npat = re.compile(r'(\\w+)@(\\w+\\.\\w+)')",
                "email = 'alice@example.com'",
                "m = pat.match(email)\nuser = m.group(1)\ndomain = m.group(2)",
                "print(f'user={user},domain={domain}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "user=alice,domain=example.com" in out

        nb_runner.set_cell_source(2, "email = 'bob@work.org'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "user=bob,domain=work.org" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestRegexCompileGroups:
    """Test compiled regex with named groups across cells."""

    def test_regex_compile_finditer(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: compile pattern with named groups
                "import re\npattern = re.compile(r'(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})')\nprint(f'pattern_type={type(pattern).__name__}')",
                # Cell 2: finditer over text
                "text = 'Born 1990-05-14, graduated 2012-06-20, married 2018-09-03'\nmatches = [m.groupdict() for m in pattern.finditer(text)]\nprint(f'count={len(matches)}')\nfor m in matches:\n    print(f\"{m['year']}/{m['month']}/{m['day']}\")",
                # Cell 3: derive summary
                "years = [int(m['year']) for m in matches]\nspan = max(years) - min(years)\nprint(f'year_span={span}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "pattern_type=Pattern" in out1
        out2 = nb_runner.get_output(2)
        assert "count=3" in out2
        assert "1990/05/14" in out2
        out3 = nb_runner.get_output(3)
        assert "year_span=28" in out3

    def test_regex_pattern_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\npattern = re.compile(r'(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})')\nprint(f'pattern_type={type(pattern).__name__}')",
                "text = 'Born 1990-05-14, graduated 2012-06-20, married 2018-09-03'\nmatches = [m.groupdict() for m in pattern.finditer(text)]\nprint(f'count={len(matches)}')",
                "years = [int(m['year']) for m in matches]\nspan = max(years) - min(years)\nprint(f'year_span={span}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "year_span=28" in nb_runner.get_output(3)

        # Edit text to have more dates
        nb_runner.set_cell_source(
            2,
            "text = 'Born 1990-05-14, grad 2012-06-20, married 2018-09-03, child 2020-11-15'\nmatches = [m.groupdict() for m in pattern.finditer(text)]\nprint(f'count={len(matches)}')",
        )
        nb_runner.run_cells([2, 3])
        assert "count=4" in nb_runner.get_output(2)
        assert "year_span=30" in nb_runner.get_output(3)

    def test_regex_cache_correctness(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\npattern = re.compile(r'(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})')\nprint(f'compiled=True')",
                "text = 'Event on 2023-01-15 and 2023-12-31'\nmatches = [m.groupdict() for m in pattern.finditer(text)]\nprint(f'count={len(matches)}')",
                "months = sorted(set(int(m['month']) for m in matches))\nprint(f'months={months}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "months=[1, 12]" in nb_runner.get_output(3)

        # Re-run without changes - should use cache
        nb_runner.run_all()
        assert "months=[1, 12]" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestReFindallSplit:
    """Test re.findall and re.split across cells."""

    def test_findall_split(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: findall with groups
                "import re\ntext = 'John:25, Jane:30, Bob:22'\nmatches = re.findall(r'(\\w+):(\\d+)', text)\nprint(f'matches={matches}')",
                # Cell 2: re.split
                "data = '1-2-3--4---5'\nparts = re.split(r'-+', data)\nprint(f'parts={parts}')",
                # Cell 3: use findall results
                "names = [m[0] for m in matches]\nages = [int(m[1]) for m in matches]\navg_age = sum(ages) / len(ages)\nprint(f'names={names}')\nprint(f'avg_age={avg_age:.1f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "('John', '25')" in out1
        assert "('Jane', '30')" in out1
        out2 = nb_runner.get_output(2)
        assert "parts=['1', '2', '3', '4', '5']" in out2
        out3 = nb_runner.get_output(3)
        assert "names=['John', 'Jane', 'Bob']" in out3
        assert "avg_age=25.7" in out3

    def test_findall_split_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\ntext = 'a1b2c3'\ndigits = re.findall(r'\\d', text)\nprint(f'digits={digits}')",
                "total = sum(int(d) for d in digits)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "digits=['1', '2', '3']" in nb_runner.get_output(1)
        assert "total=6" in nb_runner.get_output(2)

        # Edit text
        nb_runner.set_cell_source(
            1, "import re\ntext = 'x9y8z7w6'\ndigits = re.findall(r'\\d', text)\nprint(f'digits={digits}')"
        )
        nb_runner.run_cells([1, 2])
        assert "total=30" in nb_runner.get_output(2)

    def test_findall_split_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\nwords = re.findall(r'\\w+', 'Hello, World! How are you?')\nprint(f'words={words}')",
                "count = len(words)\nprint(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words=['Hello', 'World', 'How', 'are', 'you']" in nb_runner.get_output(1)
        assert "count=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestReFindallSub:
    """re module findall and sub patterns."""

    def test_findall(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\ntext = 'Price: $10.50, Discount: $2.00, Total: $8.50'",
                "prices = re.findall(r'\\$([\\d.]+)', text)\nprint(f'prices={prices}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "prices=['10.50', '2.00', '8.50']" in nb_runner.get_output(2)

    def test_sub(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\ntext = 'foo  bar   baz    qux'",
                "cleaned = re.sub(r'\\s+', ' ', text)\nprint(f'cleaned={cleaned}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=foo bar baz qux" in nb_runner.get_output(2)

    def test_regex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\ndata = 'user123_test456'",
                "nums = re.findall(r'\\d+', data)\nprint(f'nums={nums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "nums=['123', '456']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import re\ndata = 'abc99_def88_ghi77'")
        nb_runner.run_all()
        assert "nums=['99', '88', '77']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestReSubFindallComplex:
    """re.sub and re.findall complex patterns."""

    def test_sub_replacement(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re",
                "text = 'Call 123-456-7890 or 098-765-4321'\nmasked = re.sub(r'(\\d{3})-(\\d{3})-(\\d{4})', r'XXX-XXX-\\3', text)\nprint(f'masked={masked}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "XXX-XXX-7890" in out
        assert "XXX-XXX-4321" in out

    def test_findall_groups(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re",
                "text = 'prices: $12.50, $3.99, $100.00'\nprices = re.findall(r'\\$(\\d+\\.\\d{2})', text)\ntotal = sum(float(p) for p in prices)\nprint(f'prices={prices} total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "12.50" in out
        assert "total=116.49" in out

    def test_re_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re",
                "text = 'abc 123 def'\nnums = re.findall(r'\\d+', text)\nprint(f'nums={nums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "nums=['123']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "text = 'a1 b22 c333'\nnums = re.findall(r'\\d+', text)\nprint(f'nums={nums}')")
        nb_runner.run_all()
        assert "nums=['1', '22', '333']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestReSubFunctionReplace:
    """Test re.sub with function replacement across cells."""

    def test_re_sub_function(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: sub with function replacement
                "import re\ndef double_num(m):\n    return str(int(m.group()) * 2)\n\ntext = 'item1 costs 5 dollars and item2 costs 10 dollars'\nresult = re.sub(r'\\d+', double_num, text)\nprint(f'result={result}')",
                # Cell 2: sub with backreference
                "swapped = re.sub(r'(\\w+) costs (\\d+)', r'\\2 for \\1', text)\nprint(f'swapped={swapped}')",
                # Cell 3: count substitutions
                "count = 0\ndef counter_replace(m):\n    global count\n    count += 1\n    return f'[{count}]'\n\ncounted = re.sub(r'\\d+', counter_replace, text)\nprint(f'counted={counted}')\nprint(f'replacements={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "item2" in out1 and "10" in out1 and "20" in out1
        out2 = nb_runner.get_output(2)
        assert "5 for item1" in out2
        out3 = nb_runner.get_output(3)
        assert "replacements=" in out3

    def test_re_sub_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\npattern = r'[aeiou]'\ntext = 'hello world'\nresult = re.sub(pattern, '*', text)\nprint(f'result={result}')",
                "vowel_free_len = len(result.replace('*', ''))\nprint(f'consonants={vowel_free_len}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=h*ll* w*rld" in nb_runner.get_output(1)
        # h,l,l,' ',w,r,l,d = 8 non-vowel chars
        assert "consonants=8" in nb_runner.get_output(2)

        # Edit pattern to uppercase too
        nb_runner.set_cell_source(
            1,
            "import re\npattern = r'[aeiouAEIOU]'\ntext = 'Hello World'\nresult = re.sub(pattern, '*', text)\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "result=H*ll* W*rld" in nb_runner.get_output(1)

    def test_re_sub_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import re\nraw = '  hello   world  '\ncleaned = re.sub(r'\\s+', ' ', raw).strip()\nprint(f'cleaned={cleaned}')",
                "word_count = len(cleaned.split())\nprint(f'words={word_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=hello world" in nb_runner.get_output(1)
        assert "words=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "words=2" in nb_runner.get_output(2)
