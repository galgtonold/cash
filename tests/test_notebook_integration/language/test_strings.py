"""String methods: split, join, partition, translate, casefold."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


# String manipulation chain interaction tests.
#
# Tests editing string processing pipelines across cells
# with transformations, formatting, and parsing.
@pytest.mark.upstream
class TestStringChainEdits:
    """String processing chain edits."""

    def test_edit_string_transform(self, nb_runner):
        """Edit a string transformation in a chain."""
        nb_runner.create_notebook(
            [
                "raw = '  Hello, World!  '  # raw input string",
                "cleaned = raw.strip()",
                "result = cleaned.lower()\nprint(f'result = [{result}]')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [hello, world!]" in nb_runner.get_output(3)

        # Change to upper
        nb_runner.set_cell_source(3, "result = cleaned.upper()\nprint(f'result = [{result}]')")
        nb_runner.run_all()
        assert "result = [HELLO, WORLD!]" in nb_runner.get_output(3)

    def test_edit_source_string(self, nb_runner):
        """Edit the source string, verify chain updates."""
        nb_runner.create_notebook(
            [
                "text = 'python is great'  # source text",
                "words = text.split()",
                "result = '-'.join(words)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = python-is-great" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "text = 'cash is amazing'  # source text changed")
        nb_runner.run_all()
        assert "result = cash-is-amazing" in nb_runner.get_output(3)

    def test_edit_join_separator(self, nb_runner):
        """Edit the separator in a join operation."""
        nb_runner.create_notebook(
            [
                "parts = ['2024', '01', '15']  # date parts",
                "date_str = '-'.join(parts)\nprint(f'date = {date_str}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "date = 2024-01-15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "date_str = '/'.join(parts)\nprint(f'date = {date_str}')")
        nb_runner.run_all()
        assert "date = 2024/01/15" in nb_runner.get_output(2)


@pytest.mark.upstream
class TestStringParsingEdits:
    """String parsing with edits."""

    def test_edit_regex_pattern(self, nb_runner):
        """Edit a regex pattern used for parsing."""
        nb_runner.create_notebook(
            [
                "import re",
                "text = 'price: $42.50, tax: $3.50'  # text to parse",
                "matches = re.findall(r'\\$([\\d.]+)', text)\nprint(f'matches = {matches}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42.50" in nb_runner.get_output(3)
        assert "3.50" in nb_runner.get_output(3)

        # Change to find only integers
        nb_runner.set_cell_source(3, "matches = re.findall(r'\\d+', text)\nprint(f'matches = {matches}')")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "42" in out
        assert "50" in out

    def test_edit_string_template(self, nb_runner):
        """Edit a string format template."""
        nb_runner.create_notebook(
            [
                "name = 'Alice'\nage = 30",
                "msg = f'{name} is {age} years old'\nprint(f'msg = {msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = Alice is 30 years old" in nb_runner.get_output(2)

        # Change template
        nb_runner.set_cell_source(
            2,
            "msg = f'Name: {name}, Age: {age}'\nprint(f'msg = {msg}')",
        )
        nb_runner.run_all()
        assert "msg = Name: Alice, Age: 30" in nb_runner.get_output(2)

    def test_multiline_string_edit(self, nb_runner):
        """Edit a multiline string."""
        nb_runner.create_notebook(
            [
                "lines = ['line1', 'line2', 'line3']  # lines data",
                "text = '\\n'.join(lines)\nline_count = len(text.splitlines())\nprint(f'lines = {line_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lines = 3" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "lines = ['a', 'b', 'c', 'd', 'e']  # lines data more")
        nb_runner.run_all()
        assert "lines = 5" in nb_runner.get_output(2)


# Interaction test: string casefold and unicode normalization.
# Tests casefold for case-insensitive comparisons, unicode normalization
# concepts, and cross-cell string equality checks.
class TestStringCasefoldNorm:
    """Test string casefold and normalization across cells."""

    def test_casefold_comparisons(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: casefold comparisons
                "s1 = 'Straße'\ns2 = 'STRASSE'\ns3 = 'straße'\nprint(f'lower_eq={s1.lower() == s2.lower()}')\nprint(f'casefold_eq={s1.casefold() == s2.casefold()}')\nprint(f's1_cf={s1.casefold()}')",
                # Cell 2: build case-insensitive lookup
                "words = ['Hello', 'WORLD', 'Python', 'hello', 'python']\nunique_cf = set(w.casefold() for w in words)\nprint(f'original={len(words)}')\nprint(f'unique={len(unique_cf)}')",
                # Cell 3: case-insensitive search
                "search = 'HELLO'\nfound = [w for w in words if w.casefold() == search.casefold()]\nprint(f'matches={found}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "lower_eq=False" in out1
        assert "casefold_eq=True" in out1
        assert "s1_cf=strasse" in out1
        out2 = nb_runner.get_output(2)
        assert "original=5" in out2
        assert "unique=3" in out2
        out3 = nb_runner.get_output(3)
        assert "matches=['Hello', 'hello']" in out3

    def test_casefold_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['Cat', 'DOG', 'cat', 'Bird', 'dog']\nunique_cf = sorted(set(w.casefold() for w in words))\nprint(f'unique={unique_cf}')",
                "lookup = {w.casefold(): w for w in words}\nprint(f'keys={sorted(lookup.keys())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=['bird', 'cat', 'dog']" in nb_runner.get_output(1)

        # Add more words
        nb_runner.set_cell_source(
            1,
            "words = ['Cat', 'DOG', 'cat', 'Bird', 'dog', 'FISH']\nunique_cf = sorted(set(w.casefold() for w in words))\nprint(f'unique={unique_cf}')",
        )
        nb_runner.run_cells([1, 2])
        assert "unique=['bird', 'cat', 'dog', 'fish']" in nb_runner.get_output(1)

    def test_casefold_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'The Quick Brown Fox'\ncf = text.casefold()\nprint(f'cf={cf}')",
                "word_count = len(cf.split())\nprint(f'words={word_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cf=the quick brown fox" in nb_runner.get_output(1)
        assert "words=4" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "words=4" in nb_runner.get_output(2)


class TestStringMaketransCipher:
    """string maketrans and translate cipher."""

    def test_caesar_cipher(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import string",
                "shift = 3\nalpha = string.ascii_lowercase\nshifted = alpha[shift:] + alpha[:shift]\ntable = str.maketrans(alpha, shifted)\nencoded = 'hello world'.translate(table)\nprint(f'encoded={encoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded=khoor zruog" in nb_runner.get_output(2)

    def test_remove_chars(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'Hello, World! 123'",
                "table = str.maketrans('', '', '!,. ')\ncleaned = text.translate(table)\nprint(f'cleaned={cleaned}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=HelloWorld123" in nb_runner.get_output(2)

    def test_cipher_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import string",
                "shift = 1\nalpha = string.ascii_lowercase\nshifted = alpha[shift:] + alpha[:shift]\ntable = str.maketrans(alpha, shifted)\nresult = 'abc'.translate(table)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=bcd" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "shift = 2\nalpha = string.ascii_lowercase\nshifted = alpha[shift:] + alpha[:shift]\ntable = str.maketrans(alpha, shifted)\nresult = 'abc'.translate(table)\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=cde" in nb_runner.get_output(2)


# Interaction test: string maketrans with translate and multi-char replacement.
# Tests str.maketrans with 3-arg form (intab, outtab, delchars), translate,
# and cross-cell text transformation pipelines.
class TestStringMaketransTranslate:
    """Test str.maketrans and translate across cells."""

    def test_maketrans_translate_pipeline(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create translation table
                "table = str.maketrans('aeiou', '12345', '!?.')\nprint(f'table_size={len(table)}')",
                # Cell 2: apply translations
                "text = 'Hello, World! Are you okay?'\ntranslated = text.translate(table)\nprint(f'result={translated}')",
                # Cell 3: analyze
                "vowel_count = sum(1 for c in 'Hello, World! Are you okay?' if c in 'aeiou')\ndigit_count = sum(1 for c in translated if c.isdigit())\npunct_removed = all(c not in translated for c in '!?.')\nprint(f'vowels={vowel_count}')\nprint(f'digits={digit_count}')\nprint(f'punct_clean={punct_removed}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "table_size=" in out1
        out2 = nb_runner.get_output(2)
        assert "result=" in out2
        out3 = nb_runner.get_output(3)
        assert "punct_clean=True" in out3

    def test_maketrans_edit_mapping(self, nb_runner):
        nb_runner.create_notebook(
            [
                "table = str.maketrans({'a': '@', 'e': '3', 'i': '!', 'o': '0', 's': '$'})\nprint(f'table_type={type(table).__name__}')",
                "text = 'secret message'\nencoded = text.translate(table)\nprint(f'encoded={encoded}')",
                "diff_count = sum(1 for a, b in zip(text, encoded) if a != b)\nprint(f'changes={diff_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "encoded=$3cr3t m3$$@g3" in out2

        # Edit mapping
        nb_runner.set_cell_source(
            1,
            "table = str.maketrans({'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5'})\nprint(f'table_type={type(table).__name__}')",
        )
        nb_runner.run_cells([1, 2, 3])
        assert "encoded=53cr3t m3554g3" in nb_runner.get_output(2)

    def test_maketrans_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "rot13_in = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\nrot13_out = 'nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM'\ntable = str.maketrans(rot13_in, rot13_out)\nprint('rot13_ready')",
                "msg = 'Hello World'\nencoded = msg.translate(table)\ndecoded = encoded.translate(table)\nprint(f'encoded={encoded}')\nprint(f'roundtrip={decoded == msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "encoded=Uryyb Jbeyq" in out
        assert "roundtrip=True" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "roundtrip=True" in nb_runner.get_output(2)

    # Interaction test: string maketrans and translate.
    # Tests str.maketrans for character mapping, translate application,
    # ROT13-like transformations, and cross-cell encoding pipelines.
    def test_maketrans_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic translation
                "table = str.maketrans('aeiou', '12345')\ntext = 'hello world'\ntranslated = text.translate(table)\nprint(f'translated={translated}')",
                # Cell 2: delete characters
                "delete_table = str.maketrans('', '', 'lo')\ncleaned = text.translate(delete_table)\nprint(f'cleaned={cleaned}')",
                # Cell 3: combine
                "combined = translated.translate(delete_table)\nprint(f'combined={combined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "translated=h2ll4 w4rld" in out1
        out2 = nb_runner.get_output(2)
        assert "cleaned=he wrd" in out2
        out3 = nb_runner.get_output(3)
        assert "combined=h2" in out3

    def test_maketrans_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "table = str.maketrans('abc', 'xyz')\ntext = 'abcdef'\nresult = text.translate(table)\nprint(f'result={result}')",
                "has_a = 'a' in result\nprint(f'has_a={has_a}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=xyzdef" in nb_runner.get_output(1)
        assert "has_a=False" in nb_runner.get_output(2)

        # Edit mapping
        nb_runner.set_cell_source(
            1,
            "table = str.maketrans('def', 'DEF')\ntext = 'abcdef'\nresult = text.translate(table)\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "result=abcDEF" in nb_runner.get_output(1)
        assert "has_a=True" in nb_runner.get_output(2)

    def test_maketrans_result_feeds_a_later_cell(self, nb_runner):
        nb_runner.create_notebook(
            [
                "table = str.maketrans('0123456789', 'abcdefghij')\ndigits = '12345'\nencoded = digits.translate(table)\nprint(f'encoded={encoded}')",
                "length = len(encoded)\nprint(f'length={length}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded=bcdef" in nb_runner.get_output(1)
        assert "length=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "length=5" in nb_runner.get_output(2)

    def test_translate_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "msg = 'abc'",
                "table = str.maketrans('abc', 'xyz')\nresult = msg.translate(table)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=xyz" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "msg = 'aabbcc'")
        nb_runner.run_all()
        assert "result=xxyyzz" in nb_runner.get_output(2)


class TestMultilineStringManip:
    """multi-line string manipulation and triple quotes."""

    def test_triple_quote_strip(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = '''  \n  hello  \n  world  \n  '''",
                "lines = [line.strip() for line in text.strip().split('\\n')]\nfiltered = [l for l in lines if l]\nprint(f'filtered={filtered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filtered=['hello', 'world']" in nb_runner.get_output(2)

    def test_multiline_join(self, nb_runner):
        nb_runner.create_notebook(
            [
                "parts = ['line one', 'line two', 'line three']",
                "combined = '\\n'.join(parts)\ncount = combined.count('\\n')\nprint(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)

    def test_multiline_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = 'a,b,c\\n1,2,3\\n4,5,6'",
                "rows = data.strip().split('\\n')\nheader = rows[0].split(',')\nnum_rows = len(rows) - 1\nprint(f'header={header} num_rows={num_rows}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header=['a', 'b', 'c']" in nb_runner.get_output(2)
        assert "num_rows=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "data = 'x,y\\n10,20\\n30,40\\n50,60'")
        nb_runner.run_all()
        assert "header=['x', 'y']" in nb_runner.get_output(2)
        assert "num_rows=3" in nb_runner.get_output(2)


# Interaction test: string methods partition and rpartition.
# Tests str.partition and rpartition for splitting around separators,
# with cross-cell parsing pipelines.
class TestPartitionRpartition:
    """Test str.partition and rpartition across cells."""

    def test_partition_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic partition
                "url = 'https://example.com:8080/path/to/resource'\nscheme, _, rest = url.partition('://')\nhost_port, _, path = rest.partition('/')\nprint(f'scheme={scheme}')\nprint(f'host_port={host_port}')\nprint(f'path={path}')",
                # Cell 2: rpartition for last separator
                "filepath = 'home/user/docs/report.final.pdf'\ndir_part, _, filename = filepath.rpartition('/')\nbase, _, ext = filename.rpartition('.')\nprint(f'dir={dir_part}')\nprint(f'base={base}')\nprint(f'ext={ext}')",
                # Cell 3: combine parsed info
                "info = f'{scheme}://{host_port}/{base}.{ext}'\nprint(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "scheme=https" in out1
        assert "host_port=example.com:8080" in out1
        assert "path=path/to/resource" in out1
        out2 = nb_runner.get_output(2)
        assert "dir=home/user/docs" in out2
        assert "base=report.final" in out2
        assert "ext=pdf" in out2
        out3 = nb_runner.get_output(3)
        assert "info=https://example.com:8080/report.final.pdf" in out3

    def test_partition_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "email = 'user@example.com'\nlocal, _, domain = email.partition('@')\nprint(f'local={local}')\nprint(f'domain={domain}')",
                "tld = domain.rpartition('.')[2]\nprint(f'tld={tld}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "local=user" in nb_runner.get_output(1)
        assert "tld=com" in nb_runner.get_output(2)

        # Edit email
        nb_runner.set_cell_source(
            1,
            "email = 'admin@mail.example.co.uk'\nlocal, _, domain = email.partition('@')\nprint(f'local={local}')\nprint(f'domain={domain}')",
        )
        nb_runner.run_cells([1, 2])
        assert "local=admin" in nb_runner.get_output(1)
        assert "tld=uk" in nb_runner.get_output(2)

    def test_partition_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "kv = 'name=John Doe'\nkey, _, val = kv.partition('=')\nprint(f'key={key}')\nprint(f'val={val}')",
                "upper_key = key.upper()\nprint(f'upper={upper_key}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "key=name" in nb_runner.get_output(1)
        assert "upper=NAME" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "upper=NAME" in nb_runner.get_output(2)


class TestStringSplitJoinPartition:
    """string split join partition operations."""

    def test_split_join_round(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'hello world python'",
                "words = text.split()\njoined = '-'.join(words)\nprint(f'words={words} joined={joined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "hello" in out
        assert "joined=hello-world-python" in out

    def test_partition(self, nb_runner):
        nb_runner.create_notebook(
            [
                "path = 'user@host:port'",
                "user, sep, rest = path.partition('@')\nhost, sep2, port = rest.partition(':')\nprint(f'user={user} host={host} port={port}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "user=user" in out
        assert "host=host" in out
        assert "port=port" in out

    def test_split_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "csv_line = 'a,b,c'",
                "parts = csv_line.split(',')\nprint(f'parts={parts}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "parts=['a', 'b', 'c']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "csv_line = 'x,y,z,w'")
        nb_runner.run_all()
        assert "parts=['x', 'y', 'z', 'w']" in nb_runner.get_output(2)


class TestStringCasefoldUnicode:
    """string casefold and unicode normalization."""

    def test_casefold(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = 'Straße'\nb = 'STRASSE'",
                "match = a.casefold() == b.casefold()\nfolded = a.casefold()\nprint(f'match={match} folded={folded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "match=True" in nb_runner.get_output(2)
        assert "folded=strasse" in nb_runner.get_output(2)

    def test_unicode_normalize(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import unicodedata\ns1 = 'caf\\u00e9'\ns2 = 'cafe\\u0301'",
                "eq_raw = s1 == s2\nn1 = unicodedata.normalize('NFC', s1)\nn2 = unicodedata.normalize('NFC', s2)\neq_norm = n1 == n2\nprint(f'eq_raw={eq_raw} eq_norm={eq_norm}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "eq_raw=False" in nb_runner.get_output(2)
        assert "eq_norm=True" in nb_runner.get_output(2)

    def test_casefold_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "word = 'Hello'",
                "lower = word.lower()\ncfold = word.casefold()\nprint(f'lower={lower} cfold={cfold}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lower=hello" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "word = 'Straße'")
        nb_runner.run_all()
        assert "cfold=strasse" in nb_runner.get_output(2)


# String chain operations interaction tests.
# Tests split→join, replace→strip chains with cache invalidation.
@pytest.mark.integration
class TestStringChainOpsInteraction:
    """Test string chain operations with cache invalidation."""

    def test_split_upper_join_edit(self, nb_runner):
        """Editing input string should propagate through split/upper/join."""
        nb_runner.create_notebook(
            [
                "raw = 'hello world python coding'",
                "words = raw.split()",
                "upper_words = [w.upper() for w in words]",
                "result = '-'.join(upper_words)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=HELLO-WORLD-PYTHON-CODING" in out

        nb_runner.set_cell_source(1, "raw = 'foo bar baz'")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=FOO-BAR-BAZ" in out

    def test_replace_chain_edit(self, nb_runner):
        """Editing replace targets should propagate."""
        nb_runner.create_notebook(
            [
                "text = 'Hello, World! Hello, Python!'",
                "step1 = text.replace('Hello', 'Hi')",
                "step2 = step1.replace('!', '.')",
                "print(f'out={step2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "out=Hi, World. Hi, Python." in out

        nb_runner.set_cell_source(2, "step1 = text.replace('Hello', 'Greetings')")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "out=Greetings, World. Greetings, Python." in out

    def test_strip_format_edit(self, nb_runner):
        """Editing whitespace handling should propagate."""
        nb_runner.create_notebook(
            [
                "raw_lines = ['  Alice  ', '  Bob  ', '  Charlie  ']",
                "cleaned = [line.strip() for line in raw_lines]",
                "formatted = [f'[{name}]' for name in cleaned]",
                "result = ', '.join(formatted)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=[Alice], [Bob], [Charlie]" in out

        nb_runner.set_cell_source(1, "raw_lines = ['  X  ', '  Y  ']")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=[X], [Y]" in out


# Interaction test: string join with generator expressions.
# Tests str.join with various iterables including generators,
# conditional joins, and cross-cell string building.
class TestStringJoinGenerator:
    """Test str.join with generator expressions across cells."""

    def test_join_generators(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: join with generators
                "nums = list(range(1, 6))\ncsv_line = ','.join(str(n) for n in nums)\ndashed = '-'.join(f'{n:02d}' for n in nums)\nprint(f'csv={csv_line}')\nprint(f'dashed={dashed}')",
                # Cell 2: conditional join
                "words = ['hello', 'world', 'foo', 'bar', 'baz']\nlong_words = ' '.join(w.upper() for w in words if len(w) > 3)\nprint(f'long={long_words}')",
                # Cell 3: nested join
                "matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\ntable = '\\n'.join(' | '.join(str(c) for c in row) for row in matrix)\nprint(f'table:\\n{table}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "csv=1,2,3,4,5" in out1
        assert "dashed=01-02-03-04-05" in out1
        out2 = nb_runner.get_output(2)
        assert "long=HELLO WORLD" in out2
        out3 = nb_runner.get_output(3)
        assert "1 | 2 | 3" in out3
        assert "7 | 8 | 9" in out3

    def test_join_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = ['apple', 'banana', 'cherry']\nresult = ', '.join(items)\nprint(f'result={result}')",
                "upper_result = result.upper()\nword_count = len(result.split(', '))\nprint(f'upper={upper_result}')\nprint(f'count={word_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=apple, banana, cherry" in nb_runner.get_output(1)
        assert "count=3" in nb_runner.get_output(2)

        # Add more items
        nb_runner.set_cell_source(
            1,
            "items = ['apple', 'banana', 'cherry', 'date', 'elderberry']\nresult = ', '.join(items)\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "count=5" in nb_runner.get_output(2)

    def test_join_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "path_parts = ['usr', 'local', 'bin', 'python']\npath = '/'.join(path_parts)\nprint(f'path=/{path}')",
                "depth = path.count('/')\nprint(f'depth={depth}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "path=/usr/local/bin/python" in nb_runner.get_output(1)
        assert "depth=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "depth=3" in nb_runner.get_output(2)


# String manipulation and formatting edit tests.
#
# Tests editing cells with string formatting, regex, and text
# processing patterns.
@pytest.mark.upstream
class TestStringManipEdits:
    """Editing string manipulation patterns."""

    def test_edit_join_separator(self, nb_runner):
        """Edit the separator in a join operation."""
        nb_runner.create_notebook(
            [
                "words = ['hello', 'world', 'python']",
                "joined = ', '.join(words)\nprint(f'joined = {joined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "joined = hello, world, python" in nb_runner.get_output(2)

        # Change separator
        nb_runner.set_cell_source(2, "joined = ' | '.join(words)\nprint(f'joined = {joined}')")
        nb_runner.run_all()
        assert "joined = hello | world | python" in nb_runner.get_output(2)

    def test_edit_string_split_and_rejoin(self, nb_runner):
        """Edit a split-transform-rejoin pipeline."""
        nb_runner.create_notebook(
            [
                "text = 'hello world python'",
                "parts = text.split()\ntransformed = [p.upper() for p in parts]\nresult = '-'.join(transformed)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HELLO-WORLD-PYTHON" in nb_runner.get_output(2)

        # Change transform to title case
        nb_runner.set_cell_source(
            2,
            "parts = text.split()\ntransformed = [p.title() for p in parts]\nresult = ' '.join(transformed)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = Hello World Python" in nb_runner.get_output(2)

    def test_edit_regex_pattern(self, nb_runner):
        """Edit a regex pattern."""
        nb_runner.create_notebook(
            [
                "import re\ntext = 'The price is $42.50 and tax is $3.25'",
                "prices = re.findall(r'\\$[\\d.]+', text)\nprint(f'prices = {prices}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "$42.50" in nb_runner.get_output(2)
        assert "$3.25" in nb_runner.get_output(2)

        # Change to find only dollar amounts over $10
        nb_runner.set_cell_source(
            2, "prices = [p for p in re.findall(r'\\$[\\d.]+', text) if float(p[1:]) > 10]\nprint(f'prices = {prices}')"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "$42.50" in out
        assert "$3.25" not in out


class TestStringMethodChain:
    """string methods chain and text processing edits."""

    def test_string_chain_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = '  Hello, World!  '",
                "processed = text.strip().lower().replace(',', '').replace('!', '')\nwords = processed.split()\nprint(f'words={words}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words=['hello', 'world']" in nb_runner.get_output(2)

    def test_string_chain_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'foo-bar-baz'",
                "parts = text.split('-')\nresult = '_'.join(p.upper() for p in parts)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=FOO_BAR_BAZ" in nb_runner.get_output(2)
        # Edit input
        nb_runner.set_cell_source(1, "text = 'alpha-beta'")
        nb_runner.run_all()
        assert "result=ALPHA_BETA" in nb_runner.get_output(2)

    def test_multiline_string_processing(self, nb_runner):
        nb_runner.create_notebook(
            [
                "lines = 'line1\\nline2\\nline3\\nline2\\nline1'",
                "unique = list(dict.fromkeys(lines.split('\\n')))\ncount = len(unique)\nprint(f'unique={unique} count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "unique=['line1', 'line2', 'line3']" in out
        assert "count=3" in out


class TestStringPartitionJoin:
    """string partition, join patterns, and format_map."""

    def test_partition(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'key=value=extra'",
                "key, sep, rest = text.partition('=')\nprint(f'key={key} rest={rest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "key=key rest=value=extra" in nb_runner.get_output(2)

    def test_join_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['hello', 'beautiful', 'world']",
                "sentence = ' '.join(words)\ncsv_line = ','.join(words)\nprint(f'sentence={sentence}')\nprint(f'csv={csv_line}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sentence=hello beautiful world" in nb_runner.get_output(2)
        assert "csv=hello,beautiful,world" in nb_runner.get_output(2)
        # Edit words
        nb_runner.set_cell_source(1, "words = ['foo', 'bar']")
        nb_runner.run_all()
        assert "sentence=foo bar" in nb_runner.get_output(2)
        assert "csv=foo,bar" in nb_runner.get_output(2)

    def test_format_map_pattern(self, nb_runner):
        nb_runner.create_notebook(
            [
                "template = '{name} scored {score} points'\ndata = {'name': 'Alice', 'score': 95}",
                "result = template.format_map(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Alice scored 95 points" in nb_runner.get_output(2)


# Interaction test: string partition and rpartition methods.
# Tests str.partition, str.rpartition for splitting around separators,
# cross-cell string processing pipelines.
class TestStringPartitionRpartition:
    """Test string partition and rpartition across cells."""

    def test_partition_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: partition from left
                "url = 'https://example.com/path/to/resource'\nscheme, sep, rest = url.partition('://')\nprint(f'scheme={scheme}')\nprint(f'rest={rest}')",
                # Cell 2: rpartition from right
                "path_part, slash, filename = rest.rpartition('/')\nprint(f'path_part={path_part}')\nprint(f'filename={filename}')",
                # Cell 3: combine results
                "full_path = f'{scheme}://{path_part}'\nprint(f'full_path={full_path}')\nprint(f'file={filename}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "scheme=https" in out1
        assert "rest=example.com/path/to/resource" in out1
        out2 = nb_runner.get_output(2)
        assert "path_part=example.com/path/to" in out2
        assert "filename=resource" in out2
        out3 = nb_runner.get_output(3)
        assert "full_path=https://example.com/path/to" in out3
        assert "file=resource" in out3

    def test_partition_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "email = 'user@example.com'\nlocal, at, domain = email.partition('@')\nprint(f'local={local}')\nprint(f'domain={domain}')",
                "result = f'{local} at {domain}'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=user at example.com" in nb_runner.get_output(2)

        # Edit email
        nb_runner.set_cell_source(
            1,
            "email = 'admin@company.org'\nlocal, at, domain = email.partition('@')\nprint(f'local={local}')\nprint(f'domain={domain}')",
        )
        nb_runner.run_cells([1, 2])
        assert "result=admin at company.org" in nb_runner.get_output(2)

    def test_partition_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "line = 'key=value=extra'\nk, eq, v = line.partition('=')\nprint(f'key={k}')\nprint(f'value={v}')",
                "info = f'{k} -> {v}'\nprint(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "key=key" in nb_runner.get_output(1)
        assert "value=value=extra" in nb_runner.get_output(1)
        assert "info=key -> value=extra" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "info=key -> value=extra" in nb_runner.get_output(2)


class TestStringPartitionRsplit:
    """string partition and rsplit operations."""

    def test_partition(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'user@domain.com'",
                "before, sep, after = text.partition('@')\nprint(f'before={before} sep={sep} after={after}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "before=user" in out
        assert "sep=@" in out
        assert "after=domain.com" in out

    def test_rsplit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "path = 'a/b/c/d/e'",
                "parts = path.rsplit('/', maxsplit=2)\nprint(f'parts={parts}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "parts=['a/b/c', 'd', 'e']" in nb_runner.get_output(2)

    def test_partition_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "line = 'key=value'",
                "k, _, v = line.partition('=')\nprint(f'k={k} v={v}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "k=key" in nb_runner.get_output(2)
        assert "v=value" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "line = 'name:Alice'")
        nb_runner.set_cell_source(2, "k, _, v = line.partition(':')\nprint(f'k={k} v={v}')")
        nb_runner.run_all()
        assert "k=name" in nb_runner.get_output(2)
        assert "v=Alice" in nb_runner.get_output(2)


class TestStringValidation:
    """string isdigit/isalpha/isalnum validation."""

    def test_is_methods(self, nb_runner):
        nb_runner.create_notebook(
            [
                "s1 = '12345'\ns2 = 'hello'\ns3 = 'hello123'\ns4 = 'Hello World'",
                "d = s1.isdigit()\na = s2.isalpha()\nan = s3.isalnum()\nsp = s4.isalpha()\nprint(f'd={d} a={a} an={an} sp={sp}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=True" in out
        assert "a=True" in out
        assert "an=True" in out
        assert "sp=False" in out

    def test_isupper_islower(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = 'HELLO'\nb = 'hello'\nc = 'Hello'",
                "r = f'{a.isupper()},{b.islower()},{c.istitle()}'\nprint(f'r={r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=True,True,True" in nb_runner.get_output(2)

    def test_validation_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "token = 'abc123'",
                "valid = token.isalnum()\nprint(f'valid={valid}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid=True" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "token = 'abc 123'")
        nb_runner.run_all()
        assert "valid=False" in nb_runner.get_output(2)


class TestStringRemovePrefixSuffix:
    """string removeprefix and removesuffix (3.9+)."""

    def test_removeprefix(self, nb_runner):
        nb_runner.create_notebook(
            [
                "text = 'TestCaseExample'",
                "no_prefix = text.removeprefix('Test')\nno_miss = text.removeprefix('Foo')\nprint(f'no_prefix={no_prefix} no_miss={no_miss}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "no_prefix=CaseExample" in nb_runner.get_output(2)
        assert "no_miss=TestCaseExample" in nb_runner.get_output(2)

    def test_remove_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "s = 'prefix_content_suffix'",
                "r = s.removeprefix('prefix_').removesuffix('_suffix')\nprint(f'r={r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=content" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "s = 'prefix_hello_world_suffix'")
        nb_runner.run_all()
        assert "r=hello_world" in nb_runner.get_output(2)


class TestStringTranslate:
    """string translate and maketrans patterns."""

    def test_translate_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "cipher_shift = 3",
                "import string\nlower = string.ascii_lowercase\nshifted = lower[cipher_shift:] + lower[:cipher_shift]\ntable = str.maketrans(lower, shifted)\nencoded = 'hello'.translate(table)\nprint(f'encoded={encoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded=khoor" in nb_runner.get_output(2)
        # Edit shift
        nb_runner.set_cell_source(1, "cipher_shift = 1")
        nb_runner.run_all()
        assert "encoded=ifmmp" in nb_runner.get_output(2)

    def test_remove_punctuation(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import string\ntext = 'Hello, World! How are you?'",
                "no_punct = text.translate(str.maketrans('', '', string.punctuation))\nwords = no_punct.split()\nprint(f'words={words}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words=['Hello', 'World', 'How', 'are', 'you']" in nb_runner.get_output(2)
