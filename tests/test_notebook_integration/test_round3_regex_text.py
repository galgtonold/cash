"""Batch 51: Regex & text processing — cash caching with re, textwrap, string ops."""
import textwrap
import pytest


@pytest.mark.stress
class TestRegexPatterns:
    """Test regex compilation and matching across cells."""

    def test_compiled_regex_cached(self, nb_runner):
        """Compiled regex pattern cached across cells."""
        nb_runner.create_notebook([
            "import re",
            textwrap.dedent("""\
                email_pattern = re.compile(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+')
                test_text = "Contact alice@example.com or bob@test.org for info"
                emails = email_pattern.findall(test_text)
                print(f"emails={sorted(emails)}")
            """),
            textwrap.dedent("""\
                # Use same compiled pattern
                more_text = "No emails here! But admin@company.net is one"
                more_emails = email_pattern.findall(more_text)
                all_emails = sorted(emails + more_emails)
                print(f"all={all_emails}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "alice@example.com" in nb_runner.get_output(2)
        assert "admin@company.net" in nb_runner.get_output(3)

    def test_regex_groups_and_substitution(self, nb_runner):
        """Named groups and substitution across cells."""
        nb_runner.create_notebook([
            "import re",
            textwrap.dedent("""\
                date_pat = re.compile(r'(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})')
                text = "Dates: 2024-01-15 and 2024-12-25"
                matches = [m.groupdict() for m in date_pat.finditer(text)]
                print(f"count={len(matches)} first_year={matches[0]['year']}")
            """),
            textwrap.dedent("""\
                # Substitute format
                reformatted = date_pat.sub(r'\\g<day>/\\g<month>/\\g<year>', text)
                print(f"reformatted={reformatted}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2 first_year=2024" in nb_runner.get_output(2)
        assert "15/01/2024" in nb_runner.get_output(3)

    def test_regex_split_pattern(self, nb_runner):
        """Regex split with pattern changes."""
        nb_runner.create_notebook([
            "import re",
            textwrap.dedent("""\
                splitter = re.compile(r'[,;\\s]+')
                data = "apple, banana; cherry  date,elderberry"
                tokens = splitter.split(data)
                print(f"tokens={tokens}")
            """),
            textwrap.dedent("""\
                count = len(tokens)
                upper_tokens = [t.upper() for t in tokens]
                print(f"count={count} upper={upper_tokens}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "apple" in nb_runner.get_output(2)
        out = nb_runner.get_output(3)
        assert "count=5" in out
        assert "APPLE" in out


@pytest.mark.stress
class TestTextProcessing:
    """Test text processing patterns."""

    def test_string_template(self, nb_runner):
        """String Template across cells."""
        nb_runner.create_notebook([
            "from string import Template",
            textwrap.dedent("""\
                tmpl = Template("Hello $name, you have $$${amount} in your account")
                msg1 = tmpl.substitute(name="Alice", amount="100")
                print(f"msg1={msg1}")
            """),
            textwrap.dedent("""\
                msg2 = tmpl.substitute(name="Bob", amount="250")
                print(f"msg2={msg2}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello Alice" in nb_runner.get_output(2)
        assert "Hello Bob" in nb_runner.get_output(3)

    def test_text_word_frequency(self, nb_runner):
        """Word frequency analysis across cells."""
        nb_runner.create_notebook([
            "from collections import Counter\nimport re",
            textwrap.dedent("""\
                text = "the quick brown fox jumps over the lazy dog the fox"
                words = re.findall(r'\\w+', text.lower())
                freq = Counter(words)
                print(f"total_words={len(words)} unique={len(freq)}")
            """),
            textwrap.dedent("""\
                top3 = freq.most_common(3)
                print(f"top3={top3}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total_words=11" in nb_runner.get_output(2)
        out = nb_runner.get_output(3)
        assert "the" in out
        assert "3" in out  # "the" appears 3 times

    def test_csv_like_parsing(self, nb_runner):
        """Manual CSV-like parsing across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                raw_data = '''name,age,city
                Alice,30,NYC
                Bob,25,LA
                Charlie,35,Chicago'''

                lines = [l.strip() for l in raw_data.strip().split('\\n')]
                headers = lines[0].split(',')
                rows = [dict(zip(headers, line.split(','))) for line in lines[1:]]
                print(f"headers={headers} row_count={len(rows)}")
            """),
            textwrap.dedent("""\
                ages = [int(r['age']) for r in rows]
                avg_age = sum(ages) / len(ages)
                cities = [r['city'] for r in rows]
                print(f"avg_age={avg_age:.1f} cities={cities}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "row_count=3" in nb_runner.get_output(1)
        assert "avg_age=30.0" in nb_runner.get_output(2)
        assert "NYC" in nb_runner.get_output(2)
