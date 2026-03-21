"""Batch 88 – regex and string processing patterns."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestRegexPatterns:
    """Regular expression caching scenarios."""

    def test_compiled_regex(self, nb_runner):
        """Compiled regex patterns with findall/sub."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import re
                text = "Call 555-1234 or 555-5678. Email: user@example.com or admin@test.org"
                phone_pat = re.compile(r'\\d{3}-\\d{4}')
                email_pat = re.compile(r'[\\w.]+@[\\w.]+')
                phones = phone_pat.findall(text)
                emails = email_pat.findall(text)
            """),
            "print(f'phones={phones}')\nprint(f'emails={emails}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "555-1234" in out
        assert "555-5678" in out
        assert "user@example.com" in out
        assert "admin@test.org" in out

    def test_regex_groups(self, nb_runner):
        """Named groups in regex."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import re
                log_line = '2024-01-15 10:30:45 ERROR Database connection failed'
                pattern = re.compile(r'(?P<date>[\\d-]+) (?P<time>[\\d:]+) (?P<level>\\w+) (?P<msg>.+)')
                m = pattern.match(log_line)
                parsed = m.groupdict() if m else {}
            """),
            "print(f'parsed={parsed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "2024-01-15" in out
        assert "ERROR" in out
        assert "Database connection failed" in out

    def test_regex_substitution_propagation(self, nb_runner):
        """Regex substitution with upstream change propagation."""
        nb_runner.create_notebook([
            "replacement = 'REDACTED'",
            textwrap.dedent("""\
                import re
                text = "SSN: 123-45-6789, Phone: 555-1234"
                ssn_pat = re.compile(r'\\d{3}-\\d{2}-\\d{4}')
                cleaned = ssn_pat.sub(replacement, text)
            """),
            "print(f'cleaned={cleaned}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "REDACTED" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "replacement = '***-**-****'")
        nb_runner.run_cells([1, 2, 3])
        assert "***-**-****" in nb_runner.get_output(3)


class TestStringProcessing:
    """Advanced string processing patterns."""

    def test_string_template(self, nb_runner):
        """string.Template safe substitution."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from string import Template
                tmpl = Template('Hello $name, your balance is $$${amount}.')
                result = tmpl.safe_substitute(name='Alice', amount='1500')
            """),
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Hello Alice" in out
        assert "$1500" in out

    def test_textwrap_processing(self, nb_runner):
        """textwrap fill/dedent/indent."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import textwrap
                long_text = "This is a very long line of text that should be wrapped at a certain width to make it more readable in terminal output."
                wrapped = textwrap.fill(long_text, width=40)
                indented = textwrap.indent(wrapped, '  > ')
                line_count = len(indented.strip().split('\\n'))
            """),
            "print(f'lines={line_count}')\nprint(indented)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lines=" in out
        assert ">" in out
