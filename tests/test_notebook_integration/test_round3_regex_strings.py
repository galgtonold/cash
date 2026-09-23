"""regex and string processing patterns."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


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
