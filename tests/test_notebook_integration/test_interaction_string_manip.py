"""Batch 232 – String manipulation and formatting edit tests.

Tests editing cells with string formatting, regex, and text
processing patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestStringManipEdits:
    """Editing string manipulation patterns."""


    def test_edit_join_separator(self, nb_runner):
        """Edit the separator in a join operation."""
        nb_runner.create_notebook([
            "words = ['hello', 'world', 'python']",
            "joined = ', '.join(words)\nprint(f'joined = {joined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "joined = hello, world, python" in nb_runner.get_output(2)

        # Change separator
        nb_runner.set_cell_source(2, "joined = ' | '.join(words)\nprint(f'joined = {joined}')")
        nb_runner.run_all()
        assert "joined = hello | world | python" in nb_runner.get_output(2)

    def test_edit_string_split_and_rejoin(self, nb_runner):
        """Edit a split-transform-rejoin pipeline."""
        nb_runner.create_notebook([
            "text = 'hello world python'",
            "parts = text.split()\ntransformed = [p.upper() for p in parts]\nresult = '-'.join(transformed)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HELLO-WORLD-PYTHON" in nb_runner.get_output(2)

        # Change transform to title case
        nb_runner.set_cell_source(2, "parts = text.split()\ntransformed = [p.title() for p in parts]\nresult = ' '.join(transformed)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = Hello World Python" in nb_runner.get_output(2)

    def test_edit_regex_pattern(self, nb_runner):
        """Edit a regex pattern."""
        nb_runner.create_notebook([
            "import re\ntext = 'The price is $42.50 and tax is $3.25'",
            "prices = re.findall(r'\\$[\\d.]+', text)\nprint(f'prices = {prices}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "$42.50" in nb_runner.get_output(2)
        assert "$3.25" in nb_runner.get_output(2)

        # Change to find only dollar amounts over $10
        nb_runner.set_cell_source(2, "prices = [p for p in re.findall(r'\\$[\\d.]+', text) if float(p[1:]) > 10]\nprint(f'prices = {prices}')")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "$42.50" in out
        assert "$3.25" not in out
