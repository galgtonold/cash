"""Batch 171 – String manipulation chain interaction tests.

Tests editing string processing pipelines across cells
with transformations, formatting, and parsing.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestStringChainEdits:
    """String processing chain edits."""

    def test_edit_string_transform(self, nb_runner):
        """Edit a string transformation in a chain."""
        nb_runner.create_notebook([
            "raw = '  Hello, World!  '  # raw input string",
            "cleaned = raw.strip()",
            "result = cleaned.lower()\nprint(f'result = [{result}]')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [hello, world!]" in nb_runner.get_output(3)

        # Change to upper
        nb_runner.set_cell_source(3, "result = cleaned.upper()\nprint(f'result = [{result}]')")
        nb_runner.run_all()
        assert "result = [HELLO, WORLD!]" in nb_runner.get_output(3)

    def test_edit_source_string(self, nb_runner):
        """Edit the source string, verify chain updates."""
        nb_runner.create_notebook([
            "text = 'python is great'  # source text",
            "words = text.split()",
            "result = '-'.join(words)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = python-is-great" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "text = 'cash is amazing'  # source text changed")
        nb_runner.run_all()
        assert "result = cash-is-amazing" in nb_runner.get_output(3)

    def test_edit_join_separator(self, nb_runner):
        """Edit the separator in a join operation."""
        nb_runner.create_notebook([
            "parts = ['2024', '01', '15']  # date parts",
            "date_str = '-'.join(parts)\nprint(f'date = {date_str}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "date = 2024-01-15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "date_str = '/'.join(parts)\nprint(f'date = {date_str}')"
        )
        nb_runner.run_all()
        assert "date = 2024/01/15" in nb_runner.get_output(2)


class TestStringParsingEdits:
    """String parsing with edits."""

    def test_edit_regex_pattern(self, nb_runner):
        """Edit a regex pattern used for parsing."""
        nb_runner.create_notebook([
            "import re",
            "text = 'price: $42.50, tax: $3.50'  # text to parse",
            "matches = re.findall(r'\\$([\\d.]+)', text)\nprint(f'matches = {matches}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42.50" in nb_runner.get_output(3)
        assert "3.50" in nb_runner.get_output(3)

        # Change to find only integers
        nb_runner.set_cell_source(
            3, "matches = re.findall(r'\\d+', text)\nprint(f'matches = {matches}')"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "42" in out
        assert "50" in out

    def test_edit_string_template(self, nb_runner):
        """Edit a string format template."""
        nb_runner.create_notebook([
            "name = 'Alice'\nage = 30",
            "msg = f'{name} is {age} years old'\nprint(f'msg = {msg}')",
        ])
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
        nb_runner.create_notebook([
            "lines = ['line1', 'line2', 'line3']  # lines data",
            "text = '\\n'.join(lines)\nline_count = len(text.splitlines())\nprint(f'lines = {line_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lines = 3" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1, "lines = ['a', 'b', 'c', 'd', 'e']  # lines data more"
        )
        nb_runner.run_all()
        assert "lines = 5" in nb_runner.get_output(2)
