"""Batch 189 – Regex and string method chain interaction tests.

Tests editing regex patterns, substitutions, and chained
string method calls.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestRegexEdits:
    """Editing regex patterns."""

    def test_edit_regex_pattern(self, nb_runner):
        """Edit the regex pattern in a search."""
        nb_runner.create_notebook([
            "import re",
            "text = 'Hello World 123'  # regex source",
            "match = re.findall(r'\\d+', text)\nprint(f'match = {match}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "match = ['123']" in nb_runner.get_output(3)

        # Change pattern to words
        nb_runner.set_cell_source(
            3, "match = re.findall(r'[A-Z][a-z]+', text)\nprint(f'match = {match}')"
        )
        nb_runner.run_all()
        assert "match = ['Hello', 'World']" in nb_runner.get_output(3)

    def test_edit_regex_substitution(self, nb_runner):
        """Edit regex substitution."""
        nb_runner.create_notebook([
            "import re",
            "text = 'foo bar baz'  # sub source",
            "result = re.sub(r'\\s+', '-', text)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = foo-bar-baz" in nb_runner.get_output(3)

        # Change replacement
        nb_runner.set_cell_source(
            3, "result = re.sub(r'\\s+', '_', text)\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = foo_bar_baz" in nb_runner.get_output(3)


class TestStringMethodChains:
    """Editing chained string method calls."""

    def test_edit_string_chain(self, nb_runner):
        """Edit a chain of string methods."""
        nb_runner.create_notebook([
            "raw = '  Hello, World!  '  # string chain source",
            "result = raw.strip().lower()\nprint(f'result = [{result}]')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [hello, world!]" in nb_runner.get_output(2)

        # Change chain
        nb_runner.set_cell_source(
            2, "result = raw.strip().upper().replace(' ', '')\nprint(f'result = [{result}]')"
        )
        nb_runner.run_all()
        assert "result = [HELLO,WORLD!]" in nb_runner.get_output(2)

    def test_edit_split_join(self, nb_runner):
        """Edit split/join operations."""
        nb_runner.create_notebook([
            "text = 'a,b,c,d'  # split join source",
            "parts = text.split(',')\njoined = '-'.join(parts)\nprint(f'joined = {joined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "joined = a-b-c-d" in nb_runner.get_output(2)

        # Edit the source
        nb_runner.set_cell_source(1, "text = 'x;y;z'  # split join source v2")
        nb_runner.set_cell_source(
            2, "parts = text.split(';')\njoined = '|'.join(parts)\nprint(f'joined = {joined}')"
        )
        nb_runner.run_all()
        assert "joined = x|y|z" in nb_runner.get_output(2)
