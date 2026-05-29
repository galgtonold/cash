"""Batch 142 – String operations and formatting interaction tests.

Tests where users perform string operations across cells,
edit string content and formatting, and verify caching
handles string changes correctly.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestStringEdits:
    """String manipulation with cell edits."""


    def test_edit_format_string(self, nb_runner):
        """Edit the format string itself."""
        nb_runner.create_notebook([
            "x = 42",
            "msg = f'The answer is {x}'\nprint(msg)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "The answer is 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "msg = f'Value: {x} (hex: {hex(x)})'\nprint(msg)")
        nb_runner.run_all()
        assert "Value: 42 (hex: 0x2a)" in nb_runner.get_output(2)

    def test_string_concatenation_chain(self, nb_runner):
        """Chain of string concatenation, edit source."""
        nb_runner.create_notebook([
            "first = 'Hello'",
            "second = first + ' World'",
            "third = second + '!'\nprint(third)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World!" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "first = 'Goodbye'")
        nb_runner.run_all()
        assert "Goodbye World!" in nb_runner.get_output(3)

    def test_string_method_chain_edit(self, nb_runner):
        """String methods, edit the method call."""
        nb_runner.create_notebook([
            "text = '  Hello World  '",
            "processed = text.strip()\nprint(f'|{processed}|')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "|Hello World|" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "processed = text.strip().upper()\nprint(f'|{processed}|')")
        nb_runner.run_all()
        assert "|HELLO WORLD|" in nb_runner.get_output(2)


class TestStringParsingEdits:
    """String parsing patterns with cell edits."""

    def test_split_and_join_edit_delimiter(self, nb_runner):
        """Split/join with delimiter change."""
        nb_runner.create_notebook([
            "raw = 'a,b,c,d'",
            "parts = raw.split(',')\nresult = '-'.join(parts)\nprint(result)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a-b-c-d" in nb_runner.get_output(2)

        # Change join delimiter
        nb_runner.set_cell_source(
            2, "parts = raw.split(',')\nresult = ' | '.join(parts)\nprint(result)"
        )
        nb_runner.run_all()
        assert "a | b | c | d" in nb_runner.get_output(2)

    def test_regex_pattern_edit(self, nb_runner):
        """Regex pattern change."""
        nb_runner.create_notebook([
            "import re\ntext = 'abc 123 def 456'",
            "nums = re.findall(r'\\d+', text)\nprint(f'nums = {nums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "nums = ['123', '456']" in nb_runner.get_output(2)

        # Change to match words
        nb_runner.set_cell_source(
            2, "words = re.findall(r'[a-z]+', text)\nprint(f'words = {words}')"
        )
        nb_runner.run_all()
        assert "words = ['abc', 'def']" in nb_runner.get_output(2)
