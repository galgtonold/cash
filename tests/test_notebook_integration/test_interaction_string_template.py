"""Batch 252 – String template and formatting edit patterns.

Tests various string formatting approaches with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringTemplateEdits:
    """String template and formatting edit propagation."""

    def test_format_string_template(self, nb_runner):
        """Edit format template, output updates."""
        nb_runner.create_notebook([
            "template = '{name} has {count} items'",
            "name = 'Alice'\ncount = 5",
            "msg = template.format(name=name, count=count)\nprint(f'msg = {msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = Alice has 5 items" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "template = 'User {name}: {count} items remaining'")
        nb_runner.run_all()
        assert "msg = User Alice: 5 items remaining" in nb_runner.get_output(3)

    def test_join_pattern_edit(self, nb_runner):
        """Edit separator in join operation."""
        nb_runner.create_notebook([
            "sep = ', '",
            "words = ['hello', 'world', 'python']\nresult = sep.join(words)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = hello, world, python" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "sep = ' | '")
        nb_runner.run_all()
        assert "result = hello | world | python" in nb_runner.get_output(2)

    def test_multiline_string_edit(self, nb_runner):
        """Edit multiline string template."""
        nb_runner.create_notebook([
            "header = 'Report'\nfooter = 'End'",
            "body = 'Data: 42'",
            "doc = f'{header}\\n{body}\\n{footer}'\nprint(doc)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Report" in out
        assert "Data: 42" in out
        assert "End" in out

        nb_runner.set_cell_source(1, "header = '=== Summary ==='\nfooter = '=== Done ==='")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "=== Summary ===" in out2
        assert "=== Done ===" in out2

    def test_regex_sub_edit(self, nb_runner):
        """Edit regex pattern, substitution updates."""
        nb_runner.create_notebook([
            "import re\npattern = r'\\d+'",
            "text = 'item1 and item22 plus item333'\nresult = re.sub(pattern, '#', text)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = item# and item# plus item#" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "import re\npattern = r'[a-z]+'")
        nb_runner.run_all()
        assert "result = #1 # #22 # #333" in nb_runner.get_output(2)
