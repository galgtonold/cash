"""Batch 224 – String template/formatting interaction tests.

Tests editing cells with various string formatting
approaches and verifying correct output.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestTemplatePatternEdits:
    """Editing string template/formatting patterns."""

    def test_edit_format_template(self, nb_runner):
        """Edit data used in string format template."""
        nb_runner.create_notebook([
            "name = 'Alice'\nrole = 'engineer'",
            "msg = '{} is a {}'.format(name, role)\nprint(msg)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice is a engineer" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "name = 'Bob'\nrole = 'designer'")
        nb_runner.run_all()
        assert "Bob is a designer" in nb_runner.get_output(2)

    def test_edit_template_string(self, nb_runner):
        """Edit a Template string pattern."""
        nb_runner.create_notebook([
            "from string import Template\ntmpl = Template('Hello, $name! You have $count messages.')",
            "result = tmpl.substitute(name='Alice', count=5)\nprint(result)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello, Alice! You have 5 messages." in nb_runner.get_output(2)

        # Change template
        nb_runner.set_cell_source(1, "from string import Template\ntmpl = Template('Hi $name, $count items in cart.')")
        nb_runner.run_all()
        assert "Hi Alice, 5 items in cart." in nb_runner.get_output(2)

    def test_edit_format_spec(self, nb_runner):
        """Edit format specification."""
        nb_runner.create_notebook([
            "value = 1234.5678",
            "formatted = f'{value:,.2f}'\nprint(f'formatted = {formatted}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted = 1,234.57" in nb_runner.get_output(2)

        # Change value
        nb_runner.set_cell_source(1, "value = 9876543.21")
        nb_runner.run_all()
        assert "formatted = 9,876,543.21" in nb_runner.get_output(2)

    def test_edit_multiline_template(self, nb_runner):
        """Edit data used in multiline template."""
        nb_runner.create_notebook([
            "items = [('apple', 2), ('banana', 3)]",
            "lines = []\nfor name, qty in items:\n    lines.append(f'{name}: {qty}')\noutput = ', '.join(lines)\nprint(output)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "apple: 2, banana: 3" in nb_runner.get_output(2)

        # Change items
        nb_runner.set_cell_source(1, "items = [('x', 10), ('y', 20), ('z', 30)]")
        nb_runner.run_all()
        assert "x: 10, y: 20, z: 30" in nb_runner.get_output(2)
