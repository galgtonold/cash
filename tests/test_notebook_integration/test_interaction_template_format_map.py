"""
Batch 323: string.Template and format_map patterns with caching.
Tests Template substitution, safe_substitute, format_map, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestTemplateFormatMap:
    """Test string Template and format_map caching."""

    def test_template_substitute(self, nb_runner):
        """string.Template substitution with caching."""
        nb_runner.create_notebook([
            "from string import Template",
            "tmpl = Template('Hello, $name! You are $age years old.')",
            "data = {'name': 'Alice', 'age': 30}",
            "result = tmpl.substitute(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Hello, Alice! You are 30 years old." in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "Hello, Alice!" in out2

    def test_template_edit_data(self, nb_runner):
        """Edit template data, verify output changes."""
        nb_runner.create_notebook([
            "from string import Template",
            "tmpl = Template('$item costs $$${price}')",
            "data = {'item': 'Book', 'price': '25'}",
            "result = tmpl.substitute(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Book costs $25" in out

        nb_runner.set_cell_source(3, "data = {'item': 'Pen', 'price': '5'}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "Pen costs $5" in out2

    def test_format_map(self, nb_runner):
        """str.format_map with caching."""
        nb_runner.create_notebook([
            "template = '{city} has {pop} people'",
            "data = {'city': 'NYC', 'pop': '8M'}",
            "result = template.format_map(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "NYC has 8M people" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "NYC has 8M people" in out2
