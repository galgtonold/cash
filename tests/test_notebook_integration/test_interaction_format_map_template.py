"""Batch 489: string formatting with format_map and template."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringFormatMapTemplate:
    def test_format_map_dict(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import defaultdict",
            "template = '{name} is {age} years old from {city}'\ndata = defaultdict(lambda: 'N/A', name='Alice', age='30')\nresult = template.format_map(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Alice is 30 years old from N/A" in nb_runner.get_output(2)

    def test_string_template_safe(self, nb_runner):
        nb_runner.create_notebook([
            "from string import Template",
            "t = Template('$name owes $$${amount}')\nresult = t.safe_substitute(name='Bob')\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Bob owes $${amount}" in nb_runner.get_output(2)

    def test_format_map_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = {'x': 10, 'y': 20}",
            "result = '{x}+{y}'.format_map(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10+20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "data = {'x': 100, 'y': 200}")
        nb_runner.run_all()
        assert "result=100+200" in nb_runner.get_output(2)
