"""
Interaction test: string Template substitution.
Tests string.Template with safe_substitute, missing keys,
custom delimiters, and cross-cell template pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringTemplateSubstitute:
    """Test string Template substitution across cells."""

    def test_template_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: basic template
            "from string import Template\ntpl = Template('Hello $name, you are $age years old')\nresult = tpl.substitute(name='Alice', age=30)\nprint(f'result={result}')",
            # Cell 2: safe_substitute with missing key
            "tpl2 = Template('$greeting $name, welcome to $place')\nsafe = tpl2.safe_substitute(greeting='Hi', name='Bob')\nprint(f'safe={safe}')",
            # Cell 3: template from cell 1 data
            "report_tpl = Template('Report: $name is $age')\nreport = report_tpl.substitute(name='Alice', age=30)\nprint(f'report={report}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "result=Hello Alice, you are 30 years old" in out1
        out2 = nb_runner.get_output(2)
        assert "safe=Hi Bob, welcome to $place" in out2
        out3 = nb_runner.get_output(3)
        assert "report=Report: Alice is 30" in out3

    def test_template_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from string import Template\ntpl = Template('$item costs $$${price}')\nresult = tpl.substitute(item='Widget', price='9.99')\nprint(f'result={result}')",
            "msg = f'Buy now: {result}'\nprint(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "Widget" in out1 and "9.99" in out1

        # Edit template
        nb_runner.set_cell_source(1, "from string import Template\ntpl = Template('$item on sale for $$${price}')\nresult = tpl.substitute(item='Gadget', price='4.99')\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        out2 = nb_runner.get_output(2)
        assert "Gadget" in out2 and "4.99" in out2

    def test_template_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from string import Template\ntpl = Template('$x + $y = $z')\neq = tpl.substitute(x='2', y='3', z='5')\nprint(f'eq={eq}')",
            "length = len(eq)\nprint(f'length={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "eq=2 + 3 = 5" in nb_runner.get_output(1)
        assert "length=9" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "length=9" in nb_runner.get_output(2)
