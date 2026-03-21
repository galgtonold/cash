"""
Batch 295: String formatting and template interaction tests.
Tests various string formatting patterns (f-strings, format(), Template)
with cache invalidation when underlying data changes.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringFormattingInteraction:
    """Test string formatting patterns with cache invalidation."""

    def test_format_method_edit(self, nb_runner):
        """Editing data used in str.format() should propagate."""
        nb_runner.create_notebook([
            "name = 'Alice'\nage = 30",
            "template = '{name} is {age} years old'",
            "msg = template.format(name=name, age=age)",
            "print(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Alice is 30 years old" in out

        nb_runner.set_cell_source(1, "name = 'Bob'\nage = 25")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Bob is 25 years old" in out

    def test_string_template_edit(self, nb_runner):
        """Editing data used in string.Template should propagate."""
        nb_runner.create_notebook([
            "from string import Template\nproduct = 'Widget'\nprice = 9.99",
            "t = Template('Buy $product for $$$price')",
            "msg = t.substitute(product=product, price=price)",
            "print(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Buy Widget for $9.99" in out

        nb_runner.set_cell_source(1, "from string import Template\nproduct = 'Gadget'\nprice = 19.99")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Buy Gadget for $19.99" in out

    def test_multiline_format_edit(self, nb_runner):
        """Editing data used in multiline formatting should propagate."""
        nb_runner.create_notebook([
            "items = [('Apple', 3), ('Banana', 5)]",
            "lines = []\nfor name, qty in items:\n    lines.append(f'{name}: {qty}')",
            "report = '\\n'.join(lines)",
            "print(report)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Apple: 3" in out
        assert "Banana: 5" in out

        nb_runner.set_cell_source(1, "items = [('Cherry', 10), ('Date', 7), ('Fig', 2)]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Cherry: 10" in out
        assert "Date: 7" in out
        assert "Fig: 2" in out
