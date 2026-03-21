"""Batch 159 – Print and display output interaction tests.

Tests where print/display formatting changes, output cells
are edited, and print modes are toggled.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestPrintFormatEdits:
    """Edit print formatting."""

    def test_edit_print_format_style(self, nb_runner):
        """Change from f-string to format()."""
        nb_runner.create_notebook([
            "x = 42\ny = 3.14",
            "print(f'x={x}, y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=42, y=3.14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "print('x={}, y={:.1f}'.format(x, y))")
        nb_runner.run_all()
        assert "x=42, y=3.1" in nb_runner.get_output(2)

    def test_add_more_prints(self, nb_runner):
        """Add additional print statements."""
        nb_runner.create_notebook([
            "a = 1\nb = 2\nc = 3",
            "print(f'a = {a}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 1" in nb_runner.get_output(2)

        # Add more prints
        nb_runner.set_cell_source(
            2, "print(f'a = {a}')\nprint(f'b = {b}')\nprint(f'c = {c}')"
        )
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "a = 1" in output
        assert "b = 2" in output
        assert "c = 3" in output

    def test_change_output_variable(self, nb_runner):
        """Change which variable is printed."""
        nb_runner.create_notebook([
            "first = 'hello'\nsecond = 'world'",
            "print(first)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hello" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "print(second)")
        nb_runner.run_all()
        assert "world" in nb_runner.get_output(2)


class TestOutputCollectionEdits:
    """Build up output collections, edit formatting."""

    def test_list_to_table_format(self, nb_runner):
        """Change from list output to table-like format."""
        nb_runner.create_notebook([
            "data = [('A', 1), ('B', 2), ('C', 3)]",
            "for name, val in data:\n    print(f'{name}: {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "A: 1" in output
        assert "C: 3" in output

        # Change to tabular
        nb_runner.set_cell_source(
            2,
            "header = f'{\"Name\":>10} | {\"Value\":>5}'\nprint(header)\nfor name, val in data:\n    print(f'{name:>10} | {val:>5}')",
        )
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "Name" in output
        assert "Value" in output

    def test_json_output_edit(self, nb_runner):
        """Change between JSON and plain output."""
        nb_runner.create_notebook([
            "import json\nresult = {'status': 'ok', 'count': 42}",
            "print(json.dumps(result))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert '"status": "ok"' in output or '"status":"ok"' in output

        # Switch to pretty print
        nb_runner.set_cell_source(2, "print(json.dumps(result, indent=2))")
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert '"status": "ok"' in output
