"""What a cell prints or displays, edited: formats, tables, large output, echo and suppression."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestComputeDisplaySplit:
    """Computation and display in separate cells."""

    def test_edit_computation_only(self, nb_runner):
        """Edit computation cell, display cell stays same."""
        nb_runner.create_notebook(
            [
                "values = [1, 2, 3, 4, 5]  # compute data",
                "total = sum(values)\navg = total / len(values)",
                "print(f'total={total} avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15 avg=3.0" in nb_runner.get_output(3)

        # Edit source data
        nb_runner.set_cell_source(1, "values = [10, 20, 30]  # compute data changed")
        nb_runner.run_all()
        assert "total=60 avg=20.0" in nb_runner.get_output(3)

    def test_edit_display_only(self, nb_runner):
        """Edit display cell, computation stays same."""
        nb_runner.create_notebook(
            [
                "nums = [2, 4, 6, 8]  # display nums",
                "s = sum(nums)\nm = max(nums)",
                "print(f'sum={s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum=20" in nb_runner.get_output(3)

        # Change display to show max
        nb_runner.set_cell_source(3, "print(f'max={m}')")
        nb_runner.run_all()
        assert "max=8" in nb_runner.get_output(3)

    def test_edit_both_compute_and_display(self, nb_runner):
        """Edit both computation and display cells."""
        nb_runner.create_notebook(
            [
                "x = 7  # base value for compute+display",
                "squared = x ** 2",
                "print(f'squared = {squared}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squared = 49" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10  # base value updated")
        nb_runner.set_cell_source(2, "cubed = x ** 3")
        nb_runner.set_cell_source(3, "print(f'cubed = {cubed}')")
        nb_runner.run_all()
        assert "cubed = 1000" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestPrintFormatEdits:
    """Edit print formatting."""

    def test_edit_print_format_style(self, nb_runner):
        """Change from f-string to format()."""
        nb_runner.create_notebook(
            [
                "x = 42\ny = 3.14",
                "print(f'x={x}, y={y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=42, y=3.14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "print('x={}, y={:.1f}'.format(x, y))")
        nb_runner.run_all()
        assert "x=42, y=3.1" in nb_runner.get_output(2)

    def test_add_more_prints(self, nb_runner):
        """Add additional print statements."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = 2\nc = 3",
                "print(f'a = {a}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 1" in nb_runner.get_output(2)

        # Add more prints
        nb_runner.set_cell_source(2, "print(f'a = {a}')\nprint(f'b = {b}')\nprint(f'c = {c}')")
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "a = 1" in output
        assert "b = 2" in output
        assert "c = 3" in output

    def test_change_output_variable(self, nb_runner):
        """Change which variable is printed."""
        nb_runner.create_notebook(
            [
                "first = 'hello'\nsecond = 'world'",
                "print(first)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hello" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "print(second)")
        nb_runner.run_all()
        assert "world" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestFormattedOutput:
    """Formatted output patterns with edits."""

    def test_table_format_edit(self, nb_runner):
        """Edit table-like output formatting."""
        nb_runner.create_notebook(
            [
                "items = [('apple', 3), ('banana', 5)]  # items for table",
                "for name, count in items:\n    print(f'{name}: {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple: 3" in out

        # Change to aligned format
        nb_runner.set_cell_source(2, "for name, count in items:\n    print(f'{name:>10s} | {count:>3d}')")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple" in out
        assert "3" in out

    def test_multi_line_output_edit(self, nb_runner):
        """Edit multi-line output generation."""
        nb_runner.create_notebook(
            [
                "header = 'Results'  # output header",
                "lines = [header, '-' * len(header), 'Item 1: OK', 'Item 2: OK']\nprint('\\n'.join(lines))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Results" in out
        assert "Item 1: OK" in out

        nb_runner.set_cell_source(1, "header = 'Summary'  # output header changed")
        nb_runner.run_all()
        assert "Summary" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestOutputCollectionEdits:
    """Build up output collections, edit formatting."""

    def test_list_to_table_format(self, nb_runner):
        """Change from list output to table-like format."""
        nb_runner.create_notebook(
            [
                "data = [('A', 1), ('B', 2), ('C', 3)]",
                "for name, val in data:\n    print(f'{name}: {val}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import json\nresult = {'status': 'ok', 'count': 42}",
                "print(json.dumps(result))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert '"status": "ok"' in output or '"status":"ok"' in output

        # Switch to pretty print
        nb_runner.set_cell_source(2, "print(json.dumps(result, indent=2))")
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert '"status": "ok"' in output


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestLargeOutput:
    """Large output scenarios."""

    def test_large_list_output(self, nb_runner):
        """Generate a large list, edit the size."""
        nb_runner.create_notebook(
            [
                "n = 100  # list size",
                "data = list(range(n))\nprint(f'len = {len(data)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len = 100" in nb_runner.get_output(2)

        # Make it bigger
        nb_runner.set_cell_source(1, "n = 1000  # list size bigger")
        nb_runner.run_all()
        assert "len = 1000" in nb_runner.get_output(2)

    def test_large_string_output(self, nb_runner):
        """Generate a large string, then edit pattern."""
        nb_runner.create_notebook(
            [
                "pattern = 'ab'  # string pattern",
                "big = pattern * 500\nprint(f'length = {len(big)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length = 1000" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "pattern = 'xyz'  # string pattern changed")
        nb_runner.run_all()
        assert "length = 1500" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.integration
class TestExpressionVsStatement:
    """Test expression vs statement behavior."""

    def test_bare_expression(self, nb_runner):
        """Bare expression in cell (like Jupyter shows last expression)."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "x",
                "y = x + 8",
                "print(y)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "50" in nb_runner.get_output(4)

    def test_semicolon_suppression(self, nb_runner):
        """Semicolons in cells."""
        nb_runner.create_notebook(
            [
                "a = 1; b = 2; c = 3",
                textwrap.dedent("""\
                total = a + b + c
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(2)
