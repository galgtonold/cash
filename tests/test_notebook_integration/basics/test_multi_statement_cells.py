"""Cells with several statements and outputs, edited and re-run."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


# Multi-statement cell interaction tests.
#
# Tests that exercise cells with multiple statements, where edits
# modify only some statements within a cell.
class TestMultiStatementCellEdits:
    """Edit individual statements within multi-statement cells."""

    @pytest.mark.core
    @pytest.mark.timeout(30)
    def test_add_statement_to_cell(self, nb_runner):
        """Add a new statement to an existing cell."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "result = x * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Add y to the first cell
        nb_runner.set_cell_source(1, "x = 10\ny = 5")
        nb_runner.set_cell_source(2, "result = x * 2 + y\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

    @pytest.mark.core
    @pytest.mark.timeout(30)
    def test_remove_statement_from_cell(self, nb_runner):
        """Remove a statement from a multi-statement cell."""
        nb_runner.create_notebook(
            [
                "x = 10\ny = 20\nz = 30",
                "total = x + y + z\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        # Remove z
        nb_runner.set_cell_source(1, "x = 10\ny = 20")
        nb_runner.set_cell_source(2, "total = x + y\nprint(f'total = {total}')")
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

    @pytest.mark.core
    @pytest.mark.timeout(30)
    def test_reorder_statements_in_cell(self, nb_runner):
        """Reorder statements within a cell."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = a + 1",
                "print(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 2" in nb_runner.get_output(2)

        # Swap order and change logic
        nb_runner.set_cell_source(1, "b = 10\na = b + 1")
        nb_runner.set_cell_source(2, "print(f'a = {a}, b = {b}')")
        nb_runner.run_all()
        assert "a = 11, b = 10" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_one_statement_in_multi_stmt_cell(self, nb_runner):
        """Edit one statement in a multi-statement cell."""
        nb_runner.create_notebook(
            [
                "x = 10\ny = x * 2\nz = y + 1",
                "print(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 21" in nb_runner.get_output(2)

        # Edit the middle statement
        nb_runner.set_cell_source(1, "x = 10\ny = x * 10\nz = y + 1")
        nb_runner.run_all()
        assert "z = 101" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_reorder_statements_feeding_a_computed_result(self, nb_runner):
        """Reorder statements within a cell."""
        nb_runner.create_notebook(
            [
                "a = 5\nb = a + 1",
                "result = b * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

        # Change to different computation order
        nb_runner.set_cell_source(1, "b = 100\na = b - 1")
        nb_runner.set_cell_source(2, "result = a * 2\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 198" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestMultiStatementWithFunction:
    """Multi-statement cells containing function defs."""

    def test_function_and_call_in_same_cell(self, nb_runner):
        """Function definition and call in same cell."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2\nresult = double(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(1)

        # Edit function body
        nb_runner.set_cell_source(
            1,
            "def double(x):\n    return x * 3\nresult = double(5)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(1)

    def test_two_functions_in_one_cell(self, nb_runner):
        """Two functions defined in one cell, used in next cell."""
        nb_runner.create_notebook(
            [
                "def add(a, b):\n    return a + b\ndef mul(a, b):\n    return a * b",
                "r1 = add(3, 4)\nr2 = mul(3, 4)\nprint(f'r1 = {r1}, r2 = {r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1 = 7, r2 = 12" in nb_runner.get_output(2)

        # Edit one function
        nb_runner.set_cell_source(
            1,
            "def add(a, b):\n    return a + b + 100\ndef mul(a, b):\n    return a * b",
        )
        nb_runner.run_all()
        assert "r1 = 107, r2 = 12" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestMultiStatementWithPrint:
    """Multi-statement cells with print statements (side effects)."""

    def test_edit_multi_statement_with_prints(self, nb_runner):
        """Edit a multi-statement cell that includes prints."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = 2\nprint(f'sum = {a + b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum = 3" in nb_runner.get_output(1)

        nb_runner.set_cell_source(1, "a = 10\nb = 20\nprint(f'sum = {a + b}')")
        nb_runner.run_all()
        assert "sum = 30" in nb_runner.get_output(1)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestMultiStatementDependencies:
    """Multi-statement cells with internal dependencies."""

    def test_internal_dependency_chain(self, nb_runner):
        """Statements within a cell depend on each other."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = a + 1\nc = b + 1\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(1)

        nb_runner.set_cell_source(1, "a = 10\nb = a + 1\nc = b + 1\nprint(f'c = {c}')")
        nb_runner.run_all()
        assert "c = 12" in nb_runner.get_output(1)

    def test_cross_cell_multi_statement(self, nb_runner):
        """Multi-statement cells with cross-cell dependencies."""
        nb_runner.create_notebook(
            [
                "x = 1\ny = 2",
                "a = x + y\nb = x * y",
                "result = a + b\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10\ny = 20")
        nb_runner.run_all()
        assert "result = 230" in nb_runner.get_output(3)


# Mixed computation and display pattern tests.
#
# Tests combining computation cells with display/print cells,
# editing either the computation or the display logic.
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


# Multi-output cell interaction tests.
#
# Tests where cells produce multiple outputs, some used by
# different downstream cells, with edits that affect
# only some of the outputs.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestMultiOutputEdits:
    """Cells that produce multiple variables."""

    def test_edit_multi_output_cell(self, nb_runner):
        """Edit cell that produces two variables."""
        nb_runner.create_notebook(
            [
                "a = 10\nb = 20",
                "result = a + b\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Edit to change both outputs
        nb_runner.set_cell_source(1, "a = 100\nb = 200")
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(2)

    def test_multi_output_different_consumers(self, nb_runner):
        """Two outputs consumed by different cells."""
        nb_runner.create_notebook(
            [
                "x = 5\ny = 10",
                "rx = x * 2\nprint(f'rx = {rx}')",
                "ry = y * 3\nprint(f'ry = {ry}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "rx = 10" in nb_runner.get_output(2)
        assert "ry = 30" in nb_runner.get_output(3)

        # Edit only x
        nb_runner.set_cell_source(1, "x = 50\ny = 10")
        nb_runner.run_all()
        assert "rx = 100" in nb_runner.get_output(2)
        assert "ry = 30" in nb_runner.get_output(3)

    def test_tuple_unpacking_edit(self, nb_runner):
        """Tuple unpacking with edits."""
        nb_runner.create_notebook(
            [
                "a, b, c = 1, 2, 3",
                "total = a + b + c\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a, b, c = 10, 20, 30")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)


# Multi-statement cell interaction tests (advanced).
#
# Tests that exercise cells with multiple statements and complex
# interactions between statements within the same cell.
@pytest.mark.core
@pytest.mark.timeout(30)
class TestMultiStatementBasic:
    """Multi-statement cells, edit individual parts."""

    def test_edit_middle_statement_in_multi(self, nb_runner):
        """Cell has 3 assignments, edit the middle one."""
        nb_runner.create_notebook(
            [
                "x = 1\ny = 2\nz = 3",
                "result = x * y * z\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 1\ny = 20\nz = 3")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

    def test_add_statement_to_cell(self, nb_runner):
        """Add a statement to a cell."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "result = x\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Add an additional computation
        nb_runner.set_cell_source(1, "x = 10\nx = x * 5")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestMultiStatementWithFunctions:
    """Multi-statement cells with function calls."""

    def test_define_and_use_in_same_cell(self, nb_runner):
        """Define function and use it in same cell, edit the function."""
        nb_runner.create_notebook(
            [
                "def calc(x):\n    return x * 2\nresult = calc(5)",
                "print(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def calc(x):\n    return x ** 2\nresult = calc(5)")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestMultiStatementWithControlFlow:
    """Multi-statement cells with control flow."""

    def test_assignment_then_loop(self, nb_runner):
        """Assignment followed by loop in same cell."""
        nb_runner.create_notebook(
            [
                "n = 5",
                "total = 0\nfor i in range(n):\n    total += i\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0+1+2+3+4 = 10
        assert "total = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "n = 10")
        nb_runner.run_all()
        # 0+1+...+9 = 45
        assert "total = 45" in nb_runner.get_output(2)

    def test_complex_multi_statement_edit(self, nb_runner):
        """Complex cell with multiple interacting statements, edit one."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "mean = sum(data) / len(data)\ndevs = [(x - mean) ** 2 for x in data]\nvariance = sum(devs) / len(devs)\nprint(f'variance = {variance}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # mean=3, devs=[4,1,0,1,4], variance=2.0
        assert "variance = 2.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 10, 10, 10, 10]")
        nb_runner.run_all()
        # mean=10, devs=[0,0,0,0,0], variance=0.0
        assert "variance = 0.0" in nb_runner.get_output(2)


# Print and display output interaction tests.
#
# Tests where print/display formatting changes, output cells
# are edited, and print modes are toggled.
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


# Multi-output cell patterns, display vs return, print ordering,
# and assignment expression (walrus) patterns.
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
