"""Cells with several statements, edited one statement at a time."""

import pytest


@pytest.mark.stress
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


@pytest.mark.stress
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


@pytest.mark.stress
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


@pytest.mark.stress
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


@pytest.mark.stress
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


@pytest.mark.stress
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


@pytest.mark.stress
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


@pytest.mark.stress
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


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMultipleOutputsPerCell:
    """Cells producing multiple variables."""

    def test_multi_output_edit_one(self, nb_runner):
        """Cell producing multiple vars, edit to change one."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = 2\nc = 3",
                "total = a + b + c\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Change one variable
        nb_runner.set_cell_source(1, "a = 100\nb = 2\nc = 3")
        nb_runner.run_all()
        assert "total = 105" in nb_runner.get_output(2)

    def test_swap_variable_assignments(self, nb_runner):
        """Swap which variables get which values."""
        nb_runner.create_notebook(
            [
                "x = 10\ny = 20",
                "diff = x - y\nprint(f'diff = {diff}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "diff = -10" in nb_runner.get_output(2)

        # Swap values
        nb_runner.set_cell_source(1, "x = 20\ny = 10")
        nb_runner.run_all()
        assert "diff = 10" in nb_runner.get_output(2)
