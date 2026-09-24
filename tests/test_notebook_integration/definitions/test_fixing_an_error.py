"""A cell that raised is fixed and re-run; nothing stale survives the error."""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.stress]


# Error handling + cell edit interaction tests.
#
# Tests that exercise error scenarios: syntax errors, runtime errors,
# exception recovery, and cell edits to fix errors.
#
# Note: nbclient raises CellExecutionError for cell errors, so we must
# use try/except to handle expected errors gracefully.
@pytest.mark.core
@pytest.mark.timeout(30)
class TestSyntaxErrorRecovery:
    """Introduce syntax error, then fix it."""

    def test_fix_syntax_error(self, nb_runner):
        """Cell has syntax error, fix it and re-run."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x *",  # Syntax error (incomplete expression)
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)  # Cell 1 should work

        # Cell 2 should fail with syntax error
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix the error and add print
        nb_runner.set_cell_source(2, "y = x * 2\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 20" in nb_runner.get_output(2)

    def test_syntax_error_does_not_corrupt_state(self, nb_runner):
        """Syntax error in one cell shouldn't corrupt other cached values."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 43" in nb_runner.get_output(2)

        # Introduce syntax error in cell 1
        nb_runner.set_cell_source(1, "x = ")
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(1)

        # Fix cell 1
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestRuntimeErrorRecovery:
    """Runtime errors (NameError, TypeError, etc.) and recovery."""

    def test_fix_name_error(self, nb_runner):
        """NameError due to undefined variable, fix by defining it."""
        nb_runner.create_notebook(
            [
                "y = undefined_var + 1",
            ]
        )
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(1)

        # Fix: define the variable first
        nb_runner.set_cell_source(1, "undefined_var = 10\ny = undefined_var + 1\nprint(f'y = {y}')")
        nb_runner.run_cell(1)
        assert "y = 11" in nb_runner.get_output(1)

    def test_fix_type_error(self, nb_runner):
        """TypeError, fix by correcting the operation."""
        nb_runner.create_notebook(
            [
                "x = 'hello'",
                "y = x + 1",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix: change to string concatenation
        nb_runner.set_cell_source(2, "y = x + ' world'\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = hello world" in nb_runner.get_output(2)

    def test_fix_zero_division(self, nb_runner):
        """ZeroDivisionError, fix by changing divisor."""
        nb_runner.create_notebook(
            [
                "x = 10\ny = 0",
                "z = x / y",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix: change y and add print
        nb_runner.set_cell_source(1, "x = 10\ny = 2")
        nb_runner.set_cell_source(2, "z = x / y\nprint(f'z = {z}')")
        nb_runner.run_all()
        assert "z = 5.0" in nb_runner.get_output(2)

    def test_fix_index_error(self, nb_runner):
        """IndexError, fix by using valid index."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "val = data[10]",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix
        nb_runner.set_cell_source(2, "val = data[2]\nprint(f'val = {val}')")
        nb_runner.run_cell(2)
        assert "val = 3" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestErrorThenSuccess:
    """Error in first run, success in subsequent runs."""

    def test_error_then_fix_sequence(self, nb_runner):
        """Error -> fix -> success sequence."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x / 0",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix
        nb_runner.set_cell_source(2, "y = x * 2\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 20" in nb_runner.get_output(2)

    def test_success_then_error_then_fix(self, nb_runner):
        """Success -> introduce error -> fix -> success."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Introduce error
        nb_runner.set_cell_source(2, "y = x / 0")
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix
        nb_runner.set_cell_source(2, "y = x * 3\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 30" in nb_runner.get_output(2)

    def test_error_in_middle_of_chain(self, nb_runner):
        """Error in middle cell, fix it, downstream should work."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = 1 / 0",  # Error
                "z = 999\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix cell 2
        nb_runner.set_cell_source(2, "y = x + 5")
        nb_runner.run_all()
        assert "z = 999" in nb_runner.get_output(3)


# Error-then-fix pattern interaction tests.
#
# Tests where cells produce errors, then are fixed, and
# the cache correctly handles the recovery.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestErrorThenFix:
    """Tests where errors are introduced then fixed."""

    def test_fix_name_error_recovery(self, nb_runner):
        """Introduce a NameError, then fix it."""
        nb_runner.create_notebook(
            [
                "x = 10  # name error recovery source",
                "result = x + y\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix by defining y
        nb_runner.set_cell_source(2, "y = 20\nresult = x + y\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        assert "result = 30" in nb_runner.get_output(2)

    def test_fix_type_error_recovery(self, nb_runner):
        """Introduce a TypeError, then fix it."""
        nb_runner.create_notebook(
            [
                "a = '10'  # type error recovery source",
                "result = a + 5\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix by converting
        nb_runner.set_cell_source(2, "result = int(a) + 5\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        assert "result = 15" in nb_runner.get_output(2)

    def test_fix_import_error_recovery(self, nb_runner):
        """Introduce an import error, then fix it."""
        nb_runner.create_notebook(
            [
                "from math import nonexistent_func  # import error recovery",
                "result = nonexistent_func(42)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(1)

        # Fix the import
        nb_runner.set_cell_source(1, "from math import sqrt  # import fixed recovery")
        nb_runner.set_cell_source(2, "result = sqrt(42)\nprint(f'result = {result:.2f}')")
        nb_runner.run_all()
        assert "result = 6.48" in nb_runner.get_output(2)

    def test_fix_then_iterate(self, nb_runner):
        """Fix an error, verify works, then edit again."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]  # fix iterate source",
                "result = data[10]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix index
        nb_runner.set_cell_source(2, "result = data[1]\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        assert "result = 2" in nb_runner.get_output(2)

        # Edit again to use a different index
        nb_runner.set_cell_source(2, "result = data[2]\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        assert "result = 3" in nb_runner.get_output(2)


# Error handling and recovery interaction tests.
#
# Tests where code errors occur, user fixes them, and caching
# should properly handle the error-recovery workflow.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestErrorThenFixCells:
    """Introduce error, then fix it."""

    def test_name_error_then_fix(self, nb_runner):
        """Cell causes NameError, fix it, run again."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "result = x + undefined_var\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix the cell
        nb_runner.set_cell_source(2, "result = x + 5\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        assert "result = 15" in nb_runner.get_output(2)

    def test_type_error_then_fix(self, nb_runner):
        """Cell causes TypeError, fix it."""
        nb_runner.create_notebook(
            [
                "val = 'hello'",
                "result = val + 10\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix
        nb_runner.set_cell_source(2, "result = val + str(10)\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        assert "result = hello10" in nb_runner.get_output(2)

    def test_index_error_then_fix(self, nb_runner):
        """Cell causes IndexError, fix it."""
        nb_runner.create_notebook(
            [
                "items = [1, 2, 3]",
                "val = items[10]\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        nb_runner.set_cell_source(2, "val = items[2]\nprint(f'val = {val}')")
        nb_runner.run_cell(2)
        assert "val = 3" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestErrorInMiddleChain:
    """Error in middle of dependency chain."""

    def test_error_in_cell2_fix_continue(self, nb_runner):
        """Error in cell 2 of 3, fix, continue."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x / 0  # will error",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        nb_runner.set_cell_source(2, "y = x * 2")
        nb_runner.run_cells([2, 3])
        assert "z = 11" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestFixSourceThenRerunCells:
    """Fix the source cell that caused downstream error."""

    def test_fix_upstream_data(self, nb_runner):
        """Downstream fails because of bad data, fix data."""
        nb_runner.create_notebook(
            [
                "data = []  # empty causes error",
                "avg = sum(data) / len(data)\nprint(f'avg = {avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "avg = 20.0" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestErrorAfterRestart:
    """Error handling after kernel restart."""

    def test_restart_after_error(self, nb_runner):
        """Error, restart, run clean."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Introduce error
        nb_runner.set_cell_source(2, "y = x / 0")
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Restart and fix
        nb_runner.shutdown()
        nb_runner.set_cell_source(2, "y = x * 5\nprint(f'y = {y}')")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 50" in nb_runner.get_output(2)
