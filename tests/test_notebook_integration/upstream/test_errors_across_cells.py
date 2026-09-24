"""A cell that raises, and the cells around it, before and after the fix."""

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestErrorHandlingPatterns:
    """Test how the caching system handles errors."""

    @pytest.mark.core
    def test_recover_after_fixing_error(self, nb_runner):
        """Fix a cell after an error — should compute correctly."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "print(f'y: {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "y: 20" in out1

        # Change to a different valid computation
        nb_runner.set_cell_source(2, "y = x ** 2")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "y: 100" in out2


@pytest.mark.core
class TestExceptionRecovery:
    """Test that errors in one cell don't break caching in subsequent cells."""

    def test_error_cell_doesnt_break_next_cell(self, nb_runner):
        """An error in cell 2 shouldn't prevent cell 3 from running."""
        from nbclient.exceptions import CellExecutionError

        nb_runner.create_notebook(
            [
                "x = 42",
                "y = 1/0  # ZeroDivisionError",
                "z = x * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        import contextlib

        with contextlib.suppress(CellExecutionError):
            nb_runner.run_cell(2)  # This will error
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "z = 84" in out, f"Got: {out}"

    def test_fix_error_and_rerun(self, nb_runner):
        """Fix a broken cell and re-run — should work correctly."""
        from nbclient.exceptions import CellExecutionError

        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + undefined_var",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        import contextlib

        with contextlib.suppress(CellExecutionError):
            nb_runner.run_cell(2)  # NameError

        # Fix cell 2
        nb_runner.set_cell_source(2, "y = x + 5")
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "y = 15" in out, f"Got: {out}"


@pytest.mark.core
class TestTryCatchPatterns:
    """Test try/except patterns in cached cells."""

    def test_try_except_switch_paths(self, nb_runner):
        """Change input to switch from success to error path."""
        nb_runner.create_notebook(
            [
                "x = '42'",
                "try:\n    val = int(x)\nexcept ValueError:\n    val = -1\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Change to trigger error path
        nb_runner.set_cell_source(1, "x = 'abc'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "val = -1" in out, f"Expected val=-1, got: {out}"
