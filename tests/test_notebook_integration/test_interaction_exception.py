"""
Batch 299: Error handling and exception propagation interaction tests.
Tests that editing code that raises/catches exceptions properly
invalidates downstream cells.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestExceptionHandlingInteraction:
    """Test exception handling patterns with cache invalidation."""

    def test_try_except_edit_exception_type(self, nb_runner):
        """Editing which exception is caught should propagate."""
        nb_runner.create_notebook([
            "def risky(x):\n    if x == 0:\n        raise ValueError('zero')\n    return 100 // x",
            "try:\n    val = risky(0)\nexcept ValueError as e:\n    val = -1",
            "print(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "val=-1" in out

        # Change to non-zero (no exception)
        nb_runner.set_cell_source(2, "try:\n    val = risky(5)\nexcept ValueError as e:\n    val = -1")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "val=20" in out

    def test_custom_exception_edit(self, nb_runner):
        """Editing custom exception handling should propagate."""
        nb_runner.create_notebook([
            (
                "class AppError(Exception):\n"
                "    def __init__(self, code, msg):\n"
                "        self.code = code\n"
                "        self.msg = msg"
            ),
            "def process(x):\n    if x < 0:\n        raise AppError(400, 'negative')\n    return x * 2",
            "try:\n    result = process(-5)\n    status = 'ok'\nexcept AppError as e:\n    result = 0\n    status = f'error:{e.code}'",
            "print(f'result={result},status={status}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=0,status=error:400" in out

        nb_runner.set_cell_source(3, "try:\n    result = process(10)\n    status = 'ok'\nexcept AppError as e:\n    result = 0\n    status = f'error:{e.code}'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=20,status=ok" in out

    def test_finally_block_edit(self, nb_runner):
        """Editing code with try/finally should propagate."""
        nb_runner.create_notebook([
            (
                "try:\n"
                "    val = 100 // 5\n"
                "    status = 'success'\n"
                "except ZeroDivisionError:\n"
                "    val = -1\n"
                "    status = 'error'\n"
                "finally:\n"
                "    cleanup = True"
            ),
            "print(f'val={val},status={status},cleanup={cleanup}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "val=20" in out
        assert "status=success" in out
        assert "cleanup=True" in out

        nb_runner.set_cell_source(1, (
            "try:\n"
            "    val = 100 // 0\n"
            "    status = 'success'\n"
            "except ZeroDivisionError:\n"
            "    val = -1\n"
            "    status = 'error'\n"
            "finally:\n"
            "    cleanup = True"
        ))
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "val=-1" in out
        assert "status=error" in out
        assert "cleanup=True" in out
