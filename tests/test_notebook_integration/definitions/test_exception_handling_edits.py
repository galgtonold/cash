"""Editing try/except blocks and custom exception classes."""

import pytest

pytestmark = [pytest.mark.stress]


# Exception handling code interaction tests.
#
# Tests where try/except blocks are edited, error paths change,
# and caching handles exception-related code modifications.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestTryExceptEdits:
    """Edit try/except blocks."""

    def test_edit_try_body(self, nb_runner):
        """Edit the try body."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "try:\n    result = data[0]\nexcept IndexError:\n    result = -1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(2)

        # Edit to access out of bounds
        nb_runner.set_cell_source(
            2,
            "try:\n    result = data[10]\nexcept IndexError:\n    result = -1\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

    def test_edit_except_handler(self, nb_runner):
        """Edit the except handler."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "try:\n    result = 100 / x\nexcept ZeroDivisionError:\n    result = 0\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0" in nb_runner.get_output(2)

        # Edit to handle differently
        nb_runner.set_cell_source(
            2,
            "try:\n    result = 100 / x\nexcept ZeroDivisionError:\n    result = -999\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = -999" in nb_runner.get_output(2)

    def test_fix_value_to_avoid_exception(self, nb_runner):
        """Fix the value so exception doesn't trigger."""
        nb_runner.create_notebook(
            [
                "divisor = 0",
                "try:\n    result = 100 / divisor\nexcept ZeroDivisionError:\n    result = -1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

        # Fix divisor
        nb_runner.set_cell_source(1, "divisor = 4")
        nb_runner.run_all()
        assert "result = 25.0" in nb_runner.get_output(2)


# Exception hierarchy and error handling edits.
#
# Tests custom exception classes and try/except flow with edits.
@pytest.mark.timeout(90)
class TestExceptionHierarchyEdit:
    """Custom exception and error handling patterns."""

    def test_custom_exception_edit(self, nb_runner):
        """Edit custom exception message format."""
        nb_runner.create_notebook(
            [
                "class AppError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        self.msg = msg\n    def __str__(self):\n        return f'Error {self.code}: {self.msg}'",
                "try:\n    raise AppError(404, 'not found')\nexcept AppError as e:\n    result = str(e)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = Error 404: not found" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class AppError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        self.msg = msg\n    def __str__(self):\n        return f'[{self.code}] {self.msg}'",
        )
        nb_runner.run_all()
        assert "result = [404] not found" in nb_runner.get_output(2)

    def test_exception_handler_edit(self, nb_runner):
        """Edit the exception handler logic."""
        nb_runner.create_notebook(
            [
                "def safe_div(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return -1",
                "result = safe_div(10, 0)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def safe_div(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return 0",
        )
        nb_runner.run_all()
        assert "result = 0" in nb_runner.get_output(2)

    def test_multiple_except_edit(self, nb_runner):
        """Edit multi-handler except block."""
        nb_runner.create_notebook(
            [
                "def parse_value(s):\n    try:\n        return int(s)\n    except ValueError:\n        return 'not_int'\n    except TypeError:\n        return 'bad_type'",
                "r1 = parse_value('abc')\nr2 = parse_value(None)\nprint(f'r1={r1} r2={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=not_int r2=bad_type" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def parse_value(s):\n    try:\n        return int(s)\n    except ValueError:\n        return -1\n    except TypeError:\n        return -2",
        )
        nb_runner.run_all()
        assert "r1=-1 r2=-2" in nb_runner.get_output(2)
