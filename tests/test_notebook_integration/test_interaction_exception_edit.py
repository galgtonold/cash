"""Batch 250 – Exception hierarchy and error handling edits.

Tests custom exception classes and try/except flow with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestExceptionHierarchyEdit:
    """Custom exception and error handling patterns."""

    def test_custom_exception_edit(self, nb_runner):
        """Edit custom exception message format."""
        nb_runner.create_notebook([
            "class AppError(Exception):\n    def __init__(self, code, msg):\n        self.code = code\n        self.msg = msg\n    def __str__(self):\n        return f'Error {self.code}: {self.msg}'",
            "try:\n    raise AppError(404, 'not found')\nexcept AppError as e:\n    result = str(e)\nprint(f'result = {result}')",
        ])
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
        nb_runner.create_notebook([
            "def safe_div(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return -1",
            "result = safe_div(10, 0)\nprint(f'result = {result}')",
        ])
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
        nb_runner.create_notebook([
            "def parse_value(s):\n    try:\n        return int(s)\n    except ValueError:\n        return 'not_int'\n    except TypeError:\n        return 'bad_type'",
            "r1 = parse_value('abc')\nr2 = parse_value(None)\nprint(f'r1={r1} r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=not_int r2=bad_type" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def parse_value(s):\n    try:\n        return int(s)\n    except ValueError:\n        return -1\n    except TypeError:\n        return -2",
        )
        nb_runner.run_all()
        assert "r1=-1 r2=-2" in nb_runner.get_output(2)
