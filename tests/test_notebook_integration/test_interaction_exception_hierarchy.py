"""Batch 207 – Exception hierarchy and custom exception interaction tests.

Tests editing custom exception classes, raise patterns,
and exception handling hierarchies.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestCustomExceptionEdits:
    """Editing custom exception classes."""

    def test_edit_custom_exception(self, nb_runner):
        """Edit a custom exception class."""
        nb_runner.create_notebook([
            "class AppError(Exception):\n    def __init__(self, msg, code=0):\n        super().__init__(msg)\n        self.code = code",
            "try:\n    raise AppError('test', code=42)\nexcept AppError as e:\n    print(f'msg={e} code={e.code}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=test code=42" in nb_runner.get_output(2)

        # Add severity to exception
        nb_runner.set_cell_source(
            1,
            "class AppError(Exception):\n    def __init__(self, msg, code=0, severity='low'):\n        super().__init__(msg)\n        self.code = code\n        self.severity = severity",
        )
        nb_runner.set_cell_source(
            2,
            "try:\n    raise AppError('fail', code=99, severity='high')\nexcept AppError as e:\n    print(f'msg={e} code={e.code} sev={e.severity}')",
        )
        nb_runner.run_all()
        assert "msg=fail code=99 sev=high" in nb_runner.get_output(2)

    def test_edit_exception_hierarchy(self, nb_runner):
        """Edit exception hierarchy."""
        nb_runner.create_notebook([
            "class BaseErr(Exception): pass\nclass ChildErr(BaseErr): pass",
            "try:\n    raise ChildErr('child')\nexcept BaseErr as e:\n    print(f'caught: {type(e).__name__}: {e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "caught: ChildErr: child" in nb_runner.get_output(2)

        # Add new intermediate error
        nb_runner.set_cell_source(
            1,
            "class BaseErr(Exception): pass\nclass MidErr(BaseErr): pass\nclass ChildErr(MidErr): pass",
        )
        nb_runner.run_all()
        assert "caught: ChildErr: child" in nb_runner.get_output(2)


class TestExceptionHandlingEdits:
    """Editing exception handling patterns."""

    def test_edit_except_clause(self, nb_runner):
        """Edit which exceptions are caught."""
        nb_runner.create_notebook([
            "def risky(x):\n    if x == 0:\n        raise ValueError('zero')\n    return 10 / x",
            "try:\n    result = risky(0)\nexcept ValueError as e:\n    result = f'error: {e}'\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = error: zero" in nb_runner.get_output(2)

        # Change to catch Exception
        nb_runner.set_cell_source(
            2,
            "try:\n    result = risky(0)\nexcept Exception as e:\n    result = f'caught: {type(e).__name__}'\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = caught: ValueError" in nb_runner.get_output(2)
