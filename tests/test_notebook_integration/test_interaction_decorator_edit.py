"""Batch 254 – Decorator patterns with edits.

Tests custom decorators being edited and downstream function behavior.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDecoratorEdits:
    """Custom decorator edit propagation."""

    def test_logging_decorator_edit(self, nb_runner):
        """Edit decorator wrapper, decorated function behavior changes."""
        nb_runner.create_notebook([
            "def logged(fn):\n    def wrapper(*args):\n        result = fn(*args)\n        return f'[LOG] {result}'\n    return wrapper",
            "@logged\ndef greet(name):\n    return f'Hello {name}'",
            "msg = greet('Alice')\nprint(f'msg = {msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = [LOG] Hello Alice" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "def logged(fn):\n    def wrapper(*args):\n        result = fn(*args)\n        return f'[DEBUG] {result}'\n    return wrapper",
        )
        nb_runner.run_all()
        assert "msg = [DEBUG] Hello Alice" in nb_runner.get_output(3)

    def test_retry_decorator_edit(self, nb_runner):
        """Edit retry count in decorator."""
        nb_runner.create_notebook([
            "def with_prefix(prefix):\n    def decorator(fn):\n        def wrapper(*args):\n            return f'{prefix}: {fn(*args)}'\n        return wrapper\n    return decorator",
            "@with_prefix('INFO')\ndef status(code):\n    return f'code={code}'",
            "result = status(200)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = INFO: code=200" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "def with_prefix(prefix):\n    def decorator(fn):\n        def wrapper(*args):\n            return f'[{prefix}] {fn(*args)}'\n        return wrapper\n    return decorator",
        )
        nb_runner.run_all()
        assert "result = [INFO] code=200" in nb_runner.get_output(3)

    def test_decorator_with_edited_function(self, nb_runner):
        """Edit decorated function body."""
        nb_runner.create_notebook([
            "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 2\n    return wrapper",
            "@double_result\ndef compute(x):\n    return x + 1",
            "val = compute(5)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 12" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "@double_result\ndef compute(x):\n    return x * 3",
        )
        nb_runner.run_all()
        assert "val = 30" in nb_runner.get_output(3)
