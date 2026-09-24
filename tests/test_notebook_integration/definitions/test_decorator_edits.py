"""Editing a decorator, or where it is applied, and the calls downstream."""

import pytest

pytestmark = [pytest.mark.stress]


class TestDecoratorEdits:
    """Decorator definition and application edits."""

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_decorator_logic(self, nb_runner):
        """Edit decorator logic, verify function behavior changes."""
        nb_runner.create_notebook(
            [
                "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 2\n    return wrapper",
                "@double_result\ndef add(a, b):\n    return a + b",
                "result = add(3, 4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(3)

        # Change decorator to triple
        nb_runner.set_cell_source(
            1,
            "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 3\n    return wrapper",
        )
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_decorated_function(self, nb_runner):
        """Edit the decorated function itself."""
        nb_runner.create_notebook(
            [
                "def negate(fn):\n    def wrapper(*args):\n        return -fn(*args)\n    return wrapper",
                "@negate\ndef compute(x):\n    return x * 2",
                "result = compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -10" in nb_runner.get_output(3)

        # Edit the base function
        nb_runner.set_cell_source(2, "@negate\ndef compute(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = -25" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_remove_decorator(self, nb_runner):
        """Remove decorator from function."""
        nb_runner.create_notebook(
            [
                "def add_ten(fn):\n    def wrapper(*args):\n        return fn(*args) + 10\n    return wrapper",
                "@add_ten\ndef square(x):\n    return x * x",
                "result = square(3)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 19" in nb_runner.get_output(3)

        # Remove decorator
        nb_runner.set_cell_source(2, "def square(x):\n    return x * x")
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(3)

    # Decorator patterns with edits.
    #
    # Tests custom decorators being edited and downstream function behavior.
    @pytest.mark.timeout(90)
    def test_logging_decorator_edit(self, nb_runner):
        """Edit decorator wrapper, decorated function behavior changes."""
        nb_runner.create_notebook(
            [
                "def logged(fn):\n    def wrapper(*args):\n        result = fn(*args)\n        return f'[LOG] {result}'\n    return wrapper",
                "@logged\ndef greet(name):\n    return f'Hello {name}'",
                "msg = greet('Alice')\nprint(f'msg = {msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = [LOG] Hello Alice" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "def logged(fn):\n    def wrapper(*args):\n        result = fn(*args)\n        return f'[DEBUG] {result}'\n    return wrapper",
        )
        nb_runner.run_all()
        assert "msg = [DEBUG] Hello Alice" in nb_runner.get_output(3)

    @pytest.mark.timeout(90)
    def test_retry_decorator_edit(self, nb_runner):
        """Edit retry count in decorator."""
        nb_runner.create_notebook(
            [
                "def with_prefix(prefix):\n    def decorator(fn):\n        def wrapper(*args):\n            return f'{prefix}: {fn(*args)}'\n        return wrapper\n    return decorator",
                "@with_prefix('INFO')\ndef status(code):\n    return f'code={code}'",
                "result = status(200)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = INFO: code=200" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "def with_prefix(prefix):\n    def decorator(fn):\n        def wrapper(*args):\n            return f'[{prefix}] {fn(*args)}'\n        return wrapper\n    return decorator",
        )
        nb_runner.run_all()
        assert "result = [INFO] code=200" in nb_runner.get_output(3)

    @pytest.mark.timeout(90)
    def test_decorator_with_edited_function(self, nb_runner):
        """Edit decorated function body."""
        nb_runner.create_notebook(
            [
                "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 2\n    return wrapper",
                "@double_result\ndef compute(x):\n    return x + 1",
                "val = compute(5)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 12" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "@double_result\ndef compute(x):\n    return x * 3",
        )
        nb_runner.run_all()
        assert "val = 30" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDecoratorWithArgs:
    """Decorators with arguments."""

    def test_edit_decorator_argument(self, nb_runner):
        """Edit the argument to a decorator factory."""
        nb_runner.create_notebook(
            [
                "def multiply_by(n):\n    def decorator(fn):\n        def wrapper(*args):\n            return fn(*args) * n\n        return wrapper\n    return decorator",
                "@multiply_by(2)\ndef add(a, b):\n    return a + b",
                "result = add(3, 4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(3)

        # Change multiplier
        nb_runner.set_cell_source(2, "@multiply_by(5)\ndef add(a, b):\n    return a + b")
        nb_runner.run_all()
        assert "result = 35" in nb_runner.get_output(3)
