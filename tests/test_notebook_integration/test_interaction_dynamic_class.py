"""Batch 208 – Dynamic class creation interaction tests.

Tests editing dynamic class creation with type(),
class factories, and mixin patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestDynamicClassEdits:
    """Editing dynamic class patterns."""

    def test_edit_type_creation(self, nb_runner):
        """Edit dynamic class created with type()."""
        nb_runner.create_notebook([
            "MyClass = type('MyClass', (), {'value': 42, 'describe': lambda self: f'val={self.value}'})",
            "obj = MyClass()\nprint(f'result = {obj.describe()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = val=42" in nb_runner.get_output(2)

        # Change value
        nb_runner.set_cell_source(
            1,
            "MyClass = type('MyClass', (), {'value': 99, 'describe': lambda self: f'val={self.value}'})",
        )
        nb_runner.run_all()
        assert "result = val=99" in nb_runner.get_output(2)

    def test_edit_class_factory(self, nb_runner):
        """Edit a class factory function."""
        nb_runner.create_notebook([
            "def make_class(prefix):\n    class Cls:\n        def greet(self):\n            return f'{prefix} World'\n    return Cls",
            "Hello = make_class('Hello')\nobj = Hello()\nprint(f'result = {obj.greet()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = Hello World" in nb_runner.get_output(2)

        # Change factory
        nb_runner.set_cell_source(
            1,
            "def make_class(prefix):\n    class Cls:\n        def greet(self):\n            return f'{prefix}!!!'\n    return Cls",
        )
        nb_runner.run_all()
        assert "result = Hello!!!" in nb_runner.get_output(2)


class TestMixinEdits:
    """Editing mixin patterns."""

    def test_edit_mixin_combined(self, nb_runner):
        """Edit a class method that determines output."""
        nb_runner.create_notebook([
            "class Greeter:\n    def greet(self, name):\n        return 'Hello ' + name",
            "g = Greeter()\nprint(g.greet('alice'))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "Hello alice" in out1

        # Edit class method
        nb_runner.set_cell_source(1, "class Greeter:\n    def greet(self, name):\n        return 'Hi ' + name")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Hi alice" in out2
