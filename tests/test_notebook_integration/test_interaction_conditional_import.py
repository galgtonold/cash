"""
Batch 302: Conditional import and lazy loading interaction tests.
Tests that editing code with conditional imports and lazy attribute
access properly invalidates downstream cells.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestConditionalImportInteraction:
    """Test conditional import/lazy loading with cache invalidation."""

    def test_conditional_import_flag_edit(self, nb_runner):
        """Editing a flag that controls conditional import should propagate."""
        nb_runner.create_notebook([
            "use_math = True",
            "if use_math:\n    from math import pi\n    val = round(pi, 4)\nelse:\n    val = 3.0",
            "result = val * 2",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=6.2832" in out

        nb_runner.set_cell_source(1, "use_math = False")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=6.0" in out

    def test_try_import_fallback_edit(self, nb_runner):
        """Editing try/except import fallback should propagate."""
        nb_runner.create_notebook([
            "module_name = 'math'",
            (
                "if module_name == 'math':\n"
                "    import math\n"
                "    sqrt_fn = math.sqrt\n"
                "else:\n"
                "    sqrt_fn = lambda x: x ** 0.5"
            ),
            "result = round(sqrt_fn(144), 2)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=12.0" in out

        nb_runner.set_cell_source(1, "module_name = 'none'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=12.0" in out  # Both paths give same result for 144

    def test_lazy_attribute_access_edit(self, nb_runner):
        """Editing a lazy-loaded object should propagate."""
        nb_runner.create_notebook([
            (
                "class Lazy:\n"
                "    def __init__(self, factory):\n"
                "        self._factory = factory\n"
                "        self._value = None\n"
                "    @property\n"
                "    def value(self):\n"
                "        if self._value is None:\n"
                "            self._value = self._factory()\n"
                "        return self._value"
            ),
            "lazy = Lazy(lambda: list(range(5)))",
            "result = sum(lazy.value)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=10" in out

        nb_runner.set_cell_source(2, "lazy = Lazy(lambda: list(range(10)))")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=45" in out
