"""
Batch 290: Module-level vs local scope interaction tests.
Tests that editing module-level constants/variables and local function
variables properly invalidates downstream cells.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestModuleLocalScopeInteraction:
    """Test module vs local scope patterns with cache invalidation."""

    def test_module_constant_edit(self, nb_runner):
        """Editing a module-level constant used inside a function should propagate."""
        nb_runner.create_notebook([
            "MULTIPLIER = 10",
            "def scale(x):\n    return x * MULTIPLIER",
            "result = scale(5)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=50" in out

        nb_runner.set_cell_source(1, "MULTIPLIER = 100")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=500" in out

    def test_global_dict_edit(self, nb_runner):
        """Editing a global config dict used in functions should propagate."""
        nb_runner.create_notebook([
            "CONFIG = {'tax_rate': 0.1, 'discount': 0.05}",
            "def compute_price(base):\n    tax = base * CONFIG['tax_rate']\n    disc = base * CONFIG['discount']\n    return base + tax - disc",
            "price = compute_price(100)",
            "print(f'price={price}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "price=105.0" in out

        nb_runner.set_cell_source(1, "CONFIG = {'tax_rate': 0.2, 'discount': 0.1}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "price=110.0" in out

    def test_nested_function_scope_edit(self, nb_runner):
        """Editing a closure variable captured from outer scope should propagate."""
        nb_runner.create_notebook([
            "base_offset = 5",
            "def make_adder():\n    offset = base_offset\n    def add(x):\n        return x + offset\n    return add",
            "adder = make_adder()\nresult = adder(10)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=15" in out

        nb_runner.set_cell_source(1, "base_offset = 50")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=60" in out
