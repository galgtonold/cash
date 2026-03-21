"""
Interaction test: property with computed cache and validation.
Tests @property for computed values with internal caching,
setter validation, and cross-cell attribute management.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPropertyComputedValidation:
    """Test property with computed values and validation across cells."""

    def test_property_computed(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: class with validated property
            "class BankAccount:\n    def __init__(self, owner, balance=0):\n        self.owner = owner\n        self._balance = balance\n    @property\n    def balance(self):\n        return self._balance\n    @balance.setter\n    def balance(self, value):\n        if value < 0:\n            raise ValueError('Negative balance')\n        self._balance = value\n    @property\n    def status(self):\n        if self._balance > 1000:\n            return 'premium'\n        elif self._balance > 0:\n            return 'active'\n        return 'empty'\n\nacc = BankAccount('Alice', 500)\nprint(f'balance={acc.balance}')\nprint(f'status={acc.status}')",
            # Cell 2: modify through property
            "acc.balance = 1500\nprint(f'new_balance={acc.balance}')\nprint(f'new_status={acc.status}')",
            # Cell 3: read state
            "info = f'{acc.owner}: ${acc.balance} ({acc.status})'\nprint(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "balance=500" in out1
        assert "status=active" in out1
        out2 = nb_runner.get_output(2)
        assert "new_balance=1500" in out2
        assert "new_status=premium" in out2
        out3 = nb_runner.get_output(3)
        assert "info=Alice: $1500 (premium)" in out3

    def test_property_computed_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Circle:\n    def __init__(self, radius):\n        self._r = radius\n    @property\n    def diameter(self):\n        return self._r * 2\n    @property\n    def circumference(self):\n        import math\n        return 2 * math.pi * self._r\n\nc = Circle(5)\nprint(f'diam={c.diameter}')\nprint(f'circ={c.circumference:.2f}')",
            "ratio = c.circumference / c.diameter\nprint(f'ratio={ratio:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "diam=10" in nb_runner.get_output(1)
        assert "ratio=3.1416" in nb_runner.get_output(2)

        # Edit radius
        nb_runner.set_cell_source(1, "class Circle:\n    def __init__(self, radius):\n        self._r = radius\n    @property\n    def diameter(self):\n        return self._r * 2\n    @property\n    def circumference(self):\n        import math\n        return 2 * math.pi * self._r\n\nc = Circle(10)\nprint(f'diam={c.diameter}')\nprint(f'circ={c.circumference:.2f}')")
        nb_runner.run_cells([1, 2])
        assert "diam=20" in nb_runner.get_output(1)
        assert "ratio=3.1416" in nb_runner.get_output(2)

    def test_property_computed_cache(self, nb_runner):
        nb_runner.create_notebook([
            "class Config:\n    def __init__(self, values):\n        self._values = values\n    @property\n    def count(self):\n        return len(self._values)\n    @property\n    def summary(self):\n        return f'{self.count} items'\n\ncfg = Config([1, 2, 3])\nprint(f'summary={cfg.summary}')",
            "has_many = cfg.count > 2\nprint(f'has_many={has_many}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary=3 items" in nb_runner.get_output(1)
        assert "has_many=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_many=True" in nb_runner.get_output(2)
