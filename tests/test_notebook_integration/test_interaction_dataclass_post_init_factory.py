"""
Interaction test: dataclass with custom __post_init__ and field factory.
Tests dataclass post-init validation, field(default_factory=...), and
computed property behavior across cells.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassPostInitFactory:
    """Test dataclass __post_init__ and default_factory across cells."""

    def test_dataclass_post_init_validation(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define dataclass with post_init
            "from dataclasses import dataclass, field\n@dataclass\nclass Product:\n    name: str\n    price: float\n    tags: list = field(default_factory=list)\n    discount_price: float = field(init=False)\n    def __post_init__(self):\n        self.discount_price = round(self.price * 0.9, 2)\n        self.name = self.name.strip().title()\nprint('Product defined')",
            # Cell 2: create instances
            "p1 = Product('  laptop  ', 999.99)\np2 = Product('mouse pad', 19.50, ['accessories', 'office'])\nprint(f'p1_name={p1.name}')\nprint(f'p1_disc={p1.discount_price}')\nprint(f'p2_tags={p2.tags}')",
            # Cell 3: aggregate
            "products = [p1, p2]\ntotal = sum(p.discount_price for p in products)\navg = round(total / len(products), 2)\nprint(f'total={total}')\nprint(f'avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "p1_name=Laptop" in out2
        assert "p1_disc=899.99" in out2
        assert "p2_tags=['accessories', 'office']" in out2
        out3 = nb_runner.get_output(3)
        assert "total=" in out3
        assert "avg=" in out3

    def test_dataclass_edit_price(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass\nclass Product:\n    name: str\n    price: float\n    discount_price: float = field(init=False)\n    def __post_init__(self):\n        self.discount_price = round(self.price * 0.9, 2)\nprint('defined')",
            "p = Product('widget', 100.0)\nprint(f'disc={p.discount_price}')",
            "savings = round(p.price - p.discount_price, 2)\nprint(f'savings={savings}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "disc=90.0" in nb_runner.get_output(2)
        assert "savings=10.0" in nb_runner.get_output(3)

        # Change discount rate
        nb_runner.set_cell_source(1, "from dataclasses import dataclass, field\n@dataclass\nclass Product:\n    name: str\n    price: float\n    discount_price: float = field(init=False)\n    def __post_init__(self):\n        self.discount_price = round(self.price * 0.8, 2)\nprint('defined')")
        nb_runner.run_cells([1, 2, 3])
        assert "disc=80.0" in nb_runner.get_output(2)
        assert "savings=20.0" in nb_runner.get_output(3)

    def test_dataclass_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass\nclass Item:\n    name: str\n    qty: int = 1\n    history: list = field(default_factory=list)\nprint('defined')",
            "item = Item('bolt', 50, ['warehouse'])\nprint(f'name={item.name} qty={item.qty} hist={item.history}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=bolt qty=50 hist=['warehouse']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "name=bolt qty=50 hist=['warehouse']" in nb_runner.get_output(2)
