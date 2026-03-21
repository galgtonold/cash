"""
Interaction test: dataclass field with default_factory.
Tests dataclass with default_factory, field metadata,
post_init processing, and cross-cell dataclass pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassDefaultFactory:
    """Test dataclass default_factory and field metadata across cells."""

    def test_dataclass_factory(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: dataclass with default_factory
            "from dataclasses import dataclass, field\n@dataclass\nclass ShoppingCart:\n    owner: str\n    items: list = field(default_factory=list)\n    discounts: dict = field(default_factory=dict)\n\ncart = ShoppingCart('Alice')\ncart.items.append('apple')\ncart.items.append('banana')\ncart.discounts['apple'] = 0.1\nprint(f'owner={cart.owner}')\nprint(f'items={cart.items}')",
            # Cell 2: use cart state
            "total_items = len(cart.items)\nhas_discount = len(cart.discounts) > 0\nprint(f'total_items={total_items}')\nprint(f'has_discount={has_discount}')",
            # Cell 3: create second cart (no sharing)
            "cart2 = ShoppingCart('Bob')\ncart2.items.append('orange')\nprint(f'cart1_items={len(cart.items)}')\nprint(f'cart2_items={len(cart2.items)}')\nprint(f'independent={cart.items is not cart2.items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "owner=Alice" in out1
        assert "items=['apple', 'banana']" in out1
        out2 = nb_runner.get_output(2)
        assert "total_items=2" in out2
        assert "has_discount=True" in out2
        out3 = nb_runner.get_output(3)
        assert "cart1_items=2" in out3
        assert "cart2_items=1" in out3
        assert "independent=True" in out3

    def test_dataclass_factory_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str\n    tags: list = field(default_factory=lambda: ['default'])\n\nc = Config('test')\nprint(f'tags={c.tags}')",
            "tag_str = ','.join(c.tags)\nprint(f'tag_str={tag_str}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "tag_str=default" in nb_runner.get_output(2)

        # Edit to add more default tags
        nb_runner.set_cell_source(1, "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str\n    tags: list = field(default_factory=lambda: ['default', 'v2'])\n\nc = Config('test')\nprint(f'tags={c.tags}')")
        nb_runner.run_cells([1, 2])
        assert "tag_str=default,v2" in nb_runner.get_output(2)

    def test_dataclass_factory_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass\nclass Stats:\n    values: list = field(default_factory=lambda: [10, 20, 30])\n\ns = Stats()\navg = sum(s.values) / len(s.values)\nprint(f'avg={avg:.1f}')",
            "spread = max(s.values) - min(s.values)\nprint(f'spread={spread}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=20.0" in nb_runner.get_output(1)
        assert "spread=20" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "spread=20" in nb_runner.get_output(2)
