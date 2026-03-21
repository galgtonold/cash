"""Batch 195 – Multi-cell class composition (has-a) interaction tests.

Tests where one class has another class as a member,
and edits propagate through the composition.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestCompositionEdits:
    """Editing composed class structures."""

    def test_edit_component_class(self, nb_runner):
        """Edit the component class in a composition."""
        nb_runner.create_notebook([
            "class Engine:\n    def __init__(self, hp):\n        self.hp = hp\n    def describe(self):\n        return f'{self.hp}hp'",
            "class Car:\n    def __init__(self, name, engine):\n        self.name = name\n        self.engine = engine\n    def info(self):\n        return f'{self.name}: {self.engine.describe()}'",
            "e = Engine(200)\nc = Car('Tesla', e)\nprint(f'info = {c.info()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info = Tesla: 200hp" in nb_runner.get_output(3)

        # Edit Engine to include type
        nb_runner.set_cell_source(
            1,
            "class Engine:\n    def __init__(self, hp, typ='gas'):\n        self.hp = hp\n        self.typ = typ\n    def describe(self):\n        return f'{self.hp}hp {self.typ}'",
        )
        nb_runner.set_cell_source(
            3, "e = Engine(300, 'electric')\nc = Car('Tesla', e)\nprint(f'info = {c.info()}')"
        )
        nb_runner.run_all()
        assert "info = Tesla: 300hp electric" in nb_runner.get_output(3)

    def test_edit_container_class(self, nb_runner):
        """Edit the container class in a composition."""
        nb_runner.create_notebook([
            "class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price",
            "class Cart:\n    def __init__(self):\n        self.items = []\n    def add(self, item):\n        self.items.append(item)\n    def total(self):\n        return sum(i.price for i in self.items)",
            "cart = Cart()\ncart.add(Item('A', 10))\ncart.add(Item('B', 20))\nprint(f'total = {cart.total()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(3)

        # Edit Cart to add tax
        nb_runner.set_cell_source(
            2,
            "class Cart:\n    def __init__(self, tax=0.1):\n        self.items = []\n        self.tax = tax\n    def add(self, item):\n        self.items.append(item)\n    def total(self):\n        subtotal = sum(i.price for i in self.items)\n        return subtotal * (1 + self.tax)",
        )
        nb_runner.run_all()
        assert "total = 33.0" in nb_runner.get_output(3)

    def test_edit_both_classes(self, nb_runner):
        """Edit both component and container."""
        nb_runner.create_notebook([
            "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
            "class Line:\n    def __init__(self, p1, p2):\n        self.p1 = p1\n        self.p2 = p2\n    def length(self):\n        return ((self.p2.x - self.p1.x)**2 + (self.p2.y - self.p1.y)**2) ** 0.5",
            "a = Point(0, 0)\nb = Point(3, 4)\nline = Line(a, b)\nprint(f'length = {line.length()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length = 5.0" in nb_runner.get_output(3)

        # Change points
        nb_runner.set_cell_source(
            3,
            "a = Point(1, 1)\nb = Point(4, 5)\nline = Line(a, b)\nprint(f'length = {line.length()}')",
        )
        nb_runner.run_all()
        assert "length = 5.0" in nb_runner.get_output(3)
