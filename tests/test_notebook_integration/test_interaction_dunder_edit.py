"""Batch 274 – Class __repr__/__str__ edit propagation.

Tests editing dunder methods on classes.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDunderMethodEdits:
    """Dunder method edit patterns."""

    def test_str_edit(self, nb_runner):
        """Edit __str__, string representation changes."""
        nb_runner.create_notebook([
            "class Item:\n    def __init__(self, name, qty):\n        self.name = name\n        self.qty = qty\n    def __str__(self):\n        return f'{self.name}({self.qty})'",
            "item = Item('widget', 5)\nprint(f'item = {item}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "item = widget(5)" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Item:\n    def __init__(self, name, qty):\n        self.name = name\n        self.qty = qty\n    def __str__(self):\n        return f'{self.name} x{self.qty}'",
        )
        nb_runner.run_all()
        assert "item = widget x5" in nb_runner.get_output(2)

    def test_eq_edit(self, nb_runner):
        """Edit __eq__, comparison results change."""
        nb_runner.create_notebook([
            "class Box:\n    def __init__(self, size):\n        self.size = size\n    def __eq__(self, other):\n        return self.size == other.size",
            "b1 = Box(10)\nb2 = Box(10)\nresult = (b1 == b2)\nprint(f'equal = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "equal = True" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Box:\n    def __init__(self, size):\n        self.size = size\n    def __eq__(self, other):\n        return self.size > other.size",
        )
        nb_runner.run_all()
        assert "equal = False" in nb_runner.get_output(2)

    def test_len_edit(self, nb_runner):
        """Edit __len__, len() call result changes."""
        nb_runner.create_notebook([
            "class Container:\n    def __init__(self, items):\n        self.items = items\n    def __len__(self):\n        return len(self.items)",
            "c = Container([1, 2, 3, 4, 5])\nsize = len(c)\nprint(f'size = {size}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size = 5" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Container:\n    def __init__(self, items):\n        self.items = items\n    def __len__(self):\n        return len(self.items) * 2",
        )
        nb_runner.run_all()
        assert "size = 10" in nb_runner.get_output(2)
