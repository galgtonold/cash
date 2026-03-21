"""Batch 356: custom __repr__, __str__, __eq__ dunder methods."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDunderReprEq:
    def test_repr_str(self, nb_runner):
        nb_runner.create_notebook([
            "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'Point({self.x}, {self.y})'\n    def __str__(self):\n        return f'({self.x}, {self.y})'",
            "p = Point(3, 4)\nrepr_s = repr(p)\nstr_s = str(p)\nprint(f'repr={repr_s} str={str_s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "repr=Point(3, 4)" in out
        assert "str=(3, 4)" in out

    def test_eq_hash_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Color:\n    def __init__(self, r, g, b):\n        self.r = r\n        self.g = g\n        self.b = b\n    def __eq__(self, other):\n        return (self.r, self.g, self.b) == (other.r, other.g, other.b)\n    def __hash__(self):\n        return hash((self.r, self.g, self.b))",
            "c1 = Color(255, 0, 0)\nc2 = Color(255, 0, 0)\nc3 = Color(0, 255, 0)\neq12 = c1 == c2\neq13 = c1 == c3\nprint(f'eq12={eq12} eq13={eq13}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "eq12=True eq13=False" in nb_runner.get_output(2)
        # Edit class
        nb_runner.set_cell_source(1, "class Color:\n    def __init__(self, r, g, b):\n        self.r = r\n        self.g = g\n        self.b = b\n    def __eq__(self, other):\n        return self.r == other.r\n    def __hash__(self):\n        return hash(self.r)")
        nb_runner.set_cell_source(2, "c1 = Color(255, 0, 0)\nc2 = Color(255, 100, 200)\neq = c1 == c2\nprint(f'eq={eq}')")
        nb_runner.run_all()
        assert "eq=True" in nb_runner.get_output(2)

    def test_lt_sort(self, nb_runner):
        nb_runner.create_notebook([
            "class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price\n    def __lt__(self, other):\n        return self.price < other.price\n    def __repr__(self):\n        return f'{self.name}:{self.price}'",
            "items = [Item('b', 30), Item('a', 10), Item('c', 20)]\nsorted_items = sorted(items)\nprint(f'sorted={sorted_items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sorted=[a:10, c:20, b:30]" in nb_runner.get_output(2)
