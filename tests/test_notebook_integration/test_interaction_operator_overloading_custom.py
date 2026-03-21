"""Batch 398: custom __add__, __mul__ operator overloading."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOperatorOverloading:
    def test_add_mul(self, nb_runner):
        nb_runner.create_notebook([
            "class Vector:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __add__(self, other):\n        return Vector(self.x + other.x, self.y + other.y)\n    def __mul__(self, scalar):\n        return Vector(self.x * scalar, self.y * scalar)\n    def __repr__(self):\n        return f'V({self.x},{self.y})'",
            "v1 = Vector(1, 2)\nv2 = Vector(3, 4)\nv3 = v1 + v2\nv4 = v1 * 3\nprint(f'v3={v3} v4={v4}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v3=V(4,6)" in nb_runner.get_output(2)
        assert "v4=V(3,6)" in nb_runner.get_output(2)

    def test_overload_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Money:\n    def __init__(self, amount):\n        self.amount = amount\n    def __add__(self, other):\n        return Money(self.amount + other.amount)\n    def __repr__(self):\n        return f'${self.amount}'",
            "m1 = Money(10)\nm2 = Money(25)\ntotal = m1 + m2\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=$35" in nb_runner.get_output(2)
        # Edit class to add sub
        nb_runner.set_cell_source(1, "class Money:\n    def __init__(self, amount):\n        self.amount = amount\n    def __add__(self, other):\n        return Money(self.amount + other.amount)\n    def __sub__(self, other):\n        return Money(self.amount - other.amount)\n    def __repr__(self):\n        return f'${self.amount}'")
        nb_runner.set_cell_source(2, "m1 = Money(50)\nm2 = Money(25)\ndiff = m1 - m2\nprint(f'diff={diff}')")
        nb_runner.run_all()
        assert "diff=$25" in nb_runner.get_output(2)

    def test_iadd(self, nb_runner):
        nb_runner.create_notebook([
            "class Accumulator:\n    def __init__(self, val=0):\n        self.val = val\n    def __iadd__(self, other):\n        self.val += other\n        return self\n    def __repr__(self):\n        return f'Acc({self.val})'",
            "a = Accumulator()\na += 10\na += 20\nprint(f'a={a}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=Acc(30)" in nb_runner.get_output(2)
