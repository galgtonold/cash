"""Batch 462: class __slots__ for memory efficiency."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSlotsMemory:
    def test_slots_basic(self, nb_runner):
        nb_runner.create_notebook([
            "class Point:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
            "p = Point(3, 4)\nhas_dict = hasattr(p, '__dict__')\nhas_slots = hasattr(p, '__slots__')\nprint(f'x={p.x} y={p.y} has_dict={has_dict} has_slots={has_slots}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "x=3" in out
        assert "has_dict=False" in out
        assert "has_slots=True" in out

    def test_slots_no_extra_attr(self, nb_runner):
        nb_runner.create_notebook([
            "class Pair:\n    __slots__ = ('a', 'b')\n    def __init__(self, a, b): self.a, self.b = a, b",
            "p = Pair(1, 2)\ntry:\n    p.c = 3\n    result = 'no_error'\nexcept AttributeError:\n    result = 'attr_error'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=attr_error" in nb_runner.get_output(2)

    def test_slots_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y): self.x, self.y = x, y\n    def mag_sq(self): return self.x**2 + self.y**2",
            "v = Vec(3, 4)\nprint(f'mag_sq={v.mag_sq()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag_sq=25" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "v = Vec(5, 12)\nprint(f'mag_sq={v.mag_sq()}')")
        nb_runner.run_all()
        assert "mag_sq=169" in nb_runner.get_output(2)
