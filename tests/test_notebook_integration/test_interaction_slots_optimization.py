"""Batch 384: class slots optimization and attribute access."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSlotsOptimization:
    def test_slots_basic(self, nb_runner):
        nb_runner.create_notebook([
            "class Point:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
            "p = Point(3, 4)\nresult = f'{p.x},{p.y}'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3,4" in nb_runner.get_output(2)

    def test_slots_edit_class(self, nb_runner):
        nb_runner.create_notebook([
            "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def mag(self):\n        return (self.x**2 + self.y**2) ** 0.5",
            "v = Vec(3, 4)\nm = v.mag()\nprint(f'm={m}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "m=5.0" in nb_runner.get_output(2)
        # Edit to 3D
        nb_runner.set_cell_source(1, "class Vec:\n    __slots__ = ('x', 'y', 'z')\n    def __init__(self, x, y, z):\n        self.x = x\n        self.y = y\n        self.z = z\n    def mag(self):\n        return (self.x**2 + self.y**2 + self.z**2) ** 0.5")
        nb_runner.set_cell_source(2, "v = Vec(1, 2, 2)\nm = v.mag()\nprint(f'm={m}')")
        nb_runner.run_all()
        assert "m=3.0" in nb_runner.get_output(2)

    def test_slots_no_dict(self, nb_runner):
        nb_runner.create_notebook([
            "class Tiny:\n    __slots__ = ('val',)\n    def __init__(self, v):\n        self.val = v",
            "t = Tiny(42)\nhas_dict = hasattr(t, '__dict__')\nprint(f'val={t.val} has_dict={has_dict}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=42 has_dict=False" in nb_runner.get_output(2)
