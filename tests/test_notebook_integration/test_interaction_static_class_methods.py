"""Batch 463: static methods and class methods."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStaticClassMethods:
    def test_classmethod(self, nb_runner):
        nb_runner.create_notebook([
            "class Date:\n    def __init__(self, y, m, d): self.y, self.m, self.d = y, m, d\n    @classmethod\n    def from_string(cls, s):\n        parts = s.split('-')\n        return cls(int(parts[0]), int(parts[1]), int(parts[2]))",
            "d = Date.from_string('2024-06-15')\nprint(f'y={d.y} m={d.m} d={d.d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y=2024" in nb_runner.get_output(2)
        assert "m=6" in nb_runner.get_output(2)
        assert "d=15" in nb_runner.get_output(2)

    def test_staticmethod(self, nb_runner):
        nb_runner.create_notebook([
            "class Math:\n    @staticmethod\n    def clamp(val, lo, hi):\n        return max(lo, min(val, hi))",
            "r1 = Math.clamp(5, 0, 10)\nr2 = Math.clamp(-5, 0, 10)\nr3 = Math.clamp(15, 0, 10)\nprint(f'r1={r1} r2={r2} r3={r3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=5" in nb_runner.get_output(2)
        assert "r2=0" in nb_runner.get_output(2)
        assert "r3=10" in nb_runner.get_output(2)

    def test_classmethod_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Config:\n    def __init__(self, **kw): self.data = kw\n    @classmethod\n    def default(cls): return cls(mode='fast', debug=False)",
            "c = Config.default()\nprint(f'mode={c.data[\"mode\"]} debug={c.data[\"debug\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mode=fast" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "class Config:\n    def __init__(self, **kw): self.data = kw\n    @classmethod\n    def default(cls): return cls(mode='slow', debug=True)")
        nb_runner.run_all()
        assert "mode=slow" in nb_runner.get_output(2)
        assert "debug=True" in nb_runner.get_output(2)
