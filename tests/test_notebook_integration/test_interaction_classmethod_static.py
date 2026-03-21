"""
Batch 335: class method / static method patterns with caching.
Tests @classmethod, @staticmethod, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestClassStaticMethod:
    """Test classmethod and staticmethod caching."""

    def test_classmethod_factory(self, nb_runner):
        """classmethod as factory with caching."""
        nb_runner.create_notebook([
            "class Date:\n    def __init__(self, y, m, d):\n        self.y = y\n        self.m = m\n        self.d = d\n    @classmethod\n    def from_string(cls, s):\n        y, m, d = map(int, s.split('-'))\n        return cls(y, m, d)\n    def __str__(self):\n        return f'{self.y}/{self.m}/{self.d}'",
            "d = Date.from_string('2024-06-15')",
            "print(f'date={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "date=2024/6/15" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "date=2024/6/15" in out2

    def test_staticmethod_edit(self, nb_runner):
        """staticmethod with edit propagation."""
        nb_runner.create_notebook([
            "class MathHelper:\n    @staticmethod\n    def clamp(val, lo, hi):\n        return max(lo, min(hi, val))",
            "val = 150",
            "result = MathHelper.clamp(val, 0, 100)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result=100" in out

        nb_runner.set_cell_source(2, "val = 50")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result=50" in out2

    def test_classmethod_counter(self, nb_runner):
        """classmethod tracking instance count."""
        nb_runner.create_notebook([
            "class Widget:\n    _count = 0\n    def __init__(self, name):\n        self.name = name\n        Widget._count += 1\n    @classmethod\n    def get_count(cls):\n        return cls._count",
            "w1 = Widget('A')\nw2 = Widget('B')\nw3 = Widget('C')\ncount = Widget.get_count()",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=3" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "count=3" in out2
