"""Batch 494: class method and static method patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestClassMethodStaticMethod:
    def test_classmethod_factory(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup cell",
            "class Date:\n    def __init__(self, y, m, d): self.y, self.m, self.d = y, m, d\n    @classmethod\n    def from_string(cls, s):\n        y, m, d = map(int, s.split('-'))\n        return cls(y, m, d)\n    def __str__(self): return f'{self.y}/{self.m}/{self.d}'\nd = Date.from_string('2024-01-15')\nprint(f'date={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "date=2024/1/15" in nb_runner.get_output(2)

    def test_staticmethod_util(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup cell",
            "class MathUtils:\n    @staticmethod\n    def is_prime(n):\n        if n < 2: return False\n        return all(n % i != 0 for i in range(2, int(n**0.5)+1))\nprimes = [n for n in range(20) if MathUtils.is_prime(n)]\nprint(f'primes={primes}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "primes=[2, 3, 5, 7, 11, 13, 17, 19]" in nb_runner.get_output(2)

    def test_classmethod_edit(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup cell",
            "class Counter:\n    count = 0\n    @classmethod\n    def increment(cls): cls.count += 1\n    @classmethod\n    def get(cls): return cls.count\nCounter.increment()\nCounter.increment()\nprint(f'count={Counter.get()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "class Counter:\n    count = 0\n    @classmethod\n    def increment(cls): cls.count += 1\n    @classmethod\n    def get(cls): return cls.count\nfor _ in range(5): Counter.increment()\nprint(f'count={Counter.get()}')")
        nb_runner.run_all()
        assert "count=5" in nb_runner.get_output(2)
