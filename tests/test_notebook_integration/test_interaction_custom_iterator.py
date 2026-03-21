"""Batch 375: custom iterator protocol (__iter__, __next__)."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCustomIterator:
    def test_range_iterator(self, nb_runner):
        nb_runner.create_notebook([
            "class CountDown:\n    def __init__(self, start):\n        self.start = start\n    def __iter__(self):\n        self.current = self.start\n        return self\n    def __next__(self):\n        if self.current <= 0:\n            raise StopIteration\n        val = self.current\n        self.current -= 1\n        return val",
            "result = list(CountDown(5))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[5, 4, 3, 2, 1]" in nb_runner.get_output(2)

    def test_iterator_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Repeat:\n    def __init__(self, val, times):\n        self.val = val\n        self.times = times\n    def __iter__(self):\n        self.count = 0\n        return self\n    def __next__(self):\n        if self.count >= self.times:\n            raise StopIteration\n        self.count += 1\n        return self.val",
            "result = list(Repeat('x', 3))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['x', 'x', 'x']" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "class Repeat:\n    def __init__(self, val, times):\n        self.val = val\n        self.times = times\n    def __iter__(self):\n        self.count = 0\n        return self\n    def __next__(self):\n        if self.count >= self.times:\n            raise StopIteration\n        self.count += 1\n        return self.val * self.count")
        nb_runner.run_all()
        assert "result=['x', 'xx', 'xxx']" in nb_runner.get_output(2)

    def test_iter_for_loop(self, nb_runner):
        nb_runner.create_notebook([
            "class Fibonacci:\n    def __init__(self, limit):\n        self.limit = limit\n    def __iter__(self):\n        a, b = 0, 1\n        while a < self.limit:\n            yield a\n            a, b = b, a + b",
            "fibs = list(Fibonacci(20))\nprint(f'fibs={fibs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fibs=[0, 1, 1, 2, 3, 5, 8, 13]" in nb_runner.get_output(2)
