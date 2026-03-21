"""Batch 458: hash() and equality protocol."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHashEquality:
    def test_hash_eq(self, nb_runner):
        nb_runner.create_notebook([
            "class Point:\n    def __init__(self, x, y): self.x, self.y = x, y\n    def __hash__(self): return hash((self.x, self.y))\n    def __eq__(self, other): return (self.x, self.y) == (other.x, other.y)",
            "p1 = Point(1, 2)\np2 = Point(1, 2)\np3 = Point(3, 4)\nprint(f'eq12={p1 == p2} eq13={p1 == p3} hash_eq={hash(p1) == hash(p2)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "eq12=True" in out
        assert "eq13=False" in out
        assert "hash_eq=True" in out

    def test_hashable_in_set(self, nb_runner):
        nb_runner.create_notebook([
            "class Color:\n    def __init__(self, r, g, b): self.r, self.g, self.b = r, g, b\n    def __hash__(self): return hash((self.r, self.g, self.b))\n    def __eq__(self, other): return (self.r, self.g, self.b) == (other.r, other.g, other.b)",
            "s = {Color(255, 0, 0), Color(0, 255, 0), Color(255, 0, 0)}\nprint(f'count={len(s)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)

    def test_hash_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Key:\n    def __init__(self, val): self.val = val\n    def __hash__(self): return hash(self.val)\n    def __eq__(self, o): return self.val == o.val",
            "d = {Key(1): 'a', Key(2): 'b'}\nresult = d[Key(1)]\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=a" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = {Key(10): 'x', Key(20): 'y'}\nresult = d[Key(20)]\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=y" in nb_runner.get_output(2)
