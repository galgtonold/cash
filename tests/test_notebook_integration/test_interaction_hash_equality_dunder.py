"""Batch 504: hash and equality dunder methods."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHashEqualityDunder:
    def test_hash_eq_in_set(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class Card:\n    def __init__(self, rank, suit):\n        self.rank, self.suit = rank, suit\n    def __eq__(self, other):\n        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit\n    def __hash__(self):\n        return hash((self.rank, self.suit))\n    def __repr__(self):\n        return f'{self.rank}{self.suit}'\nc1 = Card('A', 'S')\nc2 = Card('A', 'S')\nc3 = Card('K', 'H')\nhand = {c1, c2, c3}\nprint(f'eq={c1 == c2} len={len(hand)} hand={sorted(str(c) for c in hand)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "eq=True" in out
        assert "len=2" in out

    def test_comparable_objects(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class Score:\n    def __init__(self, v): self.v = v\n    def __lt__(self, o): return self.v < o.v\n    def __eq__(self, o): return self.v == o.v\n    def __repr__(self): return f'S({self.v})'\nscores = [Score(5), Score(3), Score(8), Score(1)]\nordered = sorted(scores)\nprint(f'ordered={ordered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ordered=[S(1), S(3), S(5), S(8)]" in nb_runner.get_output(2)

    def test_hash_edit(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "class Pt:\n    def __init__(self, x, y): self.x, self.y = x, y\n    def __hash__(self): return hash((self.x, self.y))\n    def __eq__(self, o): return (self.x, self.y) == (o.x, o.y)\ns = {Pt(1,2), Pt(1,2), Pt(3,4)}\nprint(f'len={len(s)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "class Pt:\n    def __init__(self, x, y): self.x, self.y = x, y\n    def __hash__(self): return hash((self.x, self.y))\n    def __eq__(self, o): return (self.x, self.y) == (o.x, o.y)\ns = {Pt(1,2), Pt(3,4), Pt(5,6)}\nprint(f'len={len(s)}')")
        nb_runner.run_all()
        assert "len=3" in nb_runner.get_output(2)
