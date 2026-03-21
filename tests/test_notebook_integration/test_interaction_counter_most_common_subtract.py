"""Batch 514: collections Counter most_common subtract."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCounterMostCommonSubtract:
    def test_most_common(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter",
            "text = 'abracadabra'\nc = Counter(text)\ntop3 = c.most_common(3)\nprint(f'top3={top3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('a', 5)" in out

    def test_counter_subtract(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter",
            "inventory = Counter(apples=10, bananas=5, oranges=8)\nsold = Counter(apples=3, bananas=2)\ninventory.subtract(sold)\nprint(f'apples={inventory[\"apples\"]} bananas={inventory[\"bananas\"]} oranges={inventory[\"oranges\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apples=7" in out
        assert "bananas=3" in out
        assert "oranges=8" in out

    def test_counter_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter",
            "c = Counter([1, 1, 2, 3, 3, 3])\nprint(f'most={c.most_common(1)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "most=[(3, 3)]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "c = Counter([1, 1, 1, 1, 2, 3])\nprint(f'most={c.most_common(1)}')")
        nb_runner.run_all()
        assert "most=[(1, 4)]" in nb_runner.get_output(2)
