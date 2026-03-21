"""Batch 140 – Multi-cell dependency chain stress tests.

Tests with longer dependency chains (5-8 cells) where edits
at various points in the chain verify cache propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(60)]


class TestLongChainEdits:
    """Long dependency chains with edits at different positions."""

    def test_six_cell_chain_edit_root(self, nb_runner):
        """6-cell chain, edit root."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "e = d + 1",
            "f = e + 1\nprint(f'f = {f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 6" in nb_runner.get_output(6)

        # Edit root
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "f = 15" in nb_runner.get_output(6)

    def test_six_cell_chain_edit_middle(self, nb_runner):
        """6-cell chain, edit cell 3 (middle)."""
        nb_runner.create_notebook([
            "a = 5",
            "b = a * 2",
            "c = b + 1",
            "d = c * 3",
            "e = d - 5",
            "print(f'e = {e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # b=10, c=11, d=33, e=28
        assert "e = 28" in nb_runner.get_output(6)

        # Edit cell 3
        nb_runner.set_cell_source(3, "c = b + 100")
        nb_runner.run_all()
        # b=10, c=110, d=330, e=325
        assert "e = 325" in nb_runner.get_output(6)

    def test_six_cell_chain_edit_near_end(self, nb_runner):
        """6-cell chain, edit second-to-last."""
        nb_runner.create_notebook([
            "a = 2",
            "b = a ** 2",
            "c = b + 3",
            "d = c * 2",
            "e = d + 100",
            "print(f'e = {e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # b=4, c=7, d=14, e=114
        assert "e = 114" in nb_runner.get_output(6)

        # Edit near end
        nb_runner.set_cell_source(5, "e = d * 100")
        nb_runner.run_all()
        # e = 14*100 = 1400
        assert "e = 1400" in nb_runner.get_output(6)


class TestBranchingChainEdits:
    """Branching dependency chains with edits."""

    def test_diamond_dependency_edit_shared_root(self, nb_runner):
        """Diamond: root -> (left, right) -> merge."""
        nb_runner.create_notebook([
            "root = 10",
            "left = root * 2",
            "right = root + 5",
            "merged = left + right\nprint(f'merged = {merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # left=20, right=15, merged=35
        assert "merged = 35" in nb_runner.get_output(4)

        # Edit root
        nb_runner.set_cell_source(1, "root = 100")
        nb_runner.run_all()
        # left=200, right=105, merged=305
        assert "merged = 305" in nb_runner.get_output(4)

    def test_diamond_edit_one_branch(self, nb_runner):
        """Diamond: edit one branch only."""
        nb_runner.create_notebook([
            "root = 5",
            "left = root * 3",
            "right = root + 1",
            "merged = left + right\nprint(f'merged = {merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # left=15, right=6, merged=21
        assert "merged = 21" in nb_runner.get_output(4)

        # Edit only left branch
        nb_runner.set_cell_source(2, "left = root * 10")
        nb_runner.run_all()
        # left=50, right=6, merged=56
        assert "merged = 56" in nb_runner.get_output(4)

    def test_two_independent_chains_edit_one(self, nb_runner):
        """Two independent chains, edit one and verify other unchanged."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2\nprint(f'y = {y}')",
            "a = 100",
            "b = a + 50\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)
        assert "b = 150" in nb_runner.get_output(4)

        # Edit only chain 1
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)
        assert "b = 150" in nb_runner.get_output(4)


class TestMultipleEditsInSequence:
    """Multiple sequential edits to the same chain."""

    def test_three_edits_to_root(self, nb_runner):
        """Edit root three times in sequence."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x * 10",
            "z = y + 5\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 2")
        nb_runner.run_all()
        assert "z = 25" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_all()
        assert "z = 55" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "z = 105" in nb_runner.get_output(3)

    def test_edit_different_cells_alternating(self, nb_runner):
        """Alternate between editing cell 1 and cell 2."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 5",
            "c = b * 2\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # c = (10+5)*2 = 30
        assert "c = 30" in nb_runner.get_output(3)

        # Edit cell 1
        nb_runner.set_cell_source(1, "a = 20")
        nb_runner.run_all()
        # c = (20+5)*2 = 50
        assert "c = 50" in nb_runner.get_output(3)

        # Edit cell 2
        nb_runner.set_cell_source(2, "b = a + 100")
        nb_runner.run_all()
        # c = (20+100)*2 = 240
        assert "c = 240" in nb_runner.get_output(3)

        # Edit cell 1 again
        nb_runner.set_cell_source(1, "a = 0")
        nb_runner.run_all()
        # c = (0+100)*2 = 200
        assert "c = 200" in nb_runner.get_output(3)
