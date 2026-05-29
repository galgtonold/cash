"""Batch 111 – Long cascade workflow interaction tests.

Tests that exercise multi-cell workflows with cascading edits,
partial reruns, and complex dependency chains.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(30)]


class TestCascadingEdits:
    """Edit one cell in a chain, verify all downstream update."""

    def test_five_cell_chain_edit_root(self, nb_runner):
        """5-cell chain: a -> b -> c -> d -> e. Edit a."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "e = d + 1\nprint(f'e = {e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 5" in nb_runner.get_output(5)

        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "e = 14" in nb_runner.get_output(5)

    def test_five_cell_chain_edit_middle(self, nb_runner):
        """5-cell chain, edit the middle cell."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b * 10",
            "d = c + 1",
            "e = d + 1\nprint(f'e = {e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 22" in nb_runner.get_output(5)

        nb_runner.set_cell_source(3, "c = b * 100")
        nb_runner.run_all()
        assert "e = 202" in nb_runner.get_output(5)

    def test_chain_edit_two_cells(self, nb_runner):
        """Edit two non-adjacent cells in a chain."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a * 2",
            "c = b + 10",
            "d = c * 3",
            "print(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 36" in nb_runner.get_output(5)

        # Edit a and c
        nb_runner.set_cell_source(1, "a = 5")
        nb_runner.set_cell_source(3, "c = b + 100")
        nb_runner.run_all()
        assert "d = 330" in nb_runner.get_output(5)


    def test_diamond_edit_one_branch(self, nb_runner):
        """Diamond dependency, edit one branch."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a * 2",
            "c = a * 3",
            "d = b + c\nprint(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 50" in nb_runner.get_output(4)

        # Edit only the b branch
        nb_runner.set_cell_source(2, "b = a * 10")
        nb_runner.run_all()
        assert "d = 130" in nb_runner.get_output(4)


class TestPartialReruns:
    """Run only some cells after edits."""

    def test_edit_middle_run_from_middle(self, nb_runner):
        """Edit cell 2, run cells 2-3 only."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 2",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "y = x * 10")
        nb_runner.run_cells([2, 3])
        assert "z = 51" in nb_runner.get_output(3)

    def test_edit_root_run_only_leaf(self, nb_runner):
        """Edit root cell but only run the leaf. Upstream should trigger."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 2",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(3)
        assert "z = 201" in nb_runner.get_output(3)

    def test_edit_leaf_only(self, nb_runner):
        """Edit only the leaf cell, run it."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 2",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(3, "z = y + 100\nprint(f'z = {z}')")
        nb_runner.run_cell(3)
        assert "z = 110" in nb_runner.get_output(3)


class TestMultiRoundWorkflows:
    """Multiple rounds of edits and reruns."""

    def test_three_rounds_of_edits(self, nb_runner):
        """Three successive rounds of editing the same cell."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Round 1
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        # Round 2
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)

        # Round 3
        nb_runner.set_cell_source(1, "x = 1000")
        nb_runner.run_all()
        assert "y = 1001" in nb_runner.get_output(2)

    def test_alternating_cell_edits(self, nb_runner):
        """Alternate editing two different cells."""
        nb_runner.create_notebook([
            "a = 1",
            "b = 2",
            "c = a + b\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        # Edit a
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "c = 12" in nb_runner.get_output(3)

        # Edit b
        nb_runner.set_cell_source(2, "b = 20")
        nb_runner.run_all()
        assert "c = 30" in nb_runner.get_output(3)

        # Edit a again
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()
        assert "c = 120" in nb_runner.get_output(3)

    def test_edit_with_intermediate_restart(self, nb_runner):
        """Edit, restart, edit again."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Edit and run
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Restart
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Edit again
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)

    def test_progressive_notebook_building(self, nb_runner):
        """Build a notebook progressively: run cells as they're added.
        This simulates typical notebook usage patterns."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "total = sum(data)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Now edit cell 1 to add more data
        nb_runner.set_cell_source(1, "data = [1, 2, 3, 4, 5]")
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

    def test_revert_all_changes(self, nb_runner):
        """Make edits, then revert everything back to original."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 1",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 3" in nb_runner.get_output(3)

        # Edit all cells
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.set_cell_source(2, "y = x * 2")
        nb_runner.set_cell_source(3, "z = y * 3\nprint(f'z = {z}')")
        nb_runner.run_all()
        assert "z = 600" in nb_runner.get_output(3)

        # Revert all
        nb_runner.set_cell_source(1, "x = 1")
        nb_runner.set_cell_source(2, "y = x + 1")
        nb_runner.set_cell_source(3, "z = y + 1\nprint(f'z = {z}')")
        nb_runner.run_all()
        assert "z = 3" in nb_runner.get_output(3)
