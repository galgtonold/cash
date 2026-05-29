"""Batch 128 – Cross-cell data dependency interaction tests.

Tests that exercise complex cross-cell data flows, transitive
dependencies, diamond dependencies, and dependency chain changes.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(30)]


class TestTransitiveDependencies:
    """Long transitive dependency chains + edits."""

    def test_six_level_chain_edit_root(self, nb_runner):
        """6-level transitive chain, edit the root."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a * 2",
            "c = b * 2",
            "d = c * 2",
            "e = d * 2",
            "f = e * 2\nprint(f'f = {f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 32" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "a = 3")
        nb_runner.run_all()
        assert "f = 96" in nb_runner.get_output(6)

    def test_six_level_chain_edit_middle(self, nb_runner):
        """6-level chain, edit a middle node."""
        nb_runner.create_notebook([
            "a = 2",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "e = d + 1",
            "f = e + 1\nprint(f'f = {f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=2, b=3, c=4, d=5, e=6, f=7
        assert "f = 7" in nb_runner.get_output(6)

        nb_runner.set_cell_source(3, "c = b * 10")
        nb_runner.run_all()
        # a=2, b=3, c=30, d=31, e=32, f=33
        assert "f = 33" in nb_runner.get_output(6)


class TestDiamondDependencies:
    """Diamond-shaped dependency graphs + edits."""


    def test_diamond_edit_one_branch(self, nb_runner):
        """Diamond: edit only one branch."""
        nb_runner.create_notebook([
            "base = 10",
            "branch_a = base + 1",
            "branch_b = base + 2",
            "result = branch_a * branch_b\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 11 * 12 = 132
        assert "result = 132" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "branch_a = base * 10")
        nb_runner.run_all()
        # 100 * 12 = 1200
        assert "result = 1200" in nb_runner.get_output(4)

    def test_double_diamond(self, nb_runner):
        """Double diamond: two merge points."""
        nb_runner.create_notebook([
            "x = 5",
            "a = x + 1",
            "b = x + 2",
            "mid = a + b",
            "c = mid * 2",
            "d = mid * 3",
            "final = c + d\nprint(f'final = {final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=6, b=7, mid=13, c=26, d=39, final=65
        assert "final = 65" in nb_runner.get_output(7)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        # a=11, b=12, mid=23, c=46, d=69, final=115
        assert "final = 115" in nb_runner.get_output(7)


class TestDependencyChainChanges:
    """Change which variables a cell depends on."""

    def test_switch_input_variable(self, nb_runner):
        """Switch which variable a cell reads."""
        nb_runner.create_notebook([
            "a = 10",
            "b = 20",
            "result = a * 3\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Switch from a to b
        nb_runner.set_cell_source(3, "result = b * 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(3)

    def test_add_new_dependency(self, nb_runner):
        """Add a new dependency to an existing cell."""
        nb_runner.create_notebook([
            "x = 5",
            "y = 10",
            "result = x\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5" in nb_runner.get_output(3)

        # Now depend on both x and y
        nb_runner.set_cell_source(
            3, "result = x + y\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(3)

    def test_remove_dependency(self, nb_runner):
        """Remove a dependency from a cell."""
        nb_runner.create_notebook([
            "a = 10",
            "b = 20",
            "result = a + b\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Remove dependency on b
        nb_runner.set_cell_source(
            3, "result = a * 5\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)

        # Now editing b should NOT affect result
        nb_runner.set_cell_source(2, "b = 999")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)


class TestCyclicLikePatterns:
    """Patterns that look cyclic but aren't (self-assignment chains)."""

    def test_self_assignment_chain(self, nb_runner):
        """x depends on previous x (sequential mutation pattern)."""
        nb_runner.create_notebook([
            "x = [1]",
            "x = x + [2]  # extend step 1",
            "x = x + [3]  # extend step 2",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = [1, 2, 3]" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "x = x + [20]  # extend step 1 (modified)")
        nb_runner.run_all()
        assert "x = [1, 20, 3]" in nb_runner.get_output(4)

    def test_accumulating_string(self, nb_runner):
        """String accumulation pattern."""
        nb_runner.create_notebook([
            "s = 'hello'",
            "s = s + ' world'  # add world",
            "s = s + '!'  # add exclamation",
            "print(f's = {s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = hello world!" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "s = s + ' python'  # add python")
        nb_runner.run_all()
        assert "s = hello python!" in nb_runner.get_output(4)
