"""Diamonds, fan-out and fan-in, and parallel branches, edited on one side."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestDiamondDependencies:
    """Diamond-shaped dependency graphs + edits."""

    def test_diamond_edit_one_branch(self, nb_runner):
        """Diamond: edit only one branch."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "branch_a = base + 1",
                "branch_b = base + 2",
                "result = branch_a * branch_b\nprint(f'result = {result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "x = 5",
                "a = x + 1",
                "b = x + 2",
                "mid = a + b",
                "c = mid * 2",
                "d = mid * 3",
                "final = c + d\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=6, b=7, mid=13, c=26, d=39, final=65
        assert "final = 65" in nb_runner.get_output(7)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        # a=11, b=12, mid=23, c=46, d=69, final=115
        assert "final = 115" in nb_runner.get_output(7)


@pytest.mark.upstream
@pytest.mark.stress
class TestDiamondDependency:
    """Multiple cells depend on the same upstream cell."""

    def test_diamond_edit_root(self, nb_runner):
        """
        Cell 1: x = 10
        Cell 2: y = x + 1
        Cell 3: z = x * 2
        Cell 4: w = y + z  (diamond dependency on x through y and z)
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 1",
                "z = x * 2",
                "w = y + z\nprint(f'w = {w}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "w = 31" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(4)
        assert "w = 16" in nb_runner.get_output(4)

    def test_diamond_edit_one_branch(self, nb_runner):
        """Edit only one branch of the diamond."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 1",
                "z = x * 2",
                "w = y + z\nprint(f'w = {w}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "w = 31" in nb_runner.get_output(4)

        # Change only cell 2 formula (one branch)
        # y = 10 + 100 = 110, z = 10 * 2 = 20, w = 110 + 20 = 130
        nb_runner.set_cell_source(2, "y = x + 100")
        nb_runner.run_cell(4)
        assert "w = 130" in nb_runner.get_output(4)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestBranchingChainEdits:
    """Branching dependency chains with edits."""

    def test_diamond_dependency_edit_shared_root(self, nb_runner):
        """Diamond: root -> (left, right) -> merge."""
        nb_runner.create_notebook(
            [
                "root = 10",
                "left = root * 2",
                "right = root + 5",
                "merged = left + right\nprint(f'merged = {merged}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "root = 5",
                "left = root * 3",
                "right = root + 1",
                "merged = left + right\nprint(f'merged = {merged}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2\nprint(f'y = {y}')",
                "a = 100",
                "b = a + 50\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)
        assert "b = 150" in nb_runner.get_output(4)

        # Edit only chain 1
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)
        assert "b = 150" in nb_runner.get_output(4)


@pytest.mark.integration
@pytest.mark.stress
class TestComplexDependencyGraphs:
    """Test complex dependency graph patterns."""

    def test_diamond_with_function_deps(self, nb_runner):
        """Diamond dependency with functions: A -> B, A -> C, B+C -> D."""
        nb_runner.create_notebook(
            [
                "base_value = 10",
                textwrap.dedent("""\
                def path_b(x):
                    return x ** 2
                b_result = path_b(base_value)
            """),
                textwrap.dedent("""\
                def path_c(x):
                    return x * 3
                c_result = path_c(base_value)
            """),
                textwrap.dedent("""\
                final = b_result + c_result
                print(final)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "130" in nb_runner.get_output(4)  # 100 + 30

        # Change root
        nb_runner.set_cell_source(1, "base_value = 5")
        nb_runner.run_all()
        assert "40" in nb_runner.get_output(4)  # 25 + 15

    def test_wide_fan_out_fan_in(self, nb_runner):
        """Many independent computations merging into one result."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1",
                "b = x + 2",
                "c = x + 3",
                "d = x + 4",
                "e = x + 5",
                "total = a + b + c + d + e",
                "print(total)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 11+12+13+14+15 = 65
        assert "65" in nb_runner.get_output(8)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        # 101+102+103+104+105 = 515
        assert "515" in nb_runner.get_output(8)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestParallelBranches:
    """Test parallel independent branches merging downstream."""

    def test_three_branches_one_change(self, nb_runner):
        """Three parallel branches, only one changes."""
        nb_runner.create_notebook(
            [
                "x = 1\ny = 2\nz = 3",
                "ax = x * 10",
                "by = y * 10",
                "cz = z * 10",
                "total = ax + by + cz",
                "print(total)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10+20+30=60
        assert "60" in nb_runner.get_output(6)

        # Change only y
        nb_runner.set_cell_source(1, "x = 1\ny = 20\nz = 3")
        nb_runner.run_all()
        # 10+200+30=240
        assert "240" in nb_runner.get_output(6)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestMultiOutputChains:
    """Complex multi-output cell chains."""

    @pytest.mark.core
    @pytest.mark.upstream
    def test_multi_output_diamond_invalidation(self, nb_runner):
        """Multi-output cell feeding into diamond pattern."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "x = base + 1\ny = base + 2\nz = base + 3",
                "left = x * y",  # depends on x, y
                "right = y * z",  # depends on y, z
                "final = left + right",
                "print(f'final={final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        # x=11, y=12, z=13 → left=132, right=156 → final=288
        assert "final=288" in output

        # Change base
        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(6)
        # x=101, y=102, z=103 → left=10302, right=10506 → final=20808
        assert "final=20808" in output2
