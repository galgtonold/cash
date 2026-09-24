"""An edit at any point of a chain of cells reaches every cell below it."""

import pytest


@pytest.mark.stress
@pytest.mark.integration
class TestDeepDependencyChains:
    """Test multi-level variable dependency propagation across many cells."""

    def test_linear_chain_10_cells(self, nb_runner):
        """10-cell linear dependency chain: each cell uses previous cell's output."""
        cells = [f"v{i} = v{i - 1} + 1" if i > 0 else "v0 = 1" for i in range(10)]
        cells.append("print(v9)")
        nb_runner.create_notebook(cells)
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "10" in nb_runner.get_output(len(cells))

        # Second run — all should be cached
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "10" in nb_runner.get_output(len(cells))

    def test_linear_chain_change_root(self, nb_runner):
        """Change root of a dependency chain and verify propagation."""
        nb_runner.create_notebook(["a = 10", "b = a * 2", "c = b + 5", "d = c ** 2", "print(d)"])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "625" in nb_runner.get_output(5)  # (10*2+5)^2 = 625

        # Change root
        nb_runner.set_cell_source(1, "a = 20")
        nb_runner.run_all()
        assert "2025" in nb_runner.get_output(5)  # (20*2+5)^2 = 2025

    def test_diamond_dependency(self, nb_runner):
        """Diamond pattern: A -> B, A -> C, B+C -> D."""
        nb_runner.create_notebook(["a = 5", "b = a * 2", "c = a + 3", "d = b + c", "print(d)"])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "18" in nb_runner.get_output(5)  # 5*2 + 5+3 = 18

        # Change root -> both branches update
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "33" in nb_runner.get_output(5)  # 10*2 + 10+3 = 33

    def test_fan_out_dependency(self, nb_runner):
        """One variable feeds many downstream cells."""
        nb_runner.create_notebook(
            ["base = 100", "x1 = base + 1", "x2 = base + 2", "x3 = base + 3", "total = x1 + x2 + x3", "print(total)"]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "306" in nb_runner.get_output(6)  # 101+102+103

        nb_runner.set_cell_source(1, "base = 200")
        nb_runner.run_all()
        assert "606" in nb_runner.get_output(6)  # 201+202+203

    def test_fan_in_dependency(self, nb_runner):
        """Multiple independent sources converge into one cell."""
        nb_runner.create_notebook(["x = 10", "y = 20", "z = 30", "total = x + y + z", "print(total)"])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "60" in nb_runner.get_output(5)

        # Change only one source
        nb_runner.set_cell_source(2, "y = 200")
        nb_runner.run_all()
        assert "240" in nb_runner.get_output(5)


@pytest.mark.core
class TestDeepDependencyChainEdits:
    """Test deep dependency chains spanning many cells."""

    def test_six_cell_chain_modification_propagates(self, nb_runner):
        """
        Run a 6-cell chain, modify cell 1, then re-run cell 6.
        The upstream simulation should detect the change and recompute.
        """
        nb_runner.create_notebook(
            [
                "base = 10",
                "step1 = base * 2",
                "step2 = step1 + 5",
                "step3 = step2 ** 2",
                "step4 = step3 - 100",
                "print(f'Result: {step4}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out1 = nb_runner.get_output(6)
        assert "Result: 525" in out1

        # Modify the base cell
        nb_runner.set_cell_source(1, "base = 5")
        # Re-run from cell 1 through cell 6
        nb_runner.run_cells([1, 2, 3, 4, 5, 6])

        # base=5 -> step1=10 -> step2=15 -> step3=225 -> step4=125
        out2 = nb_runner.get_output(6)
        assert "Result: 125" in out2, f"Expected 125 after modification, got: {out2}"

    def test_branching_modify_root_all_branches_update(self, nb_runner):
        """
        Modify the root of a diamond and verify both branches update.
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x * 2",
                "b = x + 5",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "c = 35" in nb_runner.get_output(4)

        # Change x from 10 to 100
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cells([1, 2, 3, 4])

        # x=100, a=200, b=105, c=305
        out = nb_runner.get_output(4)
        assert "c = 305" in out, f"Expected c=305 after modification, got: {out}"


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestCascadingEdits:
    """Edit one cell in a chain, verify all downstream update."""

    def test_five_cell_chain_edit_root(self, nb_runner):
        """5-cell chain: a -> b -> c -> d -> e. Edit a."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 5" in nb_runner.get_output(5)

        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "e = 14" in nb_runner.get_output(5)

    def test_five_cell_chain_edit_middle(self, nb_runner):
        """5-cell chain, edit the middle cell."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b * 10",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 22" in nb_runner.get_output(5)

        nb_runner.set_cell_source(3, "c = b * 100")
        nb_runner.run_all()
        assert "e = 202" in nb_runner.get_output(5)

    def test_chain_edit_two_cells(self, nb_runner):
        """Edit two non-adjacent cells in a chain."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a * 2",
                "c = b + 10",
                "d = c * 3",
                "print(f'd = {d}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a * 2",
                "c = a * 3",
                "d = b + c\nprint(f'd = {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 50" in nb_runner.get_output(4)

        # Edit only the b branch
        nb_runner.set_cell_source(2, "b = a * 10")
        nb_runner.run_all()
        assert "d = 130" in nb_runner.get_output(4)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestTransitiveDependencies:
    """Long transitive dependency chains + edits."""

    def test_six_level_chain_edit_root(self, nb_runner):
        """6-level transitive chain, edit the root."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a * 2",
                "c = b * 2",
                "d = c * 2",
                "e = d * 2",
                "f = e * 2\nprint(f'f = {f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 32" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "a = 3")
        nb_runner.run_all()
        assert "f = 96" in nb_runner.get_output(6)

    def test_six_level_chain_edit_middle(self, nb_runner):
        """6-level chain, edit a middle node."""
        nb_runner.create_notebook(
            [
                "a = 2",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1",
                "f = e + 1\nprint(f'f = {f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=2, b=3, c=4, d=5, e=6, f=7
        assert "f = 7" in nb_runner.get_output(6)

        nb_runner.set_cell_source(3, "c = b * 10")
        nb_runner.run_all()
        # a=2, b=3, c=30, d=31, e=32, f=33
        assert "f = 33" in nb_runner.get_output(6)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestLongChainEdits:
    """Long dependency chains with edits at different positions."""

    def test_six_cell_chain_edit_root(self, nb_runner):
        """6-cell chain, edit root."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1",
                "f = e + 1\nprint(f'f = {f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 6" in nb_runner.get_output(6)

        # Edit root
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "f = 15" in nb_runner.get_output(6)

    def test_six_cell_chain_edit_middle(self, nb_runner):
        """6-cell chain, edit cell 3 (middle)."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a * 2",
                "c = b + 1",
                "d = c * 3",
                "e = d - 5",
                "print(f'e = {e}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "a = 2",
                "b = a ** 2",
                "c = b + 3",
                "d = c * 2",
                "e = d + 100",
                "print(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # b=4, c=7, d=14, e=114
        assert "e = 114" in nb_runner.get_output(6)

        # Edit near end
        nb_runner.set_cell_source(5, "e = d * 100")
        nb_runner.run_all()
        # e = 14*100 = 1400
        assert "e = 1400" in nb_runner.get_output(6)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestLongChainPropagation:
    """Long dependency chain edit patterns."""

    def test_six_cell_chain(self, nb_runner):
        """Edit cell 1 in 6-cell chain, cell 6 reflects."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "step1 = base + 5",
                "step2 = step1 * 2",
                "step3 = step2 - 3",
                "step4 = step3 // 4",
                "result = step4\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10+5=15, 15*2=30, 30-3=27, 27//4=6
        assert "result = 6" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_all()
        # 100+5=105, 105*2=210, 210-3=207, 207//4=51
        assert "result = 51" in nb_runner.get_output(6)

    def test_edit_middle_of_long_chain(self, nb_runner):
        """Edit middle cell (3 of 5), tail updates."""
        nb_runner.create_notebook(
            [
                "x = 2",
                "y = x * 3",
                "z = y + 10",
                "w = z ** 2",
                "final = w - 1\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x=2, y=6, z=16, w=256, final=255
        assert "final = 255" in nb_runner.get_output(5)

        nb_runner.set_cell_source(3, "z = y + 100")
        nb_runner.run_all()
        # x=2, y=6, z=106, w=11236, final=11235
        assert "final = 11235" in nb_runner.get_output(5)

    def test_branching_chain(self, nb_runner):
        """Two branches merge in final cell."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 20",
                "left = a * 2",
                "right = b * 3",
                "combined = left + right\nprint(f'combined = {combined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10*2 + 20*3 = 20 + 60 = 80
        assert "combined = 80" in nb_runner.get_output(5)

        # Edit one branch source
        nb_runner.set_cell_source(1, "a = 50")
        nb_runner.run_all()
        # 50*2 + 20*3 = 100 + 60 = 160
        assert "combined = 160" in nb_runner.get_output(5)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDeepChainPropagation:
    """Deep dependency chains with edits at various points."""

    def test_edit_head_of_chain(self, nb_runner):
        """5-cell chain, edit the first cell."""
        nb_runner.create_notebook(
            [
                "a = 1  # head",
                "b = a + 1  # step 2",
                "c = b * 2  # step 3",
                "d = c + 10  # step 4",
                "result = d * 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=1, b=2, c=4, d=14, result=42
        assert "result = 42" in nb_runner.get_output(5)

        # Edit head
        nb_runner.set_cell_source(1, "a = 10  # head changed")
        nb_runner.run_all()
        # a=10, b=11, c=22, d=32, result=96
        assert "result = 96" in nb_runner.get_output(5)

    def test_edit_middle_of_chain(self, nb_runner):
        """5-cell chain, edit the middle cell."""
        nb_runner.create_notebook(
            [
                "x = 2  # start",
                "y = x * 3  # middle1",
                "z = y + 1  # middle2",
                "w = z ** 2  # step4",
                "out = w - 1\nprint(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x=2, y=6, z=7, w=49, out=48
        assert "out = 48" in nb_runner.get_output(5)

        # Edit middle
        nb_runner.set_cell_source(3, "z = y + 100  # middle2 boosted")
        nb_runner.run_all()
        # x=2, y=6, z=106, w=11236, out=11235
        assert "out = 11235" in nb_runner.get_output(5)

    def test_edit_tail_of_chain(self, nb_runner):
        """5-cell chain, edit the last cell."""
        nb_runner.create_notebook(
            [
                "p = 5  # start",
                "q = p * 2  # step2",
                "r = q + 3  # step3",
                "s = r - 1  # step4",
                "final = s\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # p=5, q=10, r=13, s=12, final=12
        assert "final = 12" in nb_runner.get_output(5)

        # Edit output cell
        nb_runner.set_cell_source(5, "final = s * 100\nprint(f'final = {final}')")
        nb_runner.run_all()
        assert "final = 1200" in nb_runner.get_output(5)

    def test_edit_two_points_in_chain(self, nb_runner):
        """Edit two non-adjacent cells in a chain simultaneously."""
        nb_runner.create_notebook(
            [
                "a = 1  # chain start",
                "b = a + 10  # chain step2",
                "c = b * 2  # chain step3",
                "d = c + 5  # chain step4",
                "e = d * 3\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=1, b=11, c=22, d=27, e=81
        assert "e = 81" in nb_runner.get_output(5)

        # Edit cells 1 and 4
        nb_runner.set_cell_source(1, "a = 100  # chain start big")
        nb_runner.set_cell_source(4, "d = c + 1000  # chain step4 big")
        nb_runner.run_all()
        # a=100, b=110, c=220, d=1220, e=3660
        assert "e = 3660" in nb_runner.get_output(5)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMultiCellChainEdits:
    """Editing cells in multi-step pipelines."""

    def test_edit_middle_of_3_cell_chain(self, nb_runner):
        """Edit the middle cell in a 3-cell chain."""
        nb_runner.create_notebook(
            [
                "raw = [1, 2, 3, 4, 5]",
                "processed = [x ** 2 for x in raw]",
                "total = sum(processed)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(3)

        # Edit middle cell to cube instead of square
        nb_runner.set_cell_source(2, "processed = [x ** 3 for x in raw]")
        nb_runner.run_all()
        assert "total = 225" in nb_runner.get_output(3)

    def test_edit_first_of_4_cell_chain(self, nb_runner):
        """Edit the first cell in a 4-cell chain — all downstream recompute."""
        nb_runner.create_notebook(
            [
                "a = 2",
                "b = a * 3",
                "c = b + 1",
                "print(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 7" in nb_runner.get_output(4)

        # Edit first cell
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "c = 31" in nb_runner.get_output(4)

    def test_edit_adds_intermediate_step(self, nb_runner):
        """Edit middle cell to add an extra processing step in same cell."""
        nb_runner.create_notebook(
            [
                "data = [3, 1, 4, 1, 5, 9]",
                "result = sorted(data)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 1, 3, 4, 5, 9]" in nb_runner.get_output(2)

        # Add dedup step after sort
        nb_runner.set_cell_source(2, "result = sorted(set(data))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [1, 3, 4, 5, 9]" in nb_runner.get_output(2)

    def test_edit_preserves_unrelated_cells(self, nb_runner):
        """Editing one branch shouldn't affect an unrelated branch."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = 20",
                "sum_xy = x + y\nprint(f'sum = {sum_xy}')",
                "prod_xy = x * y\nprint(f'prod = {prod_xy}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum = 30" in nb_runner.get_output(3)
        assert "prod = 200" in nb_runner.get_output(4)

        # Edit y — both downstream cells should update
        nb_runner.set_cell_source(2, "y = 30")
        nb_runner.run_all()
        assert "sum = 40" in nb_runner.get_output(3)
        assert "prod = 300" in nb_runner.get_output(4)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestCellDependencyChain:
    """Test long dependency chains with selective execution."""

    def test_long_chain_head_change(self, nb_runner):
        """Change head of a 6-cell chain, re-run all."""
        nb_runner.create_notebook(
            [
                "base = 1",
                "step1 = base * 2",  # 2
                "step2 = step1 + 3",  # 5
                "step3 = step2 * 4",  # 20
                "step4 = step3 - 5",  # 15
                "print(step4)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "base = 10")
        nb_runner.run_all()
        # 10*2=20, +3=23, *4=92, -5=87
        assert "87" in nb_runner.get_output(6)


class TestComplexUpstreamPatterns:
    """Test complex upstream dependency resolution patterns."""

    @pytest.mark.upstream
    def test_diamond_dependency(self, nb_runner):
        """
        Cell 1 → Cell 2 and Cell 3 → Cell 4 (diamond).
        Modify cell 1, run cell 4 — should cascade through both paths.
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1",
                "b = x + 2",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 23" in nb_runner.get_output(4)

        # Modify the root
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(4)

        out = nb_runner.get_output(4)
        assert "c = 203" in out, f"Expected c=203, got: {out}"

    @pytest.mark.upstream
    def test_long_chain_six_cells(self, nb_runner):
        """Six-cell chain: each transforms the previous."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "x2 = x + 1",
                "x3 = x2 + 1",
                "x4 = x3 + 1",
                "x5 = x4 + 1",
                "x6 = x5 + 1\nprint(f'x6 = {x6}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x6 = 6" in nb_runner.get_output(6)

        # Modify root
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(6)

        out = nb_runner.get_output(6)
        assert "x6 = 105" in out, f"Expected x6=105, got: {out}"

    @pytest.mark.upstream
    def test_independent_branches_no_interference(self, nb_runner):
        """
        Two independent branches should not interfere with each other.
        Cell 1: x = 10
        Cell 2: a = x + 1
        Cell 3: y = 20 (independent)
        Cell 4: b = y + 1 (depends only on y)
        Modifying x should not re-execute cell 4.
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1\nprint(f'a = {a}')",
                "y = 20",
                "b = y + 1\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11" in nb_runner.get_output(2)
        assert "b = 21" in nb_runner.get_output(4)

        # Modify x — only cell 2 should change, cell 4 stays
        nb_runner.set_cell_source(1, "x = 99")
        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        out4 = nb_runner.get_output(4)
        assert "a = 100" in out2, f"Expected a=100, got: {out2}"
        assert "b = 21" in out4, f"Expected b=21 unchanged, got: {out4}"

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    @pytest.mark.upstream
    def test_diamond_dependency_on_run_all(self, nb_runner):
        """
        Diamond pattern: A → B, A → C, B+C → D.
        Changing A should propagate through both paths to D.
        """
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a * 2",  # b depends on a
                "c = a * 3",  # c depends on a
                "d = b + c",  # d depends on b and c
                "print(f'd={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "d=50" in output1  # 20+30

        # Change root
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "d=500" in output2  # 200+300

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    @pytest.mark.upstream
    def test_deep_dependency_chain(self, nb_runner):
        """Deep chain: a → b → c → d → e → f → result."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1",
                "f = e + 1",
                "result = f + 1",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(8)
        assert "result=7" in output1

        # Change root
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(8)
        assert "result=106" in output2

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    @pytest.mark.upstream
    def test_upstream_with_function_call(self, nb_runner):
        """Upstream should track through function definitions and calls."""
        nb_runner.create_notebook(
            [
                "def multiply(x, y): return x * y",
                "a = 5",
                "b = multiply(a, 3)",
                "print(f'b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "b=15" in output1

        # Change function definition
        nb_runner.set_cell_source(1, "def multiply(x, y): return x * y + 1")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "b=16" in output2

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    @pytest.mark.upstream
    def test_upstream_with_conditional_dependency(self, nb_runner):
        """Upstream tracks through conditionals that select different paths."""
        nb_runner.create_notebook(
            [
                "mode = 'add'",
                "x = 10",
                "if mode == 'add':\n    result = x + 100\nelse:\n    result = x * 100",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "result=110" in output1

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'multiply'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=1000" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestLargeScalePatterns:
    """Test patterns that stress the system at scale."""

    @pytest.mark.core
    def test_many_cells_sequential(self, nb_runner):
        """10 cells in a sequential chain."""
        cells = [f"x{i} = {f'x{i - 1} + 1' if i > 0 else '0'}" for i in range(10)]
        cells.append("print(f'x9: {x9}')")
        nb_runner.create_notebook(cells)
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(11)
        assert "x9: 9" in out

    @pytest.mark.core
    def test_many_variables_per_cell(self, nb_runner):
        """Single cell creating many variables, used in next cell."""
        setup = "\n".join(f"v{i} = {i * 10}" for i in range(15))
        use = "total = " + " + ".join(f"v{i}" for i in range(15))
        nb_runner.create_notebook(
            [
                setup,
                use,
                "print(f'Total: {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        expected = sum(i * 10 for i in range(15))
        assert f"Total: {expected}" in out

    @pytest.mark.core
    def test_diamond_with_intermediate_transforms(self, nb_runner):
        """Complex diamond: A → B, A → C, B → D, C → D with transforms."""
        nb_runner.create_notebook(
            [
                "a = [1, 2, 3, 4, 5]",
                "b = [x * 2 for x in a]",  # doubles
                "c = [x ** 2 for x in a]",  # squares
                "d = [bi + ci for bi, ci in zip(b, c)]",
                "print(f'D: {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(5)
        assert "D: [3, 8, 15, 24, 35]" in out1

        # Change source
        nb_runner.set_cell_source(1, "a = [10, 20]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(5)
        assert "D: [120, 440]" in out2  # [20+100, 40+400]


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestChainWithFunctions:
    """Deep chains involving function definitions."""

    def test_function_chain_edit(self, nb_runner):
        """Chain where each cell defines a function using the previous."""
        nb_runner.create_notebook(
            [
                "def step1(x):\n    return x + 1",
                "def step2(x):\n    return step1(x) * 2",
                "def step3(x):\n    return step2(x) + 10",
                "result = step3(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # step1(5)=6, step2(5)=12, step3(5)=22
        assert "result = 22" in nb_runner.get_output(4)

        # Edit step1
        nb_runner.set_cell_source(1, "def step1(x):\n    return x + 100")
        nb_runner.run_all()
        # step1(5)=105, step2(5)=210, step3(5)=220
        assert "result = 220" in nb_runner.get_output(4)

    def test_lambda_chain_edit(self, nb_runner):
        """Chain of lambda functions with edits."""
        nb_runner.create_notebook(
            [
                "fn1 = lambda x: x * 2",
                "fn2 = lambda x: fn1(x) + 3",
                "fn3 = lambda x: fn2(x) ** 2",
                "out = fn3(4)\nprint(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # fn1(4)=8, fn2(4)=11, fn3(4)=121
        assert "out = 121" in nb_runner.get_output(4)

        # Edit fn1
        nb_runner.set_cell_source(1, "fn1 = lambda x: x * 10")
        nb_runner.run_all()
        # fn1(4)=40, fn2(4)=43, fn3(4)=1849
        assert "out = 1849" in nb_runner.get_output(4)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestDependencyChainChanges:
    """Change which variables a cell depends on."""

    def test_switch_input_variable(self, nb_runner):
        """Switch which variable a cell reads."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 20",
                "result = a * 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Switch from a to b
        nb_runner.set_cell_source(3, "result = b * 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(3)

    def test_add_new_dependency(self, nb_runner):
        """Add a new dependency to an existing cell."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = 10",
                "result = x\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5" in nb_runner.get_output(3)

        # Now depend on both x and y
        nb_runner.set_cell_source(3, "result = x + y\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(3)

    def test_remove_dependency(self, nb_runner):
        """Remove a dependency from a cell."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 20",
                "result = a + b\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Remove dependency on b
        nb_runner.set_cell_source(3, "result = a * 5\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)

        # Now editing b should NOT affect result
        nb_runner.set_cell_source(2, "b = 999")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestCyclicLikePatterns:
    """Patterns that look cyclic but aren't (self-assignment chains)."""

    def test_self_assignment_chain(self, nb_runner):
        """x depends on previous x (sequential mutation pattern)."""
        nb_runner.create_notebook(
            [
                "x = [1]",
                "x = x + [2]  # extend step 1",
                "x = x + [3]  # extend step 2",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = [1, 2, 3]" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "x = x + [20]  # extend step 1 (modified)")
        nb_runner.run_all()
        assert "x = [1, 20, 3]" in nb_runner.get_output(4)

    def test_accumulating_string(self, nb_runner):
        """String accumulation pattern."""
        nb_runner.create_notebook(
            [
                "s = 'hello'",
                "s = s + ' world'  # add world",
                "s = s + '!'  # add exclamation",
                "print(f's = {s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = hello world!" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "s = s + ' python'  # add python")
        nb_runner.run_all()
        assert "s = hello python!" in nb_runner.get_output(4)
