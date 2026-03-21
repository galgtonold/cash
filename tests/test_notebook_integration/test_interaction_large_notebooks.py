"""Batch 148 – Large notebook (10+ cells) interaction tests.

Tests with larger notebooks that simulate real-world scenarios
with many cells, edits at various positions, and full run-through.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestTenCellWorkflow:
    """10-cell workflow with edits."""

    def test_ten_cell_data_pipeline(self, nb_runner):
        """10-cell data pipeline, edit source and check propagation."""
        nb_runner.create_notebook([
            "raw = list(range(1, 11))",         # cell 1
            "cleaned = [x for x in raw if x > 0]",  # cell 2
            "normalized = [x / max(cleaned) for x in cleaned]",  # cell 3
            "scaled = [x * 100 for x in normalized]",  # cell 4
            "rounded = [round(x) for x in scaled]",  # cell 5
            "top5 = sorted(rounded, reverse=True)[:5]",  # cell 6
            "bottom5 = sorted(rounded)[:5]",  # cell 7
            "spread = top5[0] - bottom5[0]",  # cell 8
            "avg = sum(rounded) / len(rounded)",  # cell 9
            "print(f'spread = {spread}, avg = {avg}')",  # cell 10
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(10)
        assert "spread = " in output
        assert "avg = " in output

        # Edit source data
        nb_runner.set_cell_source(1, "raw = list(range(1, 51))")
        nb_runner.run_all()
        output2 = nb_runner.get_output(10)
        assert "spread = " in output2
        assert "avg = " in output2

    def test_ten_cell_edit_middle(self, nb_runner):
        """10-cell chain, edit cell 5 in the middle."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "e = d + 1",
            "f = e + 1",
            "g = f + 1",
            "h = g + 1",
            "i = h + 1",
            "print(f'i = {i}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "i = 9" in nb_runner.get_output(10)

        # Edit cell 5 to multiply instead of add
        nb_runner.set_cell_source(5, "e = d * 10")
        nb_runner.run_all()
        # a=1,b=2,c=3,d=4,e=40,f=41,g=42,h=43,i=44
        assert "i = 44" in nb_runner.get_output(10)


class TestEightCellWithBranching:
    """8-cell notebook with branching dependencies."""

    def test_eight_cell_split_merge(self, nb_runner):
        """8-cell notebook: shared root, two branches, merge."""
        nb_runner.create_notebook([
            "root = 10",                              # cell 1
            "branch_a1 = root * 2",                   # cell 2
            "branch_a2 = branch_a1 + 5",              # cell 3
            "branch_b1 = root + 3",                   # cell 4
            "branch_b2 = branch_b1 * 4",              # cell 5
            "merged = branch_a2 + branch_b2",         # cell 6
            "final = merged * 2",                     # cell 7
            "print(f'final = {final}')",              # cell 8
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a1=20, a2=25, b1=13, b2=52, merged=77, final=154
        assert "final = 154" in nb_runner.get_output(8)

        # Edit root
        nb_runner.set_cell_source(1, "root = 100")
        nb_runner.run_all()
        # a1=200, a2=205, b1=103, b2=412, merged=617, final=1234
        assert "final = 1234" in nb_runner.get_output(8)

        # Edit one branch
        nb_runner.set_cell_source(4, "branch_b1 = root - 50")
        nb_runner.run_all()
        # a1=200, a2=205, b1=50, b2=200, merged=405, final=810
        assert "final = 810" in nb_runner.get_output(8)


class TestProgressiveNotebookBuilding:
    """Simulate building notebook progressively — add cells one by one."""

    def test_build_notebook_incrementally(self, nb_runner):
        """Start with 2 cells, progressively add more."""
        # Start small
        nb_runner.create_notebook([
            "x = 5",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 5" in nb_runner.get_output(2)

        # Now extend by recreating with more cells
        nb_runner.shutdown()
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 2",
            "print(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(3)

        # Extend again
        nb_runner.shutdown()
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 2",
            "z = y + x",
            "print(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 15" in nb_runner.get_output(4)
