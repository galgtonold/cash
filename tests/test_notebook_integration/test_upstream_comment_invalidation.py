"""
Tests for upstream cell modification invalidation.

When a user modifies an upstream cell (e.g., comments out a line) and then
runs a downstream cell, the upstream checker should detect the change and
re-execute the upstream cell so the downstream cell sees the updated values.

Bug scenario:
  Cell A: x = {}; x['a'] = 234; print(x)
  Cell B: x

  1. Run cell A → x = {'a': 234}
  2. Run cell B → displays {'a': 234} ✓
  3. Comment out x['a'] = 234 in cell A (don't re-run it)
  4. Run cell B → should display {} but shows {'a': 234} from cache ✗

The upstream checker reads the notebook file from disk. When the file reflects
the commented-out line, the simulation should compute a different lineage for
x (without the mutation), detect that the actual x has a stale lineage, and
re-execute cell A before running cell B.
"""
import pytest

pytestmark = pytest.mark.upstream


class TestUpstreamCommentInvalidation:
    """Commenting out code in an upstream cell should invalidate downstream cache."""

    def test_comment_out_mutation_invalidates_downstream(self, nb_runner):
        """
        Cell 1: x = {}; x['a'] = 234; print(x)
        Cell 2: x

        After commenting out x['a'] = 234 in cell 1 and running cell 2,
        x should be {} not {'a': 234}.
        """
        nb_runner.create_notebook([
            "x = {}\nx['a'] = 234\nprint(x)",
            "print(x)",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # Step 1: Run both cells normally
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "'a': 234" in out1 or '"a": 234' in out1 or "a" in out1, (
            f"Expected x to contain 'a' after first run, got: {out1}"
        )

        # Step 2: Comment out the mutation line in cell 1 (don't run cell 1)
        nb_runner.set_cell_source(1, "x = {}\n#x['a'] = 234\nprint(x)")

        # Step 3: Run cell 2 only - upstream checker should detect cell 1 changed
        nb_runner.run_cell(2)
        out2 = nb_runner.get_output(2)
        out2_raw = nb_runner.get_raw_output(2)

        # x should now be {} because the upstream checker should have re-executed
        # cell 1 with the commented-out line
        assert "'a'" not in out2, (
            f"Expected x to be empty dict {{}}, but got: {out2}\n"
            f"Raw output:\n{out2_raw}"
        )

    def test_add_line_to_upstream_invalidates_downstream(self, nb_runner):
        """
        Cell 1: x = 10
        Cell 2: print(x)

        Modify cell 1 to x = 20 and run cell 2 → should see 20.
        """
        nb_runner.create_notebook([
            "x = 10",
            "print(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # Run both cells
        nb_runner.run_all()
        assert "x=10" in nb_runner.get_output(2)

        # Modify cell 1 (don't re-run it)
        nb_runner.set_cell_source(1, "x = 20")

        # Run cell 2 only
        nb_runner.run_cell(2)
        out = nb_runner.get_output(2)
        assert "x=20" in out, (
            f"Expected x=20 after upstream modification, got: {out}"
        )

    def test_uncomment_line_in_upstream_invalidates_downstream(self, nb_runner):
        """
        Cell 1: y = 5; #y = y * 10
        Cell 2: print(y)

        Uncomment y = y * 10 → downstream should see 50.
        """
        nb_runner.create_notebook([
            "y = 5\n#y = y * 10",
            "print(f'y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # Run both cells
        nb_runner.run_all()
        assert "y=5" in nb_runner.get_output(2)

        # Uncomment the line in cell 1
        nb_runner.set_cell_source(1, "y = 5\ny = y * 10")

        # Run cell 2 only
        nb_runner.run_cell(2)
        out = nb_runner.get_output(2)
        assert "y=50" in out, (
            f"Expected y=50 after uncommenting upstream line, got: {out}"
        )

    def test_multi_cell_chain_comment_invalidation(self, nb_runner):
        """
        Cell 1: data = [1, 2, 3]
        Cell 2: data = data + [4]; print(len(data))
        Cell 3: print(data)

        Comment out data = data + [4] in cell 2, run cell 3 → should see [1, 2, 3].
        """
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "data = data + [4]\nprint(f'len={len(data)}')",
            "print(data)",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # Run all cells
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "4" in out, f"Expected data to contain 4, got: {out}"

        # Comment out data = data + [4] in cell 2
        nb_runner.set_cell_source(2, "#data = data + [4]\nprint(f'len={len(data)}')")

        # Run cell 3 only - should detect cell 2 changed
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "4" not in out, (
            f"Expected data without 4 after commenting out concat, got: {out}"
        )
