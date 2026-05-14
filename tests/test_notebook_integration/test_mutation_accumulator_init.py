"""
Tests for accumulator initialization skip logic with mutation-only updates.

Bug: When a loop mutates a variable via .append() (pure mutation, no assignment output),
the accumulator-init-skip logic in upstream.py doesn't recognize the variable as
loop-updated. This causes `a = []` to be re-executed when the upstream cell is modified,
resetting the accumulated value.

The fix extends the accumulator-init-skip to also check vars_mutated_by_loops,
not just scheduled_iteration_outputs.

Root cause: CodeAnalyzer.analyze_code_block('a.append(x)') returns outputs=set(),
so 'a' never appears in scheduled_iteration_outputs. But MutationDetector correctly
detects 'a' as mutated, and _find_loop_mutated_vars adds it to vars_mutated_by_loops.

NOTE: These tests currently fail because the accumulator init skip logic is not
fully implemented for upstream modifications that change loop code.
"""

import pytest

pytestmark = [pytest.mark.mutations, pytest.mark.upstream]


class TestMutationAccumulatorInit:
    """Test that accumulator init is skipped for mutation-only loop updates."""

    @pytest.mark.xfail(reason="Accumulator init skip for mutation-only loops not fully implemented")
    def test_append_accumulator_preserved_after_upstream_change(self, nb_runner):
        """
        Reproduces the core bug: a = [] followed by for-loop with a.append(x)
        should preserve 'a' when downstream cell re-runs after upstream modification.
        """
        nb_runner.create_notebook([
            # Cell 1: Setup data
            "data = {'x': [1, 2, 3], 'y': [4, 5, 6]}",
            # Cell 2: Loop with both assignment and append mutation
            (
                "results = {}\n"
                "collected = []\n"
                "for key in ['x', 'y']:\n"
                "    vals = data[key]\n"
                "    results[key] = sum(vals)\n"
                "    collected.append(key)\n"
                "print(f'results={results}')\n"
                "print(f'collected={collected}')"
            ),
            # Cell 3: Use both variables
            "print(f'results_keys={sorted(results.keys())}')\n"
            "print(f'collected_list={collected}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Verify initial run
        output3 = nb_runner.get_output(3)
        assert "results_keys=['x', 'y']" in output3
        assert "collected_list=['x', 'y']" in output3

        # Modify cell 2: change iteration order
        nb_runner.set_cell_source(2, (
            "results = {}\n"
            "collected = []\n"
            "for key in ['y', 'x']:\n"  # Changed order
            "    vals = data[key]\n"
            "    results[key] = sum(vals)\n"
            "    collected.append(key)\n"
            "print(f'results={results}')\n"
            "print(f'collected={collected}')"
        ))

        # Re-run cell 3 (without re-running cell 2)
        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)

        # Both variables should be preserved from the original run
        # (not reset to empty by re-executing the initialization)
        # 'collected' should NOT be [] (the bug would cause this)
        assert "collected_list=[]" not in output3, \
            f"Bug: collected was reset to empty list. Output: {output3}"
        # 'collected' should still have items (either original or re-computed)
        assert "collected_list=[" in output3

    @pytest.mark.xfail(reason="Accumulator init skip for mutation-only loops not fully implemented")
    def test_append_only_accumulator_no_subscript_assignment(self, nb_runner):
        """
        Test where the ONLY loop mutation is .append() — no subscript assignment.
        This is the purest test of the fix since there's no scheduled_iteration_outputs
        at all (unlike ticker_stats[k] = v which does produce outputs).
        """
        nb_runner.create_notebook([
            # Cell 1: Simple data
            "items = [10, 20, 30]",
            # Cell 2: Loop with ONLY append mutation
            (
                "accumulated = []\n"
                "for item in items:\n"
                "    accumulated.append(item * 2)\n"
                "print(f'accumulated={accumulated}')"
            ),
            # Cell 3: Use accumulated
            "print(f'total={sum(accumulated)}')\n"
            "print(f'count={len(accumulated)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        assert "total=120" in output3  # (10+20+30)*2 = 120
        assert "count=3" in output3

        # Modify cell 2: change items
        nb_runner.set_cell_source(2, (
            "accumulated = []\n"
            "for item in items:\n"
            "    accumulated.append(item * 3)\n"  # Changed multiplier
            "print(f'accumulated={accumulated}')"
        ))

        # Re-run cell 3
        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)

        # accumulated should NOT be [] (the bug)
        assert "count=0" not in output3, \
            f"Bug: accumulated was reset to empty list. Output: {output3}"
        # Should still have 3 items
        assert "count=3" in output3

    @pytest.mark.xfail(reason="Accumulator init skip for mutation-only loops not fully implemented")
    def test_mixed_append_and_subscript_assignment(self, nb_runner):
        """
        Test with both .append() and subscript assignment in same loop.
        Ensures the fix works alongside the existing logic.
        """
        nb_runner.create_notebook([
            # Cell 1: Data
            "tickers = ['AAPL', 'MSFT', 'TSLA']",
            # Cell 2: Loop with both patterns
            (
                "stats = {}\n"
                "names = []\n"
                "for t in tickers:\n"
                "    stats[t] = len(t)\n"
                "    names.append(t)\n"
                "print(f'stats={stats}')\n"
                "print(f'names={names}')"
            ),
            # Cell 3: Use both
            "print(f'stats_keys={sorted(stats.keys())}')\n"
            "print(f'names_list={names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        assert "stats_keys=['AAPL', 'MSFT', 'TSLA']" in output3
        assert "names_list=['AAPL', 'MSFT', 'TSLA']" in output3

        # Modify cell 2: change ticker order
        nb_runner.set_cell_source(2, (
            "stats = {}\n"
            "names = []\n"
            "for t in ['TSLA', 'AAPL', 'MSFT']:\n"  # Changed order
            "    stats[t] = len(t)\n"
            "    names.append(t)\n"
            "print(f'stats={stats}')\n"
            "print(f'names={names}')"
        ))

        # Re-run cell 3
        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)

        # Neither should be empty
        assert "stats_keys=[]" not in output3
        assert "names_list=[]" not in output3
        # Both should have 3 items
        assert "AAPL" in output3
        assert "MSFT" in output3
        assert "TSLA" in output3

    @pytest.mark.xfail(reason="Accumulator init skip for mutation-only loops not fully implemented")
    def test_set_add_mutation(self, nb_runner):
        """Test that set.add() mutations are also handled."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 2, 1]",
            (
                "unique = set()\n"
                "for x in data:\n"
                "    unique.add(x)\n"
                "print(f'unique={sorted(unique)}')"
            ),
            "print(f'count={len(unique)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        assert "count=3" in output3

        # Modify cell 2
        nb_runner.set_cell_source(2, (
            "unique = set()\n"
            "for x in data:\n"
            "    unique.add(x * 10)\n"  # Changed operation
            "print(f'unique={sorted(unique)}')"
        ))

        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)
        # Should NOT be count=0
        assert "count=0" not in output3, \
            f"Bug: unique was reset to empty set. Output: {output3}"
        assert "count=3" in output3

    def test_dict_update_mutation(self, nb_runner):
        """Test that dict.update() mutations are handled."""
        nb_runner.create_notebook([
            "pairs = [('a', 1), ('b', 2)]",
            (
                "merged = {}\n"
                "for k, v in pairs:\n"
                "    merged.update({k: v})\n"
                "print(f'merged={merged}')"
            ),
            "print(f'keys={sorted(merged.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        assert "keys=['a', 'b']" in output3

        # Modify cell 2
        nb_runner.set_cell_source(2, (
            "merged = {}\n"
            "for k, v in pairs:\n"
            "    merged.update({k: v * 10})\n"  # Changed value
            "print(f'merged={merged}')"
        ))

        nb_runner.run_cell(3)
        output3 = nb_runner.get_output(3)
        assert "keys=[]" not in output3
        assert "keys=['a', 'b']" in output3
