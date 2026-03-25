"""Integration tests for loop-produced variable upstream detection.

Reproduces the bug where a variable assigned inside a for-loop body
(e.g. ``ticker_data = df[...]``) is not correctly re-established by
upstream simulation when it has been overwritten between executions.
"""
import pytest

pytestmark = [pytest.mark.loops, pytest.mark.upstream]


class TestLoopProducedVariableUpstream:
    """Variables assigned inside loops should be restored by upstream simulation."""

    def test_loop_variable_restored_after_overwrite(self, nb_runner):
        """A variable assigned inside a loop should be re-computed when queried
        after being manually overwritten.

        Scenario:
        - Cell 1: for-loop assigns ``item_data``
        - Cell 2: queries ``item_data``
        - User overwrites ``item_data = 123`` in a new cell
        - Re-running cell 2 should restore the loop-produced value, not keep 123
        """
        nb_runner.create_notebook([
            # Cell 1: loop that assigns a variable
            (
                "data = {'a': 10, 'b': 20, 'c': 30}\n"
                "results = {}\n"
                "for key in ['a', 'b', 'c']:\n"
                "    item_data = data[key]\n"
                "    results[key] = item_data * 2\n"
                "print(f'results={results}')"
            ),
            # Cell 2: use the loop-produced variable
            (
                "print(f'item_data={item_data}')"
            ),
            # Cell 3: overwrite the variable
            (
                "item_data = 999\n"
                "print(f'overwritten={item_data}')"
            ),
        ])
        nb_runner.start_kernel()

        # Run all cells
        nb_runner.run_all()
        assert "results={'a': 20, 'b': 40, 'c': 60}" in nb_runner.get_output(1)
        # item_data should be 30 (last iteration: data['c'])
        assert 'item_data=30' in nb_runner.get_output(2)
        assert 'overwritten=999' in nb_runner.get_output(3)

        # Now re-run cell 2 — upstream should detect item_data was
        # produced by cell 1 and restore it, not keep 999
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert 'item_data=30' in output, (
            f"Loop-produced variable should be restored by upstream. Got: {output}"
        )

    def test_loop_variable_not_stale_after_intervening_cell(self, nb_runner):
        """A variable produced in a loop should track its lineage correctly
        so downstream cells detect when it's stale."""
        nb_runner.create_notebook([
            # Cell 1: loop producing stats
            (
                "stats = {}\n"
                "for name in ['x', 'y', 'z']:\n"
                "    val = len(name) * 10\n"
                "    stats[name] = val\n"
                "print(f'val={val}')"
            ),
            # Cell 2: use val (last iteration value)
            (
                "result = val + 100\n"
                "print(f'result={result}')"
            ),
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        assert 'val=10' in nb_runner.get_output(1)
        assert 'result=110' in nb_runner.get_output(2)

        # Re-run cell 2 alone — should still produce result=110
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert 'result=110' in output, (
            f"Expected result=110 from loop-produced val. Got: {output}"
        )

    def test_loop_variable_updated_when_loop_changes(self, nb_runner):
        """When loop code changes, downstream cells using loop-produced vars
        should get updated values."""
        nb_runner.create_notebook([
            # Cell 1: loop assigns last_item
            (
                "items = []\n"
                "for x in [1, 2, 3]:\n"
                "    last_item = x * 10\n"
                "    items.append(last_item)\n"
                "print(f'items={items}')"
            ),
            # Cell 2: uses last_item
            (
                "print(f'last_item={last_item}')"
            ),
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        assert 'items=[10, 20, 30]' in nb_runner.get_output(1)
        assert 'last_item=30' in nb_runner.get_output(2)

        # Change the loop multiplier
        nb_runner.set_cell_source(1, (
            "items = []\n"
            "for x in [1, 2, 3]:\n"
            "    last_item = x * 100\n"
            "    items.append(last_item)\n"
            "print(f'items={items}')"
        ))

        # Re-run cell 2 — upstream should re-execute cell 1 with new code
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert 'last_item=300' in output, (
            f"Loop-produced variable should reflect updated loop code. Got: {output}"
        )

    def test_upstream_loop_reexecution_uses_per_iteration_cache(self, nb_runner):
        """When upstream re-executes a loop, it should use per-iteration caching
        so cached iterations are restored rather than re-computed.

        Scenario:
        - Cell 1: for-loop with a slow body statement
        - Cell 2: uses a loop-produced variable
        - Cell 3: overwrites the loop-produced variable
        - Re-running cell 2 should trigger upstream re-execution of the loop,
          but individual iterations should be restored from cache (fast).
        """
        import time

        nb_runner.create_notebook([
            # Cell 1: loop with a slow body
            (
                "import time\n"
                "data = {'a': 10, 'b': 20, 'c': 30}\n"
                "results = {}\n"
                "for key in ['a', 'b', 'c']:\n"
                "    time.sleep(0.5)\n"
                "    item_data = data[key]\n"
                "    results[key] = item_data * 2\n"
                "print(f'results={results}')"
            ),
            # Cell 2: use the loop-produced variable
            (
                "print(f'item_data={item_data}')"
            ),
            # Cell 3: overwrite the variable
            (
                "item_data = 999\n"
                "print(f'overwritten={item_data}')"
            ),
        ])
        nb_runner.start_kernel()

        # Run all cells — this populates the per-iteration cache
        nb_runner.run_all()
        assert "results={'a': 20, 'b': 40, 'c': 60}" in nb_runner.get_output(1)
        assert 'item_data=30' in nb_runner.get_output(2)
        assert 'overwritten=999' in nb_runner.get_output(3)

        # Re-run cell 2 — upstream should re-execute the loop but use
        # cached per-iteration results (should be much faster than 1.5s)
        start = time.time()
        nb_runner.run_cell(2)
        elapsed = time.time() - start
        output = nb_runner.get_output(2)

        assert 'item_data=30' in output, (
            f"Loop-produced variable should be restored. Got: {output}"
        )
        # If per-iteration cache is used, this should take < 1s total
        # (vs 1.5s+ if all 3 iterations recompute with 0.5s sleep each)
        assert elapsed < 3.0, (
            f"Upstream loop re-execution should use per-iteration cache. "
            f"Took {elapsed:.1f}s (expected < 3s)"
        )
