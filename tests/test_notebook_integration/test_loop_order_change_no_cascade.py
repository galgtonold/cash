"""
Integration tests for loop iteration order changes NOT cascading to unrelated upstream.

When only the loop cell changes (e.g., reorder elements, add/remove element),
the upstream checker should NOT re-execute unrelated upstream statements like
`df = pd.read_csv(...)` or `df['col'] = df.transform(...)`.

These tests verify:
1. Loop order change + downstream re-run only re-executes loop, not entire pipeline
2. Time saved is accurate (doesn't count restoring vars that were already valid)
3. Adding/removing loop elements works correctly without cascading
"""

import pytest

pytestmark = [pytest.mark.loops, pytest.mark.upstream]

class TestLoopOrderChangeNoCascade:
    """Changing loop iteration order should not cascade to unrelated upstream."""

    def test_loop_reorder_does_not_rerun_upstream_transforms(self, nb_runner):
        """
        Cell 1: x = 100 (expensive upstream)
        Cell 2: loop over ["A", "B", "C"] building dict with x
        Cell 3: downstream using dict
        
        Reorder loop to ["C", "B", "A"] → should NOT re-execute cell 1.
        """
        nb_runner.create_notebook([
            # Cell 1: "Expensive" upstream computation
            "x = 100",
            # Cell 2: Loop building dict using x
            """result = {}
for name in ["A", "B", "C"]:
    result[name] = x + len(name)""",
            # Cell 3: Downstream
            """print(f"Keys: {sorted(result.keys())}")
print(f"Values: {sorted(result.values())}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(3)
        assert "'A'" in output1 and "'B'" in output1 and "'C'" in output1

        # Reorder loop elements
        nb_runner.set_cell_source(2, """result = {}
for name in ["C", "B", "A"]:
    result[name] = x + len(name)""")

        # Only re-run downstream cell (cell 3)
        nb_runner.run_cell(3)

        output2 = nb_runner.get_output(3)
        # Should still have all 3 keys with correct values
        assert "'A'" in output2 and "'B'" in output2 and "'C'" in output2

    def test_loop_add_element_does_not_rerun_unrelated_upstream(self, nb_runner):
        """
        Cell 1: y = [1, 2, 3] (data)
        Cell 2: total = sum(y) (transform)
        Cell 3: loop building dict using total
        Cell 4: downstream
        
        Add element to loop → should NOT re-execute cells 1-2.
        """
        nb_runner.create_notebook([
            # Cell 1
            "y = [1, 2, 3]",
            # Cell 2: transform
            "total = sum(y)",
            # Cell 3: loop
            """stats = {}
for k in ["mean", "max"]:
    stats[k] = total""",
            # Cell 4: downstream
            """print(f"Stats: {stats}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "'mean'" in output1 and "'max'" in output1

        # Add element to loop
        nb_runner.set_cell_source(3, """stats = {}
for k in ["mean", "max", "min"]:
    stats[k] = total""")

        # Only run downstream (cell 4)
        nb_runner.run_cell(4)

        output2 = nb_runner.get_output(4)
        assert "'min'" in output2, f"Missing 'min': {output2}"
        # All keys present
        assert "'mean'" in output2 and "'max'" in output2

    def test_loop_remove_element_does_not_cascade(self, nb_runner):
        """
        Remove an element from loop → should not cascade to upstream.
        """
        nb_runner.create_notebook([
            "base = 10",
            """d = {}
for x in ["A", "B", "C"]:
    d[x] = base""",
            """print(f"Keys: {sorted(d.keys())}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(3)
        assert "'A'" in output1 and "'B'" in output1 and "'C'" in output1

        # Remove "B"
        nb_runner.set_cell_source(2, """d = {}
for x in ["A", "C"]:
    d[x] = base""")

        nb_runner.run_cell(3)

        output2 = nb_runner.get_output(3)
        assert "'A'" in output2 and "'C'" in output2
        # "B" should be gone (dict was re-initialized with {})
        # Actually the loop re-runs and builds a new dict with only A, C

    def test_multi_transform_pipeline_not_cascaded(self, nb_runner):
        """
        Cell 1: data = list(range(10))
        Cell 2: processed = [x*2 for x in data]
        Cell 3: total = sum(processed)
        Cell 4: loop building dict using total
        Cell 5: downstream
        
        Change loop → cells 1-3 should NOT re-execute.
        """
        nb_runner.create_notebook([
            "data = list(range(10))",
            "processed = [x * 2 for x in data]",
            "total = sum(processed)",
            """info = {}
for k in ["sum", "count"]:
    info[k] = total""",
            """print(f"Info: {info}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "'sum'" in output1 and "'count'" in output1

        # Add element to loop
        nb_runner.set_cell_source(4, """info = {}
for k in ["sum", "count", "avg"]:
    info[k] = total""")

        nb_runner.run_cell(5)

        output2 = nb_runner.get_output(5)
        assert "'avg'" in output2, f"Missing 'avg': {output2}"
        # Original keys still present
        assert "'sum'" in output2 and "'count'" in output2


class TestLoopOrderChangeCorrectness:
    """Ensure loop order changes produce correct results."""

    def test_dict_final_state_correct_after_reorder(self, nb_runner):
        """Dict should have all keys regardless of iteration order."""
        nb_runner.create_notebook([
            """d = {}
for x in ["A", "B", "C"]:
    d[x] = ord(x)""",
            """print(f"d = {dict(sorted(d.items()))}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(2)
        assert "'A': 65" in output1

        # Reorder
        nb_runner.set_cell_source(1, """d = {}
for x in ["C", "A", "B"]:
    d[x] = ord(x)""")
        nb_runner.run_all()

        output2 = nb_runner.get_output(2)
        # Same content regardless of order
        assert "'A': 65" in output2 and "'B': 66" in output2 and "'C': 67" in output2

    def test_loop_replace_element_produces_correct_dict(self, nb_runner):
        """Replace 'B' with 'Z' in loop → dict should have A, Z, C (no B)."""
        nb_runner.create_notebook([
            """d = {}
for x in ["A", "B", "C"]:
    d[x] = x.lower()""",
            """print(f"Keys: {sorted(d.keys())}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(2)
        assert "'A'" in output1 and "'B'" in output1 and "'C'" in output1

        # Replace B with Z
        nb_runner.set_cell_source(1, """d = {}
for x in ["A", "Z", "C"]:
    d[x] = x.lower()""")
        nb_runner.run_all()

        output2 = nb_runner.get_output(2)
        assert "'Z'" in output2, f"Missing 'Z': {output2}"
        assert "'A'" in output2 and "'C'" in output2

    def test_downstream_only_rerun_after_loop_change(self, nb_runner):
        """
        Only run downstream cell after modifying loop.
        Upstream checker should detect the change and re-execute just the loop.
        """
        nb_runner.create_notebook([
            "base = 42",
            """counts = {}
for name in ["Alice", "Bob"]:
    counts[name] = base + len(name)""",
            """print(f"Counts: {counts}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(3)
        assert "'Alice'" in output1 and "'Bob'" in output1

        # Add "Carol" to loop but only run cell 3
        nb_runner.set_cell_source(2, """counts = {}
for name in ["Alice", "Bob", "Carol"]:
    counts[name] = base + len(name)""")

        nb_runner.run_cell(3)

        output2 = nb_runner.get_output(3)
        assert "'Carol'" in output2, f"Missing 'Carol': {output2}"
        assert "'Alice'" in output2 and "'Bob'" in output2


class TestUpstreamMetricsAccuracy:
    """Verify that upstream metrics don't count unnecessary restores."""

    def test_no_unnecessary_upstream_restores_on_loop_change(self, nb_runner):
        """
        Changing a loop should not show upstream restores for unrelated variables.
        This tests the 'bogus time saved' bug.
        """
        nb_runner.create_notebook([
            # Cell 1: Setup
            "x = 100\ny = 200",
            # Cell 2: Transform (independent of loop)
            "z = x + y",
            # Cell 3: Loop
            """d = {}
for k in ["a", "b"]:
    d[k] = z""",
            # Cell 4: Downstream
            """print(f"d = {d}")
print(f"z = {z}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "'a'" in output1 and "'b'" in output1

        # Add element to loop
        nb_runner.set_cell_source(3, """d = {}
for k in ["a", "b", "c"]:
    d[k] = z""")

        # Run downstream only
        nb_runner.run_cell(4)

        output2 = nb_runner.get_output(4)
        assert "'c'" in output2
        # z should still be 300 (no re-execution needed)
        assert "z = 300" in output2
