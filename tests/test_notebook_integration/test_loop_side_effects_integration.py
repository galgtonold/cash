"""
Integration tests for loop caching with side effects and upstream re-execution.

Uses the nb_runner fixture to test with real notebook files and Jupyter kernels.
These tests validate end-to-end behavior of loop caching including:
1. Dict/list mutation across cells with upstream changes
2. Multi-variable loop mutation patterns
3. Upstream re-execution with loop-containing cells
4. Various iteration and mutation types
"""

import pytest

pytestmark = [pytest.mark.loops, pytest.mark.mutations]


# ============================================================================
# Group 1: The Reported Bug — Loop Code Change + Downstream
# ============================================================================

class TestLoopCodeChangeBug:
    """Tests for the specific reported bug: loop-mutated variables stale when loop code changes."""

    def test_loop_dict_mutation_stale_on_code_change(self, nb_runner):
        """
        THE reported bug: loop builds dict, downstream cell uses dict.keys(),
        loop code changes (add item) → downstream must see ALL items.
        """
        # First run: cache 4 iterations
        nb_runner.create_notebook([
            # Cell 1: Loop building a dict
            """results = {}
for x in ["A", "B", "C", "D"]:
    results[x] = x * 2""",
            # Cell 2: Downstream cell using the dict
            """print(f"Keys: {sorted(results.keys())}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "'A'" in output and "'D'" in output

        # Modify loop to add "E"
        nb_runner.set_cell_source(1, """results = {}
for x in ["A", "B", "C", "D", "E"]:
    results[x] = x * 2""")

        # Run all cells (loop cell re-executes, downstream sees new value)
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "'E'" in output, f"Missing 'E' in output: {output}"
        # Should have all 5
        for letter in ['A', 'B', 'C', 'D', 'E']:
            assert f"'{letter}'" in output, f"Missing '{letter}' in output: {output}"

    def test_loop_dict_mutation_downstream_only_rerun(self, nb_runner):
        """
        Change loop code but only re-run the DOWNSTREAM cell.
        The upstream checker should detect the change and re-execute the loop.
        """
        nb_runner.create_notebook([
            # Cell 1: Loop
            """data = {}
for k in ["x", "y"]:
    data[k] = len(k)""",
            # Cell 2: Downstream
            """print(f"Keys: {sorted(data.keys())}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(2)
        assert "'x'" in output1 and "'y'" in output1

        # Modify loop cell to add "z", but ONLY re-run downstream (cell 2)
        nb_runner.set_cell_source(1, """data = {}
for k in ["x", "y", "z"]:
    data[k] = len(k)""")

        nb_runner.run_cell(2)

        output2 = nb_runner.get_output(2)
        assert "'z'" in output2, f"Missing 'z' after upstream change: {output2}"

    def test_loop_list_mutation_stale_on_code_change(self, nb_runner):
        """Same pattern but with list.append() instead of dict subscript."""
        nb_runner.create_notebook([
            """items = []
for x in [1, 2, 3]:
    items.append(x * 10)""",
            """print(f"Items: {items}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "10" in output and "20" in output and "30" in output

        # Add item 4
        nb_runner.set_cell_source(1, """items = []
for x in [1, 2, 3, 4]:
    items.append(x * 10)""")

        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "40" in output, f"Missing 40 after adding item: {output}"


# ============================================================================
# Group 2: Multi-Variable Loop Mutation
# ============================================================================

class TestMultiVariableLoopMutation:
    """Tests for loops that mutate multiple variables."""

    def test_loop_mutates_dict_and_prints(self, nb_runner):
        """Dict mutation + print in loop body (like the financial demo)."""
        nb_runner.create_notebook([
            """stats = {}
for name in ["Alice", "Bob"]:
    stats[name] = len(name)
    print(f"Processed {name}")""",
            """print(f"Stats: {stats}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "'Alice'" in output and "'Bob'" in output

    def test_loop_mutates_two_dicts(self, nb_runner):
        """Two separate dicts mutated in same loop."""
        nb_runner.create_notebook([
            """names = {}
lengths = {}
for x in ["cat", "dog", "fish"]:
    names[x] = x.upper()
    lengths[x] = len(x)""",
            """print(f"Names: {sorted(names.keys())}")
print(f"Lengths: {sorted(lengths.keys())}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "'cat'" in output and "'dog'" in output and "'fish'" in output

    def test_loop_augmented_assign_and_dict(self, nb_runner):
        """total += x and d[k] = v in same loop."""
        nb_runner.create_notebook([
            """total = 0
details = {}
for x in [10, 20, 30]:
    total += x
    details[x] = total""",
            """print(f"Total: {total}")
print(f"Details: {details}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "Total: 60" in output
        assert "10: 10" in output or "{10: 10" in output


# ============================================================================
# Group 3: Upstream Re-execution with Loops
# ============================================================================

class TestUpstreamReexecutionWithLoops:
    """Tests for upstream re-execution when loop-containing cells change."""

    def test_upstream_reexecutes_loop_when_input_changes(self, nb_runner):
        """
        Cell 1 defines data, cell 2 loops over it, cell 3 uses result.
        Change cell 1 → cell 3 should get updated loop results.
        """
        nb_runner.create_notebook([
            # Cell 1: Data definition
            """items = ["A", "B"]""",
            # Cell 2: Loop over data
            """result = {}
for x in items:
    result[x] = x.lower()""",
            # Cell 3: Use result
            """print(f"Result: {sorted(result.keys())}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "'A'" in output and "'B'" in output

        # Change the data
        nb_runner.set_cell_source(1, """items = ["A", "B", "C"]""")

        # Run all cells — cell 2 loop should run with new data
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "'C'" in output, f"Missing 'C' after data change: {output}"

    def test_upstream_skips_unchanged_loop(self, nb_runner):
        """Same loop code, no changes → downstream should use cache."""
        nb_runner.create_notebook([
            """data = {}
for k in ["x", "y"]:
    data[k] = 1""",
            """print(f"Data: {data}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(2)
        assert "'x'" in output1 and "'y'" in output1

        # Run downstream again without changes
        nb_runner.run_cell(2)

        output2 = nb_runner.get_output(2)
        assert "'x'" in output2 and "'y'" in output2


# ============================================================================
# Group 4: Various Types
# ============================================================================

class TestVariousLoopTypes:
    """Tests for different mutation and iterator types in integration setting."""

    def test_loop_builds_list_of_tuples(self, nb_runner):
        """results.append((k, v)) pattern."""
        nb_runner.create_notebook([
            """pairs = []
for i, name in enumerate(["Alice", "Bob", "Carol"]):
    pairs.append((i, name))""",
            """print(f"Pairs: {pairs}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "Alice" in output and "Bob" in output and "Carol" in output

    def test_loop_builds_nested_dict(self, nb_runner):
        """d[k] = {nested_dict} pattern."""
        nb_runner.create_notebook([
            """registry = {}
for name in ["A", "B"]:
    registry[name] = {"length": len(name), "upper": name.upper()}""",
            """print(f"Registry keys: {sorted(registry.keys())}")
print(f"A entry: {registry['A']}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "'A'" in output and "'B'" in output
        assert "'length': 1" in output or '"length": 1' in output

    def test_loop_counter_as_int(self, nb_runner):
        """Simple counter with += (int augmented assign)."""
        nb_runner.create_notebook([
            """count = 0
for x in range(10):
    count += 1""",
            """print(f"Count: {count}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "Count: 10" in output

    def test_while_loop_in_notebook(self, nb_runner):
        """While loop producing a result used in downstream cell."""
        nb_runner.create_notebook([
            """result = []
i = 0
while i < 5:
    result.append(i * 2)
    i += 1""",
            """print(f"Result: {result}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "[0, 2, 4, 6, 8]" in output

    def test_loop_with_conditional_inside(self, nb_runner):
        """Loop with if/else branching inside."""
        nb_runner.create_notebook([
            """evens = []
odds = []
for x in range(8):
    if x % 2 == 0:
        evens.append(x)
    else:
        odds.append(x)""",
            """print(f"Evens: {evens}")
print(f"Odds: {odds}")"""
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "Evens: [0, 2, 4, 6]" in output
        assert "Odds: [1, 3, 5, 7]" in output
